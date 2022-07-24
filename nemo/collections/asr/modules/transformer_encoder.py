# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections import OrderedDict

import torch
import torch.distributed
import torch.nn as nn

from nemo.collections.asr.parts.submodules.multi_head_attention import PositionalEncoding, RelPositionalEncoding
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling
from nemo.collections.asr.parts.submodules.transformer_modules import TransformerEncoderLayer
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType

__all__ = ['TransformerEncoder']


class TransformerEncoder(NeuralModule, Exportable):
    """
    Implmentation for trasformer encoder.

    Args:
        feat_in (int): the size of feature channels
        n_layers (int): number of layers of ConformerBlock
        d_model (int): the hidden size of the model
        ff_expansion_factor (int): the expansion factor in feed forward layers
            Defaults to 4.
        self_attention_model (str): type of the attention layer and positional encoding
            'rel_pos': relative positional embedding and Transformer-XL
            'abs_pos': absolute positional embedding and Transformer
            default is rel_pos.
        pos_emb_max_len (int): the maximum length of positional embeddings
            Defaults to 20000
        n_heads (int): number of heads in multi-headed attention layers
            Defaults to 1.
        xscaling (bool): enables scaling the inputs to the multi-headed attention layers by sqrt(d_model)
            Defaults to True.
        untie_biases (bool): whether to not share (untie) the bias weights between layers of Transformer-XL
            Defaults to True.
        dropout (float): the dropout rate used in all layers except the attention layers
            Defaults to 0.0.
        dropout_emb (float): the dropout rate used for the positional embeddings
            Defaults to 0.0.
        dropout_att (float): the dropout rate used for the attention layer
            Defaults to 0.0.
    """

    def __init__(
        self,
        n_layers,
        d_model,
        ff_expansion_factor=4,
        self_attention_model='abs_pos',
        n_heads=4,
        att_context_size=None,
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        dropout=0.0,
        dropout_emb=0.0,
        dropout_ff=0.0,
        dropout_att=0.0,
        pre_norm=False,
        pre_norm_final_layer_norm=True,
    ):
        super().__init__()

        if pre_norm and pre_norm_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(d_model)
        else:
            self.final_layer_norm = None

        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self.scale = math.sqrt(self.d_model)
        if att_context_size:
            self.att_context_size = att_context_size
        else:
            self.att_context_size = [-1, -1]

        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None

        if not untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
            nn.init.zeros_(pos_bias_u)
            nn.init.zeros_(pos_bias_v)
        else:
            pos_bias_u = None
            pos_bias_v = None

        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            pos_bias_u = None
            pos_bias_v = None
            self.pos_enc = PositionalEncoding(
                d_model=d_model, dropout_rate=dropout, max_len=pos_emb_max_len, xscale=self.xscale
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = TransformerEncoderLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                n_heads=n_heads,
                dropout=dropout,
                dropout_ff=dropout_ff,
                dropout_att=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
            )
            self.layers.append(layer)

        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True

    def set_max_audio_length(self, max_audio_length):
        """
        Sets maximum input length
        """
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        seq_range = torch.arange(0, self.max_audio_length, device=device)
        if hasattr(self, 'seq_range'):
            self.seq_range = seq_range
        else:
            self.register_buffer('seq_range', seq_range, persistent=False)
        self.pos_enc.extend_pe(max_audio_length, device)

    def update_max_seq_length(self, seq_length, device):
        # Find global max audio length across all nodes
        if torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)

            # update across all ranks
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)
            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def forward(self, x, x_mask=None):
        """
        Args:
            x: transformer layer input [B, L, H]
            x_mask: input mask [B, L]
                    1's for valid and 0's for padding tokens
        """
        # update seq lengths across nodes
        self.update_max_seq_length(seq_length=x.size(1), device=x.device)

        if x_mask is None:
            x_mask = torch.ones((x.shape[0], x.shape[1]), dtype=x.dtype, layout=x.layout, device=x.device)
        att_mask = self.form_attention_mask(x_mask)

        x, pos_emb = self.pos_enc(x)

        for lth, layer in enumerate(self.layers):
            x = layer(x=x, att_mask=att_mask)

        # final norm after encoder layers
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        return x

    def form_attention_mask(self, input_mask, diagonal=None):
        """
        Build attention mask with optional masking of future tokens we forbid
        to attend to (e.g. as it is in Transformer decoder).

        Args:
            input_mask: binary mask of size B x L with 1s corresponding to valid
                tokens and 0s corresponding to padding tokens
            diagonal: diagonal where triangular future mask starts
                None -- do not mask anything
                0 -- regular translation or language modeling future masking
                1 -- query stream masking as in XLNet architecture
        Returns:
            attention_mask: mask of size B x 1 x L x L with 0s corresponding to
                tokens we plan to attend to and -10000 otherwise
        """

        if input_mask is None:
            return None
        att_mask = input_mask.to(dtype=bool).unsqueeze(1).repeat([1, input_mask.shape[1], 1])
        att_mask = torch.logical_and(att_mask, att_mask.transpose(1, 2))

        # negate the att_mask for using in masked_fill
        att_mask = ~att_mask
        return att_mask
