# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Tuple
from nemo.core.neural_types.elements import AcousticEncodedRepresentation
from omegaconf.dictconfig import DictConfig

from nemo.collections.asr.parts.submodules.jasper import (
    init_weights, 
    jasper_activations,
)

import torch
from torch import nn
from torch.nn import functional as F

from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType, SpectrogramType, LogprobsType
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule

from omegaconf import DictConfig
from collections import OrderedDict
from nemo.core.classes.common import typecheck

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransposeLast(torch.nn.Module):
    """
    Transposes last dimension. Useful for adding to a sequential block.
    """

    def forward(self, x):
        return x.transpose(-2, -1)


class SamePad(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = kernel_size % 2 == 0

    def forward(self, x):
        if self.remove:
            x = x[:, :, :-1]
        return x


class ConvFeatureEncoder(NeuralModule):
    """
        Converts input raw audio into features for downstream transformer model.
        Uses 1D convolutional blocks with GeLU activation.
    """
    @property
    def input_types(self):
        """Returns definitions of module input ports.
        source:
            0: AxisType(BatchTag)
            1: AxisType(TimeTag)
        length:
            0: AxisType(BatchTag)
        Note: length is in number of samples, not seconds
        """
        return {
            "source": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "length": NeuralType(
                tuple('B'), LengthsType()
            ),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports. 
        For compatibility, processed features are treated as Spectrogram types
        processed_signal:
            0: AxisType(BatchTag)
            1: AxisType(ChannelTag)
            2: AxisType(ProcessedTimeTag)
        processed_signal_length:
            0: AxisType(BatchTag)
        """
        return {
            "processed_signal": NeuralType(('B', 'C', 'T'), SpectrogramType()),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        extractor_mode: str = "layer_norm",
        conv_bias: bool = False,
        feature_grad_mult = 1.0,
        normalize_audio = True
    ):
        super().__init__()

        self.grad_mult = feature_grad_mult
        self.normalize_input = normalize_audio

        def block(
            n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (is_layer_norm and is_group_norm) is False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Sequential(TransposeLast(), nn.LayerNorm(dim, elementwise_affine=True), TransposeLast()),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(make_conv(), nn.GroupNorm(dim, dim, affine=True), nn.GELU(),)
            else:
                return nn.Sequential(make_conv(), nn.GELU())

        in_d = 1
        self.layer_cfg = conv_layers
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=extractor_mode == "layer_norm",
                    is_group_norm=extractor_mode == "group_norm" and i == 0, # Group norm is only for first layer
                    conv_bias=conv_bias,
                )
            )
            in_d = dim
    
    def apply_layers(self, x):
        for conv in self.conv_layers:
            x = conv(x)
        return x
    
    def normalize(self, source, lengths):
        for i in range(lengths.size(0)):
            orig = source[i, :lengths[i]]
            norm = F.layer_norm(orig, orig.shape) # From FAIR
            source[i, :lengths[i]] = norm
        return source

    def forward(self, source, length):
        if self.normalize_input:
            with torch.no_grad(): # Normalizes audio source
                source = self.normalize(source, length)

        # BxT -> BxCxT
        processed_signal = source.unsqueeze(1)

        # Applies grad mult scaling
        if self.grad_mult > 0:
            processed_signal = self.apply_layers(processed_signal)
            if self.grad_mult != 1.0: 
                processed_signal = GradMultiply.apply(processed_signal, self.grad_mult)
        else:
            with torch.no_grad(): # We use 0 to deactivate training of convolutions
                processed_signal = self.apply_layers(processed_signal)
        
        # Feature lengths will have been changed through convolutions
        processed_signal_length = self.get_lengths(audio_lengths=length)

        return processed_signal, processed_signal_length

    def get_lengths(self, audio_lengths): # from hugging face
        # converts audio lengths to timestep lengths
        with torch.no_grad():
            for conv in self.layer_cfg:
                kernel = conv[1]
                stride = conv[2]
                audio_lengths = torch.div(audio_lengths - kernel, stride, rounding_mode='floor') + 1 # from pytorch doc
        return audio_lengths



class Wav2VecTransformerEncoder(NeuralModule):
    def __init__(self, pos_embed: DictConfig, transformer: DictConfig):
        super().__init__()


        # positional convolutional embeddings
        self.pos_conv = nn.Conv1d(
            pos_embed.embedding_dim,
            pos_embed.embedding_dim,
            kernel_size=pos_embed.conv_pos,
            padding=pos_embed.conv_pos // 2,
            groups=pos_embed.conv_pos_groups,
        )

        self.dropout = transformer.dropout # For initializing BERT parameters
        std = math.sqrt((4 * (1.0 - self.dropout)) / (pos_embed.conv_pos * pos_embed.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(pos_embed.conv_pos), nn.GELU())

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=transformer.embedding_dim,
                nhead=transformer.num_attention_heads,
                dim_feedforward=transformer.ffn_embedding_dim,
                dropout=self.dropout,
                activation=transformer.activation_fn,
            ),
            num_layers=transformer.encoder_layers,
        )
        self.layer_norm = nn.LayerNorm(transformer.embedding_dim)
        self.apply(init_bert_params)

    @property
    def input_types(self):
        """Returns definitions of module output ports. 
        We treat features as SpectrogramType for Nemo compatibility
        processed_signal:
            0: AxisType(BatchTag)
            1: AxisType(ChannelTag)
            2: AxisType(ProcessedTimeTag)
        processed_length:
            0: AxisType(BatchTag)
        """
        return {
            "processed_signal": NeuralType(('B', 'C', 'T'), SpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports. 
        We're using SpectrogramType for now to keep things Nemo safe
        processed_signal:
            0: AxisType(BatchTag)
            1: AxisType(ProcessedTimeTag)
            2: AxisType(ChannelTag)
        processed_length:
            0: AxisType(BatchTag)
        """
        return {
            "context_embed": NeuralType(('B', 'T', 'C'), AcousticEncodedRepresentation()),
        }


    def forward(self, signal, length):
        signal = signal.transpose(1,2) #B, C, T -> B, T, C

        padding_mask = self.create_padding_mask(length)
        if padding_mask is not None:
            signal[padding_mask] = 0.0

        signal = signal.transpose(1,2)
        signal_conv = self.pos_conv(signal) # B, C, T
        signal_conv = signal_conv 
        signal += signal_conv

        signal = signal.transpose(1, 2) # B, C, T -> B, T, C
        signal = self.layer_norm(signal)

        signal = signal.transpose(0, 1) # B x T x C -> T x B x C
        context_emb = self.transformer_encoder(signal, src_key_padding_mask=padding_mask)

        # T x B x C -> B x T x C
        context_emb = context_emb.transpose(0, 1)

        return context_emb

    def create_padding_mask(self, length):
        # Broadcast to vectorize creating the padding mask
        max_len = max(length)
        padding_mask = torch.arange(max_len, device=DEVICE)
        padding_mask = padding_mask.expand(len(length), max_len) < length.unsqueeze(1)
        # Negate to false where no padding
        padding_mask = ~padding_mask

        return padding_mask

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bias will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, nn.TransformerEncoderLayer):
        module.self_attn.in_proj_weight.data.normal_(mean=0.0, std=0.02)

class Wav2VecLinearDecoder(NeuralModule, Exportable):
    """Simple ASR Decoder for linear projection of Wav2Vec embeddings
    """

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @property
    def input_types(self):
        return OrderedDict({"encoder_output": NeuralType(('B', 'T', 'D'), AcousticEncodedRepresentation())})

    @property
    def output_types(self):
        return OrderedDict({"logprobs": NeuralType(('B', 'T', 'D'), LogprobsType())})

    def __init__(self, feat_in, num_classes, init_mode="xavier_uniform", vocabulary=None):
        super().__init__()

        if vocabulary is not None:
            if num_classes != len(vocabulary):
                raise ValueError(
                    f"If vocabulary is specified, it's length should be equal to the num_classes. Instead got: num_classes={num_classes} and len(vocabulary)={len(vocabulary)}"
                )
            self.__vocabulary = vocabulary
        self._feat_in = feat_in
        # Add 1 for blank char
        self._num_classes = num_classes + 1

        self.projection = torch.nn.Linear(self._feat_in, self._num_classes, bias=False)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    @typecheck()
    def forward(self, encoder_output):
        return torch.nn.functional.log_softmax(self.projection(encoder_output), dim=-1)

    def input_example(self):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        bs = 8
        seq = 64
        input_example = torch.randn(bs, self._feat_in, seq).to(next(self.parameters()).device)
        return tuple([input_example])

    def _prepare_for_export(self, **kwargs):
        pass

    @property
    def vocabulary(self):
        return self.__vocabulary

    @property
    def num_classes_with_blank(self):
        return self._num_classes

class Wav2VecLinearReconstruction(NeuralModule, Exportable):
    """ ASR Decoder for reconstructing masked regions of spectrogram
    """
    @property
    def input_types(self):
        return OrderedDict({"encoder_output": NeuralType(('B', 'T', 'D'), AcousticEncodedRepresentation())})

    @property
    def output_types(self):
        return OrderedDict({"audio_recon": NeuralType(('B', 'T', 'D'), SpectrogramType())})

    
    def __init__(self, feat_in, feat_out, init_mode="xavier_uniform", activation=None):
        super().__init__()

        self.feat_in = feat_in
        self.feat_out = feat_out

        self.projection = torch.nn.Linear(self.feat_in, self.feat_out, bias=False)

        if activation:
            self.projection = nn.Sequential(self.projection, jasper_activations[activation]())

        self.apply(lambda x: init_weights(x, mode=init_mode))
    
    @typecheck()
    def forward(self, encoder_output):
        return self.projection(encoder_output)