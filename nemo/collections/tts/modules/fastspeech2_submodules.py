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
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.collections.tts.modules.transformer import PositionalEmbedding, TransformerLayer


class FFTransformer(nn.Module):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropatt,
        dropemb=0.0,
        pre_lnorm=False,
        embed_input=True,
        n_embed=84,
        padding_idx=83,
    ):
        """
        Feed-Forward Transformer submodule for FastSpeech 2 that consists of multiple TransformerLayers.
        """
        super(FFTransformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.padding_idx = padding_idx

        if embed_input:
            self.word_emb = nn.Embedding(n_embed, d_model, padding_idx=self.padding_idx)
        else:
            self.word_emb = None

        self.pos_emb = PositionalEmbedding(self.d_model)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()

        for _ in range(n_layer):
            self.layers.append(
                TransformerLayer(
                    n_head, d_model, d_head, d_inner, kernel_size, dropout, dropatt=dropatt, pre_lnorm=pre_lnorm
                )
            )

    def forward(self, dec_inp, seq_lens=None):
        if self.word_emb is None:
            inp = dec_inp
            mask = get_mask_from_lengths(seq_lens).unsqueeze(2)
        else:
            inp = self.word_emb(dec_inp)
            # [bsz x L x 1]
            mask = (dec_inp != self.padding_idx).unsqueeze(2)

        pos_seq = torch.arange(inp.size(1), device=inp.device, dtype=inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask
        out = self.drop(inp + pos_emb)

        for layer in self.layers:
            out = layer(out, mask=mask)

        # out = self.drop(out)
        return out, mask


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class VariancePredictor(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout):
        """
        Variance predictor submodule for FastSpeech 2/, used for pitch and energy prediction.

        Args:
            d_model: Input dimension.
            d_inner: Hidden dimension of the variance predictor.
            kernel_size: Conv1d kernel size.
            dropout: Dropout value for the variance predictor.
        """
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.kernel_size = kernel_size

        self.layers = nn.Sequential(
            Transpose(),
            nn.Conv1d(d_model, d_inner, kernel_size, stride=1, padding=(kernel_size // 2)),
            nn.ReLU(),
            Transpose(),
            nn.LayerNorm(d_inner),
            Transpose(),
            nn.Dropout(dropout),
            nn.Conv1d(d_inner, d_inner, kernel_size, stride=1, padding=(kernel_size // 2)),
            nn.ReLU(),
            Transpose(),
            nn.LayerNorm(d_inner),
            nn.Dropout(dropout),
            nn.Linear(d_inner, 1),
        )

    def forward(self, vp_input):
        return self.layers(vp_input).squeeze(-1)


class LengthRegulator(nn.Module):
    def forward(self, x, dur):
        """
        Expands the hidden states according to the duration target/prediction (depends on train vs inference).
        For frame i of a single batch element x, repeats each the frame dur[i] times.

        Args:
            x: Hidden states of dimension (batch, time, emb_dim)
            dur: Timings for each frame of the hiddens, dimension (batch, time)
        """
        output = []
        for x_i, dur_i in zip(x, dur):
            expanded = self.expand(x_i, dur_i)
            output.append(expanded)
        output = self.pad(output)
        return output

    def expand(self, x, dur):
        output = []
        for i, frame in enumerate(x):
            expanded_len = int(dur[i] + 0.5)
            expanded = frame.expand(expanded_len, -1)
            output.append(expanded)
        output = torch.cat(output, 0)
        return output

    def pad(self, x):
        output = []
        max_len = max([x[i].size(0) for i in range(len(x))])
        for i, seq in enumerate(x):
            padded = F.pad(seq, [0, 0, 0, max_len - seq.size(0)], 'constant', 0.0)
            output.append(padded)
        output = torch.stack(output)
        return output
