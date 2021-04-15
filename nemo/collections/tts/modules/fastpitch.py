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
#
# BSD 3-Clause License
#
# Copyright (c) 2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.utils.rnn import pad_sequence

from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    EncodedRepresentation,
    Index,
    MaskType,
    MelSpectrogramType,
    RegressionValuesType,
    SequenceToSequenceAlignmentType,
    TokenDurationType,
    TokenIndex,
    TokenLogDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType


def regulate_len(durations, enc_out, pace=1.0, mel_max_len=None):
    """If target=None, then predicted durations are applied"""
    reps = torch.round(durations.float() / pace).long()
    dec_lens = reps.sum(dim=1)

    enc_rep = pad_sequence([torch.repeat_interleave(o, r, dim=0) for o, r in zip(enc_out, reps)], batch_first=True)
    if mel_max_len:
        enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = torch.clamp_max(dec_lens, mel_max_len)
    return enc_rep, dec_lens


class ConvReLUNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0):
        super(ConvReLUNorm, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, signal):
        out = F.relu(self.conv(signal))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return self.dropout(out)


class TemporalPredictor(NeuralModule):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, filter_size, kernel_size, dropout, n_layers=2):
        super(TemporalPredictor, self).__init__()

        self.layers = nn.Sequential(
            *[
                ConvReLUNorm(
                    input_size if i == 0 else filter_size, filter_size, kernel_size=kernel_size, dropout=dropout
                )
                for i in range(n_layers)
            ]
        )
        self.fc = nn.Linear(filter_size, 1, bias=True)

    @property
    def input_types(self):
        return {
            "enc": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "enc_mask": NeuralType(('B', 'T', 1), TokenDurationType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T'), EncodedRepresentation()),
        }

    def forward(self, enc, enc_mask):
        out = enc * enc_mask
        out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(out) * enc_mask
        return out.squeeze(-1)


class FastPitchModule(NeuralModule):
    def __init__(
        self,
        encoder_module: NeuralModule,
        decoder_module: NeuralModule,
        duration_predictor: NeuralModule,
        pitch_predictor: NeuralModule,
        n_speakers: int,
        symbols_embedding_dim: int,
        pitch_embedding_kernel_size: int,
        n_mel_channels: int = 80,
        max_token_duration: int = 75,
    ):
        super().__init__()

        self.encoder = encoder_module
        self.decoder = decoder_module
        self.duration_predictor = duration_predictor
        self.pitch_predictor = pitch_predictor

        if n_speakers > 1:
            self.speaker_emb = nn.Embedding(n_speakers, symbols_embedding_dim)
        else:
            self.speaker_emb = None

        self.max_token_duration = max_token_duration

        self.pitch_emb = nn.Conv1d(
            1,
            symbols_embedding_dim,
            kernel_size=pitch_embedding_kernel_size,
            padding=int((pitch_embedding_kernel_size - 1) / 2),
        )

        # Store values precomputed from training data for convenience
        self.register_buffer('pitch_mean', torch.zeros(1))
        self.register_buffer('pitch_std', torch.zeros(1))

        self.proj = nn.Linear(self.decoder.d_model, n_mel_channels, bias=True)

    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T'), TokenIndex()),
            "durs": NeuralType(('B', 'T'), TokenDurationType()),
            "pitch": NeuralType(('B', 'T'), RegressionValuesType()),
            "speaker": NeuralType(('B'), Index()),
            "pace": NeuralType(optional=True),
        }

    @property
    def output_types(self):
        return {
            "spect": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spect_lens": NeuralType(('B'), SequenceToSequenceAlignmentType()),
            "spect_mask": NeuralType(('B', 'D', 'T'), MaskType()),
            "durs_predicted": NeuralType(('B', 'T'), TokenDurationType()),
            "log_durs_predicted": NeuralType(('B', 'T'), TokenLogDurationType()),
            "pitch_predicted": NeuralType(('B', 'T'), RegressionValuesType()),
        }

    @typecheck()
    def forward(self, *, text, durs=None, pitch=None, speaker=None, pace=1.0):

        if self.training:
            assert durs is not None
            assert pitch is not None

        # Calculate speaker embedding
        if self.speaker_emb is None or speaker is None:
            spk_emb = 0
        else:
            # if type(speaker) is int:
            #     speaker = torch.zeros_like(text[:, 0]).fill_(speaker)
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)

        # Input FFT
        enc_out, enc_mask = self.encoder(input=text, conditioning=spk_emb)

        # Embedded for predictors
        pred_enc_out, pred_enc_mask = enc_out, enc_mask

        # Predict durations
        log_durs_predicted = self.duration_predictor(pred_enc_out, pred_enc_mask)
        durs_predicted = torch.clamp(torch.exp(log_durs_predicted) - 1, 0, self.max_token_duration)

        # Predict pitch
        pitch_predicted = self.pitch_predictor(enc_out, enc_mask)
        if pitch is None:
            pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))
        else:
            pitch_emb = self.pitch_emb(pitch.unsqueeze(1))
        enc_out = enc_out + pitch_emb.transpose(1, 2)

        if durs is None:
            len_regulated, dec_lens = regulate_len(durs_predicted, enc_out, pace)
        else:
            len_regulated, dec_lens = regulate_len(durs, enc_out, pace)

        # Output FFT
        dec_out, dec_mask = self.decoder(input=len_regulated, seq_lens=dec_lens)
        spect = self.proj(dec_out)
        return (spect, dec_lens, dec_mask, durs_predicted, log_durs_predicted, pitch_predicted)
