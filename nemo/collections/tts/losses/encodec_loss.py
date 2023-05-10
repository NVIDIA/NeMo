# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import List

import torch
import torch.nn.functional as F

from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
from nemo.collections.tts.losses.loss import MaskedLoss
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import AudioSignal, LengthsType, LossType, NeuralType, VoidType


class MultiResolutionMelLoss(Loss):
    def __init__(
        self, sample_rate: int, mel_dim: int, resolutions: List[List], l1_scale: float = 1.0, l2_scale: float = 1.0
    ):
        super(MultiResolutionMelLoss, self).__init__()

        self.l1_loss_fn = MaskedLoss("l1", loss_scale=l1_scale)
        self.l2_loss_fn = MaskedLoss("l2", loss_scale=l2_scale)

        self.mel_features = torch.nn.ModuleList()
        for n_fft, hop_len, win_len in resolutions:
            mel_feature = FilterbankFeatures(
                sample_rate=sample_rate,
                nfilt=mel_dim,
                n_window_size=win_len,
                n_window_stride=hop_len,
                n_fft=n_fft,
                pad_to=1,
                mag_power=1.0,
                log_zero_guard_type="add",
                log_zero_guard_value=1.0,
                mel_norm=None,
                normalize=None,
                preemph=None,
                dither=0.0,
                use_grads=True,
            )
            self.mel_features.append(mel_feature)

    @property
    def input_types(self):
        return {
            "audio_real": NeuralType(('B', 'T'), AudioSignal()),
            "audio_gen": NeuralType(('B', 'T'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": [NeuralType(elements_type=LossType())],
        }

    @typecheck()
    def forward(self, audio_real, audio_gen, audio_len):
        len_diff = audio_real.shape[1] - audio_gen.shape[1]
        audio_gen = F.pad(audio_gen, (0, len_diff))

        loss = 0.0
        for mel_feature in self.mel_features:
            mel_real, mel_real_len = mel_feature(x=audio_real, seq_len=audio_len)
            mel_gen, _ = mel_feature(x=audio_gen, seq_len=audio_len)
            loss = loss + self.l1_loss_fn(predicted=mel_gen, target=mel_real, target_len=mel_real_len)
            loss = loss + self.l2_loss_fn(predicted=mel_gen, target=mel_real, target_len=mel_real_len)

        loss /= len(self.mel_features)

        return loss


class FeatureMatchingLoss(Loss):
    @property
    def input_types(self):
        return {
            "fmaps_real": [[NeuralType(elements_type=VoidType())]],
            "fmaps_gen": [[NeuralType(elements_type=VoidType())]],
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, fmaps_real, fmaps_gen):
        loss = 0.0
        for fmap_real, fmap_gen in zip(fmaps_real, fmaps_gen):
            # [B, ..., time]
            for feat_real, feat_gen in zip(fmap_real, fmap_gen):
                # [B, ...]
                feat_mean = torch.mean(torch.abs(feat_real), dim=-1)
                diff = torch.mean(torch.abs(feat_real - feat_gen), dim=-1)
                feat_loss = diff / (feat_mean + 1e-2)
                # [1]
                feat_loss = torch.mean(feat_loss) / len(fmap_real)
                loss = loss + feat_loss

        loss /= len(fmaps_real)

        return loss


class GeneratorLoss(Loss):
    @property
    def input_types(self):
        return {
            "disc_scores": [NeuralType(('B', 'C', 'T'), VoidType())],
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, disc_scores):
        loss = 0.0
        for disc_score in disc_scores:
            loss = loss + torch.mean((1 - disc_score) ** 2)

        loss /= len(disc_scores)

        return loss


class DiscriminatorLoss(Loss):
    @property
    def input_types(self):
        return {
            "disc_scores_real": [NeuralType(('B', 'C', 'T'), VoidType())],
            "disc_scores_gen": [NeuralType(('B', 'C', 'T'), VoidType())],
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, disc_scores_real, disc_scores_gen):
        loss = 0.0
        for disc_score_real, disc_score_gen in zip(disc_scores_real, disc_scores_gen):
            loss_real = torch.mean((1 - disc_score_real) ** 2)
            loss_gen = torch.mean(disc_score_gen ** 2)
            loss = loss + (loss_real + loss_gen) / 2

        loss /= len(disc_scores_real)

        return loss
