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

import torch
import torch.nn.functional as F

from nemo.collections.tts.helpers.helpers import OperationMode, remove
from nemo.collections.tts.modules.submodules import Invertible1x1Conv, WaveNet
from nemo.core.classes import Exportable, NeuralModule, typecheck
from nemo.core.neural_types.elements import AudioSignal, MelSpectrogramType, NormalDistributionSamplesType, VoidType
from nemo.core.neural_types.neural_type import NeuralType


class UniGlowModule(NeuralModule, Exportable):
    def __init__(
        self,
        n_mel_channels: int,
        n_flows: int,
        n_group: int,
        n_wn_channels: int,
        n_wn_layers: int,
        wn_kernel_size: int,
        upsample_factor: int,
    ):
        """
        UniGlow module
        Args:
            n_mel_channels (int): Number of mel channels to output.
            n_flows (int): Number of flow layers
            n_group (int): Number of groups to respace the inputs
            n_early_every (int): Every n_early_every layers, n_early_size gets skip connected to the output
            n_early_size (int): The size of the chunk to be skip connected
            n_wn_channels (int): Number of channels for the non-invertible wavenet transformation
            n_wn_layers (int): Number of layers for the non-invertible wavenet transformation
            wn_kernel_size (int): Kernel size for the non-invertible wavenet transformation
        """
        super().__init__()

        assert n_group % 2 == 0
        self.n_flows = n_flows
        self.n_group = n_group
        n_half = int(n_group / 2)

        self.conv = Invertible1x1Conv(n_group)
        self.wn = WaveNet(n_half, n_mel_channels, n_wn_layers, n_wn_channels, wn_kernel_size)
        self.upsample_factor = upsample_factor
        self.mode = OperationMode.infer

    @typecheck()
    def forward(self, spec, audio=None, sigma=1.0):
        """
        Args:
            spec: the melspectrogram tensor [B,F,Tfd]
            audio: the time-domain audio tensor [B,Ttd]
            sigma: the standard deviation of the latent variable z
        Returns:
            training/validation:
                z: latent variable
                lodget: the sum of log-determinants
                audio_pred: time-domain signal predicted from (spec,z') where
                    z' is a latent variable sampled from a N(0,sigma)
            inference:
                audio_pred: time-domain signal predicted from (spec,z') where
                    z' is a latent variable sampled from a N(0,sigma)
        """
        if self.training and self.mode != OperationMode.training:
            raise ValueError(f"{self} has self.training set to True but self.OperationMode was not set to training")
        if not self.training and self.mode == OperationMode.training:
            raise ValueError(f"{self} has self.training set to False but self.OperationMode was set to training")

        if audio is not None and self.mode != OperationMode.infer:
            # audio_to_normal_dist is used to calculate loss so only run this in train or val model
            z, logdet = self.audio_to_normal_dist(spec=spec, audio=audio)
        # audio_pred is used to calculate the stft-loss and when running inference
        audio_pred = self.norm_dist_to_audio(spec=spec, sigma=sigma)

        # Return the necessary tensors
        if self.mode == OperationMode.training or self.mode == OperationMode.validation:
            return z, logdet, audio_pred
        return audio_pred

    @property
    def input_types(self):
        return {
            "spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "audio": NeuralType(('B', 'T'), AudioSignal(), optional=True),
            "sigma": NeuralType(optional=True),
        }

    @property
    def output_types(self):
        if self.mode == OperationMode.training or self.mode == OperationMode.validation:
            return {
                "pred_normal_dist": NeuralType(('B', 'flowgroup', 'T'), NormalDistributionSamplesType()),
                "logdet": NeuralType(elements_type=VoidType()),
                "audio_pred": NeuralType(('B', 'T'), AudioSignal()),
            }
        else:
            return {
                "audio": NeuralType(('B', 'T'), AudioSignal()),
            }

    def input_example(self):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        par = next(self.parameters())
        mel = torch.randn((1, self.n_mel_channels, 96), device=par.device, dtype=par.dtype)
        return tuple([mel])

    def audio_to_normal_dist(self, *, spec: torch.Tensor, audio: torch.Tensor) -> (torch.Tensor, float):
        logdet = 0

        spec = spec[:, :, :-1]
        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        if spec.size(2) != audio.size(2):
            spec = F.interpolate(spec, size=audio.size(2))

        for _ in range(self.n_flows):
            audio, log_det_W = self.conv(audio)
            logdet += log_det_W

            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.wn((audio_0, spec))
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s) * audio_1 + b
            logdet += torch.sum(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        return audio, logdet

    def norm_dist_to_audio(self, *, spec, sigma: float = 1.0):
        spec = spec[:, :, :-1]
        audio_len = spec.shape[2] * self.upsample_factor
        spec = F.interpolate(spec, size=audio_len)
        audio = sigma * torch.randn(spec.size(0), self.n_group, audio_len, device=spec.device).to(spec.dtype)

        for _ in reversed(range(self.n_flows)):
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.wn((audio_0, spec))

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.conv(audio, reverse=True)

        audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1)
        return audio

    def remove_weightnorm(self):
        self.wn.start = torch.nn.utils.remove_weight_norm(self.wn.start)
        self.wn.in_layers = remove(self.wn.in_layers)
        self.wn.cond_layer = torch.nn.utils.remove_weight_norm(self.wn.cond_layer)
        self.wn.res_skip_layers = remove(self.wn.res_skip_layers)
