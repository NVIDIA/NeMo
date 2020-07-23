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

from typing import Tuple

import torch

from nemo.core.neural_types.neural_type import NeuralType
from nemo.core.neural_types.elements import AudioSignal, MelSpectrogramType, NormalDistributionSamplesType, VoidType
from nemo.collections.tts.modules.submodules import Invertible1x1Conv, WaveNet
from nemo.core.classes import NeuralModule, typecheck
from nemo.utils.decorators import experimental


@experimental  # TODO: Need to implement abstratct methods: save_to, restore_from, but how?
class WaveGlow(NeuralModule):
    def __init__(
        self,
        n_mel_channels: int,
        n_flows: int,
        n_group: int,
        n_early_every: int,
        n_early_size: int,
        n_wn_channels: int,
        n_wn_layers: int,
        wn_kernel_size: int,
    ):
        """
        WaveGlow module

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

        self.upsample = torch.nn.ConvTranspose1d(n_mel_channels, n_mel_channels, 1024, stride=256)
        assert n_group % 2 == 0
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.wavenet = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()

        n_half = int(n_group / 2)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size / 2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.wavenet.append(
                WaveNet(
                    n_half,
                    n_mel_channels * n_group,
                    n_layers=n_wn_layers,
                    n_channels=n_wn_channels,
                    kernel_size=wn_kernel_size,
                )
            )
        self.n_remaining_channels = n_remaining_channels

    @typecheck()
    def forward(self, *args, **kwargs):
        """ TODO
        """
        if self.training:
            return self.audio_to_normal_dist(**kwargs)
        return self.norm_dist_to_audio(**kwargs)

    @property
    def input_types(self):
        if self.training:
            return {
                "spect": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
                "audio": NeuralType(('B', 'T'), AudioSignal()),
            }
        else:
            return {
                "spect": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
                # "sigma": NeuralType(tuple('B'), LengthsType()),  # TODO: This is optional, how can I express this?
            }

    @property
    def output_types(self):
        if self.training:
            return {
                "pred_normal_dist": NeuralType(('B', 'T'), NormalDistributionSamplesType()),
                "log_s_list": NeuralType(('B'), VoidType()),  # TODO: Figure out a good typing
                "log_det_W_list": NeuralType(('B'), VoidType()),  # TODO: Figure out a good typing
            }
        else:
            return {
                "audio": NeuralType(('B', 'T'), AudioSignal()),
            }

    def audio_to_normal_dist(self, *, spect: torch.Tensor, audio: torch.Tensor) -> (torch.Tensor, list, list):
        #  Upsample spectrogram to size of audio
        spect = self.upsample(spect)
        assert spect.size(2) >= audio.size(1)
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, : audio.size(1)]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1)
        spect = spect.permute(0, 2, 1)

        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        output_audio = []
        log_s_list = []
        log_det_W_list = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, : self.n_early_size, :])
                audio = audio[:, self.n_early_size :, :]

            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.wavenet[k]((audio_0, spect))
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s) * audio_1 + b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list

    def norm_dist_to_audio(self, *, spect, sigma: float = 1.0):
        spect = self.upsample(spect)
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1)
        spect = spect.permute(0, 2, 1)

        audio = sigma * torch.randn(spect.size(0), self.n_remaining_channels, spect.size(2), device=spect.device).to(
            spect.dtype
        )

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.wavenet[k]((audio_0, spect))
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat((audio_0, audio_1), 1)

            audio = self.convinv[k](audio, reverse=True)
            if k % self.n_early_every == 0 and k > 0:
                z = sigma * torch.randn(spect.size(0), self.n_early_size, spect.size(2), device=spect.device).to(
                    spect.dtype
                )
                audio = torch.cat((z, audio), 1)
        return audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1)

    def save_to(self, save_path: str):
        """TODO: Implement"""
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """TODO: Implement"""
        pass
