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
from enum import Enum

import torch

from nemo.collections.tts.modules.submodules import Invertible1x1Conv, WaveNet
from nemo.core.classes import Exportable, NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    AudioSignal,
    IntType,
    MelSpectrogramType,
    NormalDistributionSamplesType,
    VoidType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils.decorators import experimental


class OperationMode(Enum):
    """Training or Inference (Evaluation) mode"""

    training = 0
    validation = 1
    infer = 2


@experimental  # TODO: Implement save_to() and restore_from()
class WaveGlowModule(NeuralModule, Exportable):
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
        self.n_mel_channels = n_mel_channels
        assert n_group % 2 == 0
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.wavenet = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()
        self.mode = OperationMode.infer

        n_half = n_group // 2

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
    def forward(self, spect, audio=None, run_inverse=True):
        """ TODO
        """
        if self.training and self.mode != OperationMode.training:
            raise ValueError(f"{self} has self.training set to True but self.OperationMode was not set to training")
        if not self.training and self.mode == OperationMode.training:
            raise ValueError(f"{self} has self.training set to False but self.OperationMode was set to training")

        audio_pred = torch.zeros((1, 1))
        if audio is not None and self.mode != OperationMode.infer:
            # audio_to_normal_dist is used to calculate loss so only run this in train or val model
            z, log_s_list, log_det_W_list = self.audio_to_normal_dist(spect=spect, audio=audio)
        if run_inverse:
            # norm_dist_to_audio is used to predict audio from spectrogram so only used in val or infer mode
            # Could also log train audio but currently not done
            audio_pred = self.norm_dist_to_audio(spect=spect)

        # Return the necessary tensors
        if self.mode == OperationMode.training or self.mode == OperationMode.validation:
            return z, log_s_list, log_det_W_list, audio_pred
        return audio_pred

    @property
    def input_types(self):
        return {
            "spect": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "audio": NeuralType(('B', 'T'), AudioSignal(), optional=True),
            "run_inverse": NeuralType(elements_type=IntType(), optional=True),
            # "sigma": NeuralType(elements_type=BoolType(), optional=True),  # TODO: Add to forward
        }

    @property
    def output_types(self):
        if self.mode == OperationMode.training or self.mode == OperationMode.validation:
            return {
                "pred_normal_dist": NeuralType(('B', 'flowgroup', 'T'), NormalDistributionSamplesType()),
                "log_s_list": NeuralType(('B', 'flowgroup', 'T'), VoidType()),  # TODO: Figure out a good typing
                "log_det_W_list": NeuralType(elements_type=VoidType()),  # TODO: Figure out a good typing
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
            n_half = audio.size(1) // 2
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
        # TODO: Implement me!!!
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        # TODO: Implement me!!!
        pass
