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

# MIT License
#
# Copyright (c) 2020 Tianren Gao, Bohan Zhai, Flora Xue,
# Daniel Rothchild, Bichen Wu, Joseph E. Gonzalez, Kurt Keutzer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

from nemo.collections.tts.helpers.helpers import OperationMode, split_view
from nemo.collections.tts.modules.squeezewave_submodules import SqueezeWaveNet
from nemo.collections.tts.modules.submodules import Invertible1x1Conv
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    AudioSignal,
    IntType,
    MelSpectrogramType,
    NormalDistributionSamplesType,
    VoidType,
)
from nemo.core.neural_types.neural_type import NeuralType


class SqueezeWaveModule(NeuralModule):
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
        SqueezeWave module

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
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.wavenet = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()
        self.mode = OperationMode.infer

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
                SqueezeWaveNet(
                    n_half, n_mel_channels, n_layers=n_wn_layers, n_channels=n_wn_channels, kernel_size=wn_kernel_size,
                )
            )
        self.n_remaining_channels = n_remaining_channels

    @typecheck()
    def forward(self, *, spec, audio=None, run_inverse=True, sigma=1.0):
        """ TODO
        """
        if self.training and self.mode != OperationMode.training:
            raise ValueError(f"{self} has self.training set to True but self.OperationMode was not set to training")
        if not self.training and self.mode == OperationMode.training:
            raise ValueError(f"{self} has self.training set to False but self.OperationMode was set to training")

        audio_pred = torch.zeros((1, 1))
        if audio is not None and self.mode != OperationMode.infer:
            # audio_to_normal_dist is used to calculate loss so only run this in train or val model
            z, log_s_list, log_det_W_list = self.audio_to_normal_dist(spec=spec, audio=audio)
        if run_inverse:
            # norm_dist_to_audio is used to predict audio from spectrogram so only used in val or infer mode
            # Could also log train audio but currently not done
            audio_pred = self.norm_dist_to_audio(spec=spec, sigma=sigma)

        # Return the necessary tensors
        if self.mode == OperationMode.training or self.mode == OperationMode.validation:
            return z, log_s_list, log_det_W_list, audio_pred
        return audio_pred

    @property
    def input_types(self):
        return {
            "spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "audio": NeuralType(('B', 'T'), AudioSignal(), optional=True),
            "run_inverse": NeuralType(elements_type=IntType(), optional=True),
            "sigma": NeuralType(optional=True),
        }

    @property
    def output_types(self):
        if self.mode == OperationMode.training or self.mode == OperationMode.validation:
            return {
                "pred_normal_dist": NeuralType(('B', 'flowgroup', 'T'), NormalDistributionSamplesType()),
                "log_s_list": [NeuralType(('B', 'flowgroup', 'T'), VoidType())],  # TODO: Figure out a good typing
                "log_det_W_list": [NeuralType(elements_type=VoidType())],  # TODO: Figure out a good typing
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

    def audio_to_normal_dist(self, *, spec: torch.Tensor, audio: torch.Tensor) -> (torch.Tensor, list, list):
        audio = split_view(audio, self.n_group, 1).permute(0, 2, 1)
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

            output = self.wavenet[k]((audio_0, spec))
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s) * audio_1 + b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list

    def norm_dist_to_audio(self, *, spec, sigma: float = 1.0):
        # Note: hard-coded 256 is hop_length for computing mel-spectrogram
        l = 256 * spec.size(2) // self.n_group
        audio = sigma * torch.randn(spec.size(0), self.n_remaining_channels, l, device=spec.device).to(spec.dtype)

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.wavenet[k]((audio_0, spec))
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat((audio_0, audio_1), 1)

            audio = self.convinv[k](audio, reverse=True)
            if k % self.n_early_every == 0 and k > 0:
                z = sigma * torch.randn(spec.size(0), self.n_early_size, l, device=spec.device).to(spec.dtype)
                audio = torch.cat((z, audio), 1)
        return audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1)
