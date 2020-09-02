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

from nemo.collections.asr.parts.features import STFTExactPad
from nemo.collections.tts.modules.squeezewave import OperationMode


class SqueezeWaveDenoiser(torch.nn.Module):
    def __init__(self, model, n_mel=80, filter_length=1024, hop_length=256, win_length=1024, window='hann'):
        super().__init__()
        assert hasattr(model, 'squeezewave')

        self.stft = STFTExactPad(
            filter_length=filter_length, hop_length=hop_length, win_length=win_length, window=window,
        ).to(model.device)

        with torch.no_grad():
            spect = torch.zeros((1, n_mel, 88)).to(model.device)
            bias_audio = model.convert_spectrogram_to_audio(spect=spect, sigma=0.0)
            bias_spect, _ = self.stft.transform(bias_audio)
            self.bias_spect = bias_spect[:, :, 0][:, :, None]

        # Reset mode to validation since `model.convert_spectrogram_to_audio` sets it to infer
        model.mode = OperationMode.validation
        model.squeezewave.mode = OperationMode.validation

    def forward(self, audio, strength=0.1):
        audio_spect, audio_angles = self.stft.transform(audio)
        audio_spect_denoised = audio_spect - self.bias_spect * strength
        audio_spect_denoised = torch.clamp(audio_spect_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spect_denoised, audio_angles)
        return audio_denoised
