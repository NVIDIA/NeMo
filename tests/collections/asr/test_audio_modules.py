# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import warnings

import numpy as np
import pytest
import torch

from nemo.collections.asr.modules.audio_modules import AudioToSpectrogram, SpectrogramToAudio

try:
    import torchaudio

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False
HAVE_TORCHAUDIO = False


class TestAudioSpectrogram:
    @pytest.mark.unit
    @pytest.mark.parametrize('fft_length', [128, 1024])
    @pytest.mark.parametrize('num_channels', [1, 4])
    def test_audio_to_spectrogram_reconstruction(self, fft_length: int, num_channels: int):
        """Test analysis and synthesis transform lead to a perfect reconstruction.
        """
        if not HAVE_TORCHAUDIO:
            warnings.warn('Could not import torchaudio. This test will be skipped.')
            return

        atol = 1e-6
        batch_size = 8
        num_samples = fft_length * 50
        num_examples = 10
        random_seed = 42

        _rng = np.random.default_rng(seed=random_seed)

        hop_length = fft_length // 4
        a2s = AudioToSpectrogram(fft_length=fft_length, hop_length=hop_length)
        s2a = SpectrogramToAudio(fft_length=fft_length, hop_length=hop_length)

        for n in range(num_examples):
            x = _rng.normal(size=(batch_size, num_samples, num_channels))

            x_hat = s2a(input=a2s(input=torch.Tensor(x)))

            assert np.isclose(
                x_hat.cpu().detach().numpy(), x, atol=atol
            ).all(), f'Reconstructed not matching for example {n}'
