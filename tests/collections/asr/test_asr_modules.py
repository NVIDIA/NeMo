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

import pytest
import torch

from nemo.collections.asr import modules


class TestASRModulesBasicTests:
    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor1(self):
        # Test 1 that should test the pure stft implementation as much as possible
        instance1 = modules.AudioToMelSpectrogramPreprocessor(
            dither=0, stft_conv=False, mag_power=1.0, normalize=False, preemph=0.0, log=False, pad_to=0
        )
        instance2 = modules.AudioToMelSpectrogramPreprocessor(
            dither=0, stft_conv=True, mag_power=1.0, normalize=False, preemph=0.0, log=False, pad_to=0
        )

        # Ensure that the two functions behave similarily
        for _ in range(10):
            input_signal = torch.randn(size=(4, 512))
            length = torch.randint(low=161, high=500, size=[4])
            res1, length1 = instance1(input_signal=input_signal, length=length)
            res2, length2 = instance2(input_signal=input_signal, length=length)
            for len1, len2 in zip(length1, length2):
                assert len1 == len2
            assert res1.shape == res2.shape
            diff = torch.mean(torch.abs(res1 - res2))
            assert diff <= 1e-3
            diff = torch.max(torch.abs(res1 - res2))
            assert diff <= 1e-2

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor2(self):
        # Test 2 that should test the stft implementation as used in ASR models
        instance1 = modules.AudioToMelSpectrogramPreprocessor(dither=0, stft_conv=False)
        instance2 = modules.AudioToMelSpectrogramPreprocessor(dither=0, stft_conv=True)

        # Ensure that the two functions behave similarily
        for _ in range(5):
            input_signal = torch.randn(size=(4, 512))
            length = torch.randint(low=161, high=500, size=[4])
            res1, length1 = instance1(input_signal=input_signal, length=length)
            res2, length2 = instance2(input_signal=input_signal, length=length)
            for len1, len2 in zip(length1, length2):
                assert len1 == len2
            assert res1.shape == res2.shape
            diff = torch.mean(torch.abs(res1 - res2))
            assert diff <= 3e-3
            diff = torch.max(torch.abs(res1 - res2))
            assert diff <= 2

    @pytest.mark.unit
    def test_SpectrogramAugmentationr(self):
        # Make sure constructor works
        instance1 = modules.SpectrogramAugmentation(freq_masks=10, time_masks=3, rect_masks=3)
        assert isinstance(instance1, modules.SpectrogramAugmentation)

        # Make sure forward doesn't throw with expected input
        instance0 = modules.AudioToMelSpectrogramPreprocessor(dither=0)
        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=161, high=500, size=[4])
        res0 = instance0(input_signal=input_signal, length=length)
        res = instance1(input_spec=res0[0])

        assert res.shape == res0[0].shape

    @pytest.mark.unit
    def test_CropOrPadSpectrogramAugmentation(self):
        # Make sure constructor works
        audio_length = 128
        instance1 = modules.CropOrPadSpectrogramAugmentation(audio_length=audio_length)
        assert isinstance(instance1, modules.CropOrPadSpectrogramAugmentation)

        # Make sure forward doesn't throw with expected input
        instance0 = modules.AudioToMelSpectrogramPreprocessor(dither=0)
        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=161, high=500, size=[4])
        res0 = instance0(input_signal=input_signal, length=length)
        res, new_length = instance1(input_signal=res0[0], length=length)

        assert res.shape == torch.Size([4, 64, audio_length])
        assert all(new_length == torch.tensor([128] * 4))
