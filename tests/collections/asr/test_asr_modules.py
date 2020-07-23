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
    def test_AudioToMelSpectrogramPreprocessor(self):
        # Make sure constructor works
        instance1 = modules.AudioToMelSpectrogramPreprocessor(dither=0)
        assert isinstance(instance1, modules.AudioToMelSpectrogramPreprocessor)

        # Make sure forward doesn't throw with expected input
        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=4, high=444, size=[4])
        res = instance1(input_signal=input_signal, length=length)
        assert isinstance(res, tuple)
        assert len(res) == 2

    @pytest.mark.unit
    def test_SpectrogramAugmentationr(self):
        # Make sure constructor works
        instance1 = modules.SpectrogramAugmentation(freq_masks=10, time_masks=3, rect_masks=3)
        assert isinstance(instance1, modules.SpectrogramAugmentation)

        # Make sure forward doesn't throw with expected input
        instance0 = modules.AudioToMelSpectrogramPreprocessor(dither=0)
        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=4, high=444, size=[4])
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
        length = torch.randint(low=4, high=444, size=[4])
        res0 = instance0(input_signal=input_signal, length=length)
        res, new_length = instance1(input_signal=res0[0], length=length)

        assert res.shape == torch.Size([4, 64, audio_length])
        assert all(new_length == torch.tensor([128] * 4))
