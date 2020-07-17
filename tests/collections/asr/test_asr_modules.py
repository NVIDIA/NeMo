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

from unittest import TestCase

import pytest
import torch

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor, SpectrogramAugmentation


class ASRModulesBasicTests(TestCase):
    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor(self):
        # Make sure constructor works
        instance1 = AudioToMelSpectrogramPreprocessor(dither=0)
        self.assertTrue(isinstance(instance1, AudioToMelSpectrogramPreprocessor))

        # Make sure forward doesn't throw with expected input
        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=4, high=444, size=[4])
        res = instance1(input_signal=input_signal, length=length)
        self.assertTrue(isinstance(res, tuple))
        self.assertEqual(len(res), 2)

    @pytest.mark.unit
    def test_SpectrogramAugmentationr(self):
        # Make sure constructor works
        instance1 = SpectrogramAugmentation(freq_masks=10, time_masks=3, rect_masks=3)
        self.assertTrue(isinstance(instance1, SpectrogramAugmentation))

        # Make sure forward doesn't throw with expected input
        instance0 = AudioToMelSpectrogramPreprocessor(dither=0)
        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=4, high=444, size=[4])
        res0 = instance0(input_signal=input_signal, length=length)
        res = instance1(input_spec=res0[0])
        self.assertEqual(res.shape, res0[0].shape)
