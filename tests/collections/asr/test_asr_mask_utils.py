# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.parts.submodules.tdnn_attention import lens_to_mask


class TestASRMaskUtils:
    @pytest.mark.unit
    def test_lens_to_mask(self):
        lens = torch.tensor([1, 2, 3], dtype=torch.int)
        max_lens = [1, 2, 3]
        expected_outputs = [
            (torch.tensor([[True], [True], [True]]).reshape(3, 1, 1), torch.tensor([1, 1, 1]).reshape(3, 1, 1)),
            (
                torch.tensor([[True, False], [True, True], [True, True]]).reshape(3, 1, 2),
                torch.tensor([1, 2, 2]).reshape(3, 1, 1),
            ),
            (
                torch.tensor([[True, False, False], [True, True, False], [True, True, True]]).reshape(3, 1, 3),
                torch.tensor([1, 2, 3]).reshape(3, 1, 1),
            ),
        ]

        for i, max_len in enumerate(max_lens):
            mask, num_items = lens_to_mask(lens, max_len)
            expected_mask, expected_num_items = expected_outputs[i]
            assert torch.equal(mask, expected_mask)
            assert torch.equal(num_items, expected_num_items)
