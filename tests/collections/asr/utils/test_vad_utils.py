# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import pytest

from nemo.collections.asr.parts.utils.vad_utils import align_labels_to_frames


class TestVADUtils:
    @pytest.mark.parametrize(["logits_len", "labels_len"], [(20, 10), (20, 11), (20, 9), (10, 21), (10, 19)])
    @pytest.mark.unit
    def test_align_label_logits(self, logits_len, labels_len):
        logits = np.arange(logits_len).tolist()
        labels = np.arange(labels_len).tolist()
        labels_new = align_labels_to_frames(probs=logits, labels=labels)

        assert len(labels_new) == len(logits)
