# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List


def build_loss_mask(input_ids: List[int], answer_start_idx: int, answer_only_loss: bool = True) -> List[float]:
    """Pad input_ids in batch to max batch length while building loss mask"""
    # function borrowed from nemo/collections/nlp/data/language_modelling/megatron/gpt_sft_dataset.py
    if answer_only_loss:
        loss_mask = [float(idx >= answer_start_idx) for idx in range(len(input_ids))]
    else:
        loss_mask = [1.0] * len(input_ids)

    return loss_mask
