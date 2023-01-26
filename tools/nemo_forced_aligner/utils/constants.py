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

import torch

BLANK_TOKEN = "<b>"

SPACE_TOKEN = "<space>"

V_NEGATIVE_NUM = -1e30
V_NEGATIVE_NUM_FP16 = -1e4
V_NEGATIVE_NUM_BF16 = -1e8

torch_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

V_NEGATIVE_NUM_MAP = {
    'fp32': V_NEGATIVE_NUM,
    'fp16': V_NEGATIVE_NUM_FP16,
    'bf16': V_NEGATIVE_NUM_BF16,
}
