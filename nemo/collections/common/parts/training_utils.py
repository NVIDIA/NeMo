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

from contextlib import nullcontext
import torch

__all__ = ['subsampling_autocast_context', 'mhsa_autocast_context']

def subsampling_autocast_context():
    """
    This is to overcome the current slowness in the conv2D layer when CUDNN
    v8 API is used with bfloat16
    """

    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16:
        return torch.cuda.amp.autocast(dtype=torch.float32)
    else:
        return nullcontext()


def mhsa_autocast_context():
    """
    This is to overcome the current issues in MHSA when 
    float16 is used
    """

    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16:
        if torch.cuda.is_bf16_supported():
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            return torch.cuda.amp.autocast(dtype=torch.float32)
    else:
        return nullcontext()
