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
#
import torch


def is_bfloat16_available():
    """ Helper function to determine if bfloat16 precision is available
    """
    if torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
        if device_capability is not None and device_capability[0] >= 8:
            return True
        else:
            return False
    else:
        return False

def get_current_precision():
    """ Determine current precision set by the trainer
    """
    if torch.cuda.is_available():
        a = torch.rand((1,1), device="cuda")
        b = torch.rand((1,1), device="cuda")
        # if trainer set autocast, this will not be torch.float32
        test = torch.mm(a,b)
        return test.dtype
    else:
        return None
