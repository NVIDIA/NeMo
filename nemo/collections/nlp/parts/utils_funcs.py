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

__all__ = ['list2str', 'tensor2list']

from typing import List, Union

from torch import Tensor


def list2str(l: List[int]) -> str:
    """ Converts list to a string"""
    return ' '.join([str(x) for x in l])


def tensor2list(tensor: Tensor) -> List[Union[int, float]]:
    """ Converts tensor to a list """
    return tensor.detach().cpu().tolist()
