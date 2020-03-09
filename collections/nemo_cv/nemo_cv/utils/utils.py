# Copyright (C) NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Tomasz Kornuta"

import itertools
from torch.nn.functional import pad


def pad_tensors_to_max(tensor_list):
    """
    Method returns list of tensors, each padded to the maximum sizes.

    Args:
        tensor_list - List of tensor to be padded.
    """
    # Get max size of tensors.
    max_sizes = max([t.size() for t in tensor_list])

    #print("MAX = ", max_sizes)
    # Number of dimensions
    dims = len(max_sizes)
    # Create the list of zeros.
    zero_sizes = [0] * dims

    # Pad list of tensors to max size.
    padded_tensors = []
    for tensor in tensor_list:
        # Get list of current sizes.
        cur_sizes = tensor.size()

        #print("cur_sizes = ", cur_sizes)

        # Create the reverted list of "desired extensions".
        ext_sizes = [m-c for (m, c) in zip(max_sizes, cur_sizes)][::-1]

        #print("ext_sizes = ", ext_sizes)

        # Interleave two lists.
        pad_sizes = list(itertools.chain(*zip(zero_sizes, ext_sizes)))

        #print("pad_sizes = ", pad_sizes)

        # Pad tensor, starting from last dimension.
        padded_tensor = pad(
            input=tensor,
            pad=pad_sizes,
            mode='constant', value=0)

        #print("Tensor after padding: ", padded_tensor.size())
        # Add to list.
        padded_tensors.append(padded_tensor)

    # Return the padded list.
    return padded_tensors
