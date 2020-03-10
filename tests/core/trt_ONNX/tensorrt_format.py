# ! /usr/bin/python
# -*- coding: utf-8 -*-

# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

import enum

from nemo import logging


# TRT does not include batch dimension.
class DataFormat(enum.IntEnum):
    UNKNOWN = 0
    NW = 1
    NHW = 2
    CHW = 3
    NHWC = 4
    NCHW = 5


def _generate_permutations():
    def is_invertible(perm):
        return min(perm) >= 0 and max(perm) < len(perm)

    def inverse_permutation(perm):
        inverse = [perm[index] for index in perm]
        return inverse

    # Inverse permutations are generated automatically below.
    # We use -1 to denote that a dummy dimension of 1 should be inserted in the convert function.
    initial_permutations = {
        (DataFormat.NCHW, DataFormat.NCHW): (0, 1, 2, 3),
        (DataFormat.NHWC, DataFormat.NHWC): (0, 1, 2, 3),
        (DataFormat.NHWC, DataFormat.NCHW): (0, 3, 1, 2),
        (DataFormat.CHW, DataFormat.CHW): (0, 1, 2),
        (DataFormat.NCHW, DataFormat.CHW): (1, 2, 3),
        (DataFormat.NHWC, DataFormat.CHW): (3, 1, 2),
        (DataFormat.NHW, DataFormat.CHW): (-1, 1, 2),
        (DataFormat.NW, DataFormat.CHW): (-1, -1, 1),
    }
    permutations = {}
    for (f1, f2), perm in initial_permutations.items():
        permutations[(f1, f2)] = perm
        if is_invertible(perm):
            permutations[(f2, f1)] = inverse_permutation(perm)
    return permutations


# This class is responsible for deducing the format of a shape,
# and converting it to the desired format (specified as a DataFormat).
class FormatManager(object):
    # Dict[Tuple[DataFormat, DataFormat], Tuple[int]]
    # This provides the correct permutation for various data format conversions.
    DATA_PERMUTATIONS = _generate_permutations()

    @staticmethod
    def deduce_format(shape):
        """
        Guesses the data format of a given shape.

        Args:
            shape (Tuple[int]): The shape, including batch dimension.

        Returns:
            DataFormat: The deduced data format.
        """
        # The smaller this ratio, the closer a and b are.
        def minmax_ratio(a, b):
            return abs(max(a, b) / min(a, b))

        # Assume all shapes include batch dimension
        if len(shape) == 4:
            # Typically, H and W are quite close, so if minmax_ratio(0, 1) > minmax_ratio(1, 2), then we assume CHW.
            if minmax_ratio(shape[1], shape[2]) > minmax_ratio(shape[2], shape[3]):
                return DataFormat.NCHW
            return DataFormat.NHWC
        elif len(shape) == 3:
            return DataFormat.NHW
        elif len(shape) == 2:
            return DataFormat.NW
        else:
            logging.warning(
                "Cannot deduce format for "
                + str(shape)
                + ". Currently only implemented for input_buffers with 1-3 non-batch dimensions. Please update this function!"
            )
            return DataFormat.UNKNOWN

    # Get the permutation required to transpose old_format to new_format
    @staticmethod
    def permutation(old_format, new_format):
        return FormatManager.DATA_PERMUTATIONS[(old_format, new_format)]

    @staticmethod
    def convert(shape, new_format):
        """
        Permutes a shape from one format to another.

        Args:
            shape (Tuple[int]): The shape to convert.
            new_format (DataFormat): The desired format of the shape.

        Returns:
            Tuple[int]: A new shape in the correct format.
        """
        old_format = FormatManager.deduce_format(shape)
        perm = FormatManager.permutation(old_format, new_format)
        return [shape[index] if index != -1 else 1 for index in perm]
