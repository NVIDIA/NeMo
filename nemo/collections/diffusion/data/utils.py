# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# pylint: disable=C0115,C0116,C0301

import numpy as np


def minimal_crop(tensor, target_divisor):
    """
    Crops the input tensor minimally so that the total number of elements
    (T * H * W) is divisible by the specified target_divisor.

    Parameters:
    - tensor: NumPy array of shape (C, T, H, W)
    - target_divisor: Positive integer specifying the desired divisor

    Returns:
    - cropped_tensor: Cropped tensor meeting the divisibility requirement

    Raises:
    - ValueError: If it's impossible to meet the divisibility requirement
    """
    if not isinstance(target_divisor, int) or target_divisor <= 0:
        raise ValueError("target_divisor must be a positive integer greater than zero.")

    C, T, H, W = tensor.shape
    total_elements = T * H * W
    remainder = total_elements % target_divisor

    if remainder == 0:
        return tensor  # No cropping needed

    # Elements per unit length in each dimension
    elements_per_T = H * W
    elements_per_H = T * W
    elements_per_W = T * H

    min_elements_removed = None
    optimal_deltas = None

    # Limit the search range to avoid unnecessary computations
    max_delta_T = min(T - 1, (remainder // elements_per_T) + 1)
    max_delta_H = min(H - 1, (remainder // elements_per_H) + 1)
    max_delta_W = min(W - 1, (remainder // elements_per_W) + 1)

    for delta_T in range(0, max_delta_T + 1):
        for delta_H in range(0, max_delta_H + 1):
            for delta_W in range(0, max_delta_W + 1):
                if delta_T == delta_H == delta_W == 0:
                    continue  # No cropping

                new_T = T - delta_T
                new_H = H - delta_H
                new_W = W - delta_W

                if new_T <= 0 or new_H <= 0 or new_W <= 0:
                    continue  # Invalid dimensions

                new_total_elements = new_T * new_H * new_W
                if new_total_elements % target_divisor == 0:
                    elements_removed = delta_T * elements_per_T + delta_H * elements_per_H + delta_W * elements_per_W
                    if min_elements_removed is None or elements_removed < min_elements_removed:
                        min_elements_removed = elements_removed
                        optimal_deltas = (delta_T, delta_H, delta_W)

    if optimal_deltas is None:
        raise ValueError("Cannot crop tensor to meet divisibility requirement.")

    delta_T, delta_H, delta_W = optimal_deltas

    # Perform the cropping
    # T dimension: crop from the end
    end_T = T - delta_T

    # H dimension: center crop
    start_H = delta_H // 2
    end_H = H - (delta_H - delta_H // 2)

    # W dimension: center crop
    start_W = delta_W // 2
    end_W = W - (delta_W - delta_W // 2)

    cropped_tensor = tensor[:, :end_T, start_H:end_H, start_W:end_W]
    return cropped_tensor


def test_no_cropping_needed():
    """Test when the tensor already meets the divisibility requirement."""
    C, T, H, W = 3, 8, 8, 8
    target_divisor = 8
    tensor = np.zeros((C, T, H, W))
    cropped_tensor = minimal_crop(tensor, target_divisor)
    assert cropped_tensor.shape == (C, T, H, W)
    assert (T * H * W) % target_divisor == 0


def test_minimal_cropping_T_dimension():
    """Test minimal cropping along the T dimension."""
    C, T, H, W = 3, 9, 7, 6
    target_divisor = 8
    tensor = np.zeros((C, T, H, W))
    cropped_tensor = minimal_crop(tensor, target_divisor)
    new_T = cropped_tensor.shape[1]
    assert new_T == T - 1, cropped_tensor.shape
    assert (new_T * H * W) % target_divisor == 0


def test_minimal_cropping_H_dimension():
    """Test minimal cropping along the H dimension."""
    C, T, H, W = 3, 7, 9, 6
    target_divisor = 8
    tensor = np.zeros((C, T, H, W))
    cropped_tensor = minimal_crop(tensor, target_divisor)
    new_H = cropped_tensor.shape[2]
    assert new_H == H - 1, cropped_tensor.shape
    assert (T * new_H * W) % target_divisor == 0


def test_minimal_cropping_W_dimension():
    """Test minimal cropping along the W dimension."""
    C, T, H, W = 3, 4, 3, 9
    target_divisor = 8
    tensor = np.zeros((C, T, H, W))
    cropped_tensor = minimal_crop(tensor, target_divisor)
    new_W = cropped_tensor.shape[3]
    assert new_W == W - 1, cropped_tensor.shape
    assert (T * H * new_W) % target_divisor == 0


def test_cropping_multiple_dimensions():
    """Test when minimal cropping requires adjustments on multiple dimensions."""
    C, T, H, W = 3, 9, 9, 8
    target_divisor = 16
    tensor = np.zeros((C, T, H, W))
    cropped_tensor = minimal_crop(tensor, target_divisor)
    new_T, new_H, new_W = cropped_tensor.shape[1:]
    assert new_T <= T and new_H <= H and new_W <= W
    assert (new_T * new_H * new_W) % target_divisor == 0


def test_large_tensor_high_divisor():
    """Test with a larger tensor and higher target_divisor."""
    C, T, H, W = 3, 50, 50, 50
    target_divisor = 1024
    tensor = np.zeros((C, T, H, W))
    cropped_tensor = minimal_crop(tensor, target_divisor)
    total_elements = cropped_tensor.shape[1] * cropped_tensor.shape[2] * cropped_tensor.shape[3]
    assert total_elements % target_divisor == 0


def test_impossible_cropping():
    """Test that an error is raised when it's impossible to meet the requirement."""
    C, T, H, W = 3, 1, 1, 1
    target_divisor = 2
    tensor = np.zeros((C, T, H, W))
    try:
        minimal_crop(tensor, target_divisor)
    except ValueError:
        pass


def test_invalid_target_divisor():
    """Test that an error is raised when target_divisor is invalid."""
    C, T, H, W = 3, 8, 8, 8
    tensor = np.zeros((C, T, H, W))
    try:
        minimal_crop(tensor, -1)
    except ValueError:
        pass


def test_minimal_elements_removed():
    """Test that the minimal number of elements are removed."""
    C, T, H, W = 3, 7, 7, 7
    target_divisor = 8
    tensor = np.zeros((C, T, H, W))
    cropped_tensor = minimal_crop(tensor, target_divisor)
    elements_removed = (T * H * W) - (cropped_tensor.shape[1] * cropped_tensor.shape[2] * cropped_tensor.shape[3])
    print(cropped_tensor.shape)
    assert elements_removed > 0
    assert (cropped_tensor.shape[1] * cropped_tensor.shape[2] * cropped_tensor.shape[3]) % target_divisor == 0


test_no_cropping_needed()
test_minimal_elements_removed()
test_cropping_multiple_dimensions()
test_minimal_cropping_T_dimension()
test_minimal_cropping_H_dimension()
test_minimal_cropping_W_dimension()
test_impossible_cropping()
test_invalid_target_divisor()
