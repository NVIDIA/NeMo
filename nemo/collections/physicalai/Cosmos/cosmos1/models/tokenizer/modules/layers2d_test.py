# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0115,C0116,C0301

"""The test for model definition of 2D layers

PYTHONPATH=$PWD pytest -v cosmos1/models/tokenizer/modules/layers2d_test.py
"""
import os

import numpy as np
import pytest
import torch
from cosmos1.models.tokenizer.inference.image_lib import ImageTokenizer
from cosmos1.models.tokenizer.inference.utils import read_image
from cosmos1.models.tokenizer.networks import TokenizerConfigs
from torchvision.transforms import CenterCrop

# test configs
TEST_CONFIGS = [
    ("CI8x8", "nvidia/Cosmos-0.1-Tokenizer-CI8x8"),
    ("CI16x16", "nvidia/Cosmos-0.1-Tokenizer-CI16x16"),
    ("DI8x8", "nvidia/Cosmos-0.1-Tokenizer-DI8x8"),
    ("DI16x16", "nvidia/Cosmos-0.1-Tokenizer-DI16x16"),
    ("CI8x8-LowRes", "nvidia/Cosmos-1.0-Tokenizer-CI8x8-LowRes"),
    ("CI16x16-LowRes", "nvidia/Cosmos-1.0-Tokenizer-CI16x16-LowRes"),
    ("DI8x8-LowRes", "nvidia/Cosmos-1.0-Tokenizer-DI8x8-LowRes"),
    ("DI16x16-LowRes", "nvidia/Cosmos-1.0-Tokenizer-DI16x16-LowRes"),
]


@pytest.fixture(scope="module")
def image_tensor():
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_data", "image.png")
    print(f"image_path: {image_path}")
    image = read_image(image_path)

    assert image.shape[0] >= 512, "Image height should be at least 512 pixels"
    assert image.shape[1] >= 512, "Image width should be at least 512 pixels"
    assert image.shape[2] == 3, "Image should have 3 channels"

    input_tensor = CenterCrop(512)(
        torch.from_numpy(image[np.newaxis, ...]).to("cuda").to(torch.bfloat16).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
    )
    return input_tensor


@pytest.mark.parametrize("config", TEST_CONFIGS)
def test_tokenizer(config, image_tensor):
    name, model_id = config
    continuous = name.startswith(("C", "c"))
    [
        spatial_compression,
    ] = list(map(int, name[2:].split("x")[:1]))
    print(f"\nTesting tokenizer: {model_id}")
    print(f"spatial_compression={spatial_compression}")
    print(f"checkpoint_enc=checkpoints/{os.path.basename(model_id)}/encoder.jit")
    print(f"checkpoint_dec=checkpoints/{os.path.basename(model_id)}/decoder.jit")

    _config = TokenizerConfigs[name.replace("-", "_")].value
    autoencoder = ImageTokenizer(
        checkpoint_enc=f"checkpoints/{os.path.basename(model_id)}/encoder.jit",
        checkpoint_dec=f"checkpoints/{os.path.basename(model_id)}/decoder.jit",
        tokenizer_config=_config,
        device="cuda",
        dtype="bfloat16",
    )

    try:
        # Test shape check
        reconstructed_tensor = auto_shape_check(image_tensor, autoencoder, spatial_compression, continuous)
    finally:
        # Cleanup
        del autoencoder
        del reconstructed_tensor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def auto_shape_check(input_tensor, autoencoder, spatial_compression, continuous):
    if continuous:
        (latent,) = autoencoder.encode(input_tensor)
        torch.testing.assert_close(latent.shape, (1, 16, 512 // spatial_compression, 512 // spatial_compression))
        reconstructed_tensor = autoencoder.decode(latent)
    else:
        (indices, codes) = autoencoder.encode(input_tensor)
        torch.testing.assert_close(indices.shape, (1, 512 // spatial_compression, 512 // spatial_compression))
        torch.testing.assert_close(codes.shape, (1, 6, 512 // spatial_compression, 512 // spatial_compression))
        reconstructed_tensor = autoencoder.decode(indices)

    torch.testing.assert_close(reconstructed_tensor.shape, input_tensor.shape)
    return reconstructed_tensor
