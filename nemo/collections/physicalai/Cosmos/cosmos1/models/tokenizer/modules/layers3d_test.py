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

"""The test for model definition of 3D layers

PYTHONPATH=$PWD pytest -v cosmos1/models/tokenizer/modules/layers3d_test.py
"""
import os

import numpy as np
import pytest
import torch
from cosmos1.models.tokenizer.inference.utils import read_video
from cosmos1.models.tokenizer.inference.video_lib import CausalVideoTokenizer
from cosmos1.models.tokenizer.networks import TokenizerConfigs
from torchvision.transforms import CenterCrop

# test configs
TEST_CONFIGS = [
    ("CV4x8x8", "nvidia/Cosmos-0.1-Tokenizer-CV4x8x8"),
    ("CV8x8x8", "nvidia/Cosmos-0.1-Tokenizer-CV8x8x8"),
    ("CV8x16x16", "nvidia/Cosmos-0.1-Tokenizer-CV8x16x16"),
    ("DV4x8x8", "nvidia/Cosmos-0.1-Tokenizer-DV4x8x8"),
    ("DV8x8x8", "nvidia/Cosmos-0.1-Tokenizer-DV8x8x8"),
    ("DV8x16x16", "nvidia/Cosmos-0.1-Tokenizer-DV8x16x16"),
    ("CV8x8x8", "nvidia/Cosmos-1.0-Tokenizer-CV8x8x8"),
    ("DV8x16x16", "nvidia/Cosmos-1.0-Tokenizer-DV8x16x16"),
    ("CV4x8x8-LowRes", "nvidia/Cosmos-1.0-Tokenizer-CV4x8x8-LowRes"),
    ("DV4x8x8-LowRes", "nvidia/Cosmos-1.0-Tokenizer-DV4x8x8-LowRes"),
]


@pytest.fixture(scope="module")
def video_tensor():
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_data", "video.mp4")
    print(f"video_path: {video_path}")
    video = read_video(video_path)

    assert video.shape[0] >= 17, "Video length should be at least 17 frames"
    assert video.shape[1] >= 512, "Video height should be at least 512 pixels"
    assert video.shape[2] >= 512, "Video width should be at least 512 pixels"
    assert video.shape[3] == 3, "Video should have 3 channels"

    input_tensor = CenterCrop(512)(
        torch.from_numpy(video[np.newaxis, ...])[:, :17].to("cuda").to(torch.bfloat16).permute(0, 4, 1, 2, 3)
        / 255.0
        * 2.0
        - 1.0
    )
    return input_tensor


@pytest.mark.parametrize("config", TEST_CONFIGS)
def test_tokenizer(config, video_tensor):
    name, model_id = config
    continuous = name.startswith(("C", "c"))
    temporal_compression, spatial_compression = list(map(int, name[2:].split("x")[:2]))
    print(f"\nTesting tokenizer: {model_id}")
    print(f"temporal_compression={temporal_compression}")
    print(f"spatial_compression={spatial_compression}")
    print(f"checkpoint_enc=checkpoints/{os.path.basename(model_id)}/encoder.jit")
    print(f"checkpoint_dec=checkpoints/{os.path.basename(model_id)}/decoder.jit")

    _config = TokenizerConfigs[name.replace("-", "_")].value
    autoencoder = CausalVideoTokenizer(
        checkpoint_enc=f"checkpoints/{os.path.basename(model_id)}/encoder.jit",
        checkpoint_dec=f"checkpoints/{os.path.basename(model_id)}/decoder.jit",
        tokenizer_config=_config,
        device="cuda",
        dtype="bfloat16",
    )

    try:
        # Test shape check
        reconstructed_tensor = auto_shape_check(
            video_tensor, autoencoder, temporal_compression, spatial_compression, continuous
        )
    finally:
        # Cleanup
        del autoencoder
        del reconstructed_tensor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def auto_shape_check(input_tensor, autoencoder, temporal_compression, spatial_compression, continuous):
    if continuous:
        (latent,) = autoencoder.encode(input_tensor)
        torch.testing.assert_close(
            latent.shape,
            (1, 16, (17 - 1) // temporal_compression + 1, 512 // spatial_compression, 512 // spatial_compression),
        )
        reconstructed_tensor = autoencoder.decode(latent)
    else:
        (indices, codes) = autoencoder.encode(input_tensor)
        torch.testing.assert_close(
            indices.shape,
            (1, (17 - 1) // temporal_compression + 1, 512 // spatial_compression, 512 // spatial_compression),
        )
        torch.testing.assert_close(
            codes.shape,
            (1, 6, (17 - 1) // temporal_compression + 1, 512 // spatial_compression, 512 // spatial_compression),
        )
        reconstructed_tensor = autoencoder.decode(indices)

    torch.testing.assert_close(reconstructed_tensor.shape, input_tensor.shape)
    return reconstructed_tensor
