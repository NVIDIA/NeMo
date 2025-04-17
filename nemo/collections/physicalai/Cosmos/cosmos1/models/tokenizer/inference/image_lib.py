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

"""A library for image tokenizers inference."""

from typing import Any

import numpy as np
import torch

from cosmos1.models.tokenizer.inference.utils import (
    load_decoder_model,
    load_encoder_model,
    load_model,
    numpy2tensor,
    pad_image_batch,
    tensor2numpy,
    unpad_image_batch,
)


class ImageTokenizer(torch.nn.Module):
    def __init__(
        self,
        checkpoint: str = None,
        checkpoint_enc: str = None,
        checkpoint_dec: str = None,
        tokenizer_config: dict[str, Any] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__()
        self._device = device
        self._dtype = getattr(torch, dtype)
        self._full_model = (
            load_model(checkpoint, tokenizer_config, device).to(self._dtype) if checkpoint is not None else None
        )
        self._enc_model = (
            load_encoder_model(checkpoint_enc, tokenizer_config, device).to(self._dtype)
            if checkpoint_enc is not None
            else None
        )
        self._dec_model = (
            load_decoder_model(checkpoint_dec, tokenizer_config, device).to(self._dtype)
            if checkpoint_dec is not None
            else None
        )

    @torch.no_grad()
    def autoencode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Reconstrcuts a batch of image tensors after embedding into a latent.

        Args:
            input_tensor: The input image Bx3xHxW layout, range [-1..1].
        Returns:
            The reconstructed tensor, layout Bx3xHxW, range [-1..1].
        """
        if self._full_model is not None:
            output_tensor = self._full_model(input_tensor)
            output_tensor = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
        else:
            output_latent = self.encode(input_tensor)[0]
            output_tensor = self.decode(output_latent)
        return output_tensor

    @torch.no_grad()
    def decode(self, input_latent: torch.Tensor) -> torch.Tensor:
        """Decodes an image from a provided latent embedding.

        Args:
            input_latent: The continuous latent Bx16xhxw for CI,
                    or the discrete indices Bxhxw for DI.
        Returns:
            The output tensor in Bx3xHxW, range [-1..1].
        """
        return self._dec_model(input_latent)

    @torch.no_grad()
    def encode(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor]:
        """Encodes an image into a latent embedding or code.

        Args:
            input_tensor: The input tensor Bx3xHxW layout, range [-1..1].
        Returns:
            For continuous image (CI) tokenizer, the tuple contains:
                - The latent embedding, Bx16x(h)x(w), where the compression
                rate is (H/h x W/w), and channel dimension of 16.
            For discrete image (DI) tokenizer, the tuple contains:
                - The indices, Bx(h)x(w), from a codebook of size 64K, which
                corresponds to FSQ levels of (8,8,8,5,5,5).
               - The discrete code, Bx6x(h)x(w), where the compression rate is
                again (H/h x W/w), and channel dimension of 6.
        """
        output_latent = self._enc_model(input_tensor)
        if isinstance(output_latent, torch.Tensor):
            return output_latent
        return output_latent[:-1]

    @torch.no_grad()
    def forward(self, image: np.ndarray) -> np.ndarray:
        """Reconstructs an image using a pre-trained tokenizer.

        Args:
            image: The input image BxHxWxC layout, range [0..255].
        Returns:
            The reconstructed image in range [0..255], layout BxHxWxC.
        """
        padded_input_image, crop_region = pad_image_batch(image)
        input_tensor = numpy2tensor(padded_input_image, dtype=self._dtype, device=self._device)
        output_tensor = self.autoencode(input_tensor)
        padded_output_image = tensor2numpy(output_tensor)
        return unpad_image_batch(padded_output_image, crop_region)
