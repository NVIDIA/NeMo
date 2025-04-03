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

from typing import Optional

import torch
from einops import rearrange

from cosmos1.models.autoregressive.tokenizer.quantizers import FSQuantizer

# Make sure jit model output consistenly during consecutive calls
# Check here: https://github.com/pytorch/pytorch/issues/74534
torch._C._jit_set_texpr_fuser_enabled(False)


def load_jit_model(jit_filepath: str = None, device: str = "cuda") -> torch.jit.ScriptModule:
    """Loads a torch.jit.ScriptModule from a filepath.

    Args:
        jit_filepath: The filepath to the JIT-compiled model.
        device: The device to load the model onto, default=cuda.
    Returns:
        The JIT compiled model loaded to device and on eval mode.
    """
    # Make sure jit model output consistenly during consecutive calls
    # Check here: https://github.com/pytorch/pytorch/issues/74534
    torch._C._jit_set_texpr_fuser_enabled(False)

    model = torch.jit.load(jit_filepath)
    return model.eval().to(device)


class BaseDiscreteVideoFSQTokenizer(torch.nn.Module):
    """
    A base class for  Discrete Video FSQ Tokenizer that handles data type conversions, and normalization
    using provided mean and standard deviation values for latent space representation.
    Derived classes should load pre-trained encoder and decoder components into a encoder and decoder attributes.

    Attributes:
        encoder (Module | Callable): Encoder loaded from storage.
        decoder (Module | Callable): Decoder loaded from storage.
        dtype (dtype): Data type for model tensors, determined by whether bf16 is enabled.

    Args:
        name (str): Name of the model, used for differentiating cache file paths.
        latent_ch (int, optional): Number of latent channels (default is 6).
        is_bf16 (bool, optional): Flag to use Brain Floating Point 16-bit data type (default is True).
        pixel_chunk_duration (int): The duration (in number of frames) of each chunk of video data at the pixel level.
        latent_chunk_duration (int): The duration (in number of frames) of each chunk at the latent representation level.
        max_enc_batch_size (int): The maximum batch size to process in one go to avoid memory overflow.
        level (list[int]): The level defined in FSQ quantizer.
        compression_ratio (list[int]): The compression factor for (T, H, W).
    """

    def __init__(
        self,
        name: str,
        latent_ch: int = 6,
        is_bf16: bool = True,
        pixel_chunk_duration: int = 25,
        latent_chunk_duration: int = 4,
        max_enc_batch_size: int = 8,
        max_dec_batch_size: int = 4,
        levels: list[int] = [8, 8, 8, 5, 5, 5],
        compression_ratio: list[int] = [8, 16, 16],
    ):
        super().__init__()
        self.channel = latent_ch
        self.name = name
        dtype = torch.bfloat16 if is_bf16 else torch.float32
        self.dtype = dtype
        self.pixel_chunk_duration = pixel_chunk_duration
        self.latent_chunk_duration = latent_chunk_duration
        self.max_enc_batch_size = max_enc_batch_size
        self.max_dec_batch_size = max_dec_batch_size
        self.levels = levels
        self.compress_ratio = compression_ratio
        self.fsq_quantizer = FSQuantizer(levels)

    @property
    def latent_ch(self) -> int:
        """
        Returns the number of latent channels in the tokenizer.
        """
        return self.channel

    @torch.no_grad()
    def encode(self, state: torch.Tensor, pixel_chunk_duration: Optional[int] = None) -> torch.Tensor:
        B, C, T, H, W = state.shape
        if pixel_chunk_duration is None:
            # Use the default pixel chunk duration and latent chunk duration
            pixel_chunk_duration = self.pixel_chunk_duration
            latent_chunk_duration = self.latent_chunk_duration
        else:
            # Update the latent chunk duration based on the given pixel chunk duration
            latent_chunk_duration = 1 + (pixel_chunk_duration - 1) // self.compress_ratio[0]

        assert (
            T % pixel_chunk_duration == 0
        ), f"Temporal dimension {T} is not divisible by chunk_length {pixel_chunk_duration}"
        state = rearrange(state, "b c (n t) h w -> (b n) c t h w", t=pixel_chunk_duration)

        # use max_enc_batch_size to avoid OOM
        if state.shape[0] > self.max_enc_batch_size:
            quantized_out_list = []
            indices_list = []
            for i in range(0, state.shape[0], self.max_enc_batch_size):
                indices, quantized_out, _ = self.encoder(state[i : i + self.max_enc_batch_size].to(self.dtype))
                quantized_out_list.append(quantized_out)
                indices_list.append(indices)
            quantized_out = torch.cat(quantized_out_list, dim=0)
            indices = torch.cat(indices_list, dim=0)
        else:
            indices, quantized_out, _ = self.encoder(state.to(self.dtype))
        assert quantized_out.shape[2] == latent_chunk_duration
        return rearrange(quantized_out, "(b n) c t h w -> b c (n t) h w", b=B), rearrange(
            indices, "(b n) t h w -> b (n t) h w", b=B
        )

    @torch.no_grad()
    def decode(self, indices: torch.Tensor, pixel_chunk_duration: Optional[int] = None) -> torch.Tensor:
        B, T, _, _ = indices.shape
        if pixel_chunk_duration is None:
            pixel_chunk_duration = self.pixel_chunk_duration
            latent_chunk_duration = self.latent_chunk_duration
        else:
            latent_chunk_duration = 1 + (pixel_chunk_duration - 1) // self.compress_ratio[0]
        assert (
            T % latent_chunk_duration == 0
        ), f"Temporal dimension {T} is not divisible by chunk_length {latent_chunk_duration}"
        indices = rearrange(indices, "b (n t) h w -> (b n) t h w", t=latent_chunk_duration)

        # use max_dec_batch_size to avoid OOM
        if indices.shape[0] > self.max_dec_batch_size:
            state = []
            for i in range(0, indices.shape[0], self.max_dec_batch_size):
                state.append(self.decoder(indices[i : i + self.max_dec_batch_size]))
            state = torch.cat(state, dim=0)
        else:
            state = self.decoder(indices)

        assert state.shape[2] == pixel_chunk_duration
        return rearrange(state, "(b n) c t h w -> b c (n t) h w", b=B)

    def reset_dtype(self, *args, **kwargs):
        """
        Resets the data type of the encoder and decoder to the model's default data type.

        Args:
            *args, **kwargs: Unused, present to allow flexibility in method calls.
        """
        del args, kwargs
        self.decoder.to(self.dtype)
        self.encoder.to(self.dtype)


class DiscreteVideoFSQJITTokenizer(BaseDiscreteVideoFSQTokenizer):
    """
    A JIT compiled Discrete Video FSQ Tokenizer that loads pre-trained encoder
    and decoder components from a remote store, handles data type conversions, and normalization
    using provided mean and standard deviation values for latent space representation.

    Attributes:
        encoder (Module): The JIT compiled encoder loaded from storage.
        decoder (Module): The JIT compiled decoder loaded from storage.
        dtype (dtype): Data type for model tensors, determined by whether bf16 is enabled.

    Args:
        enc_fp (str): File path to the encoder's JIT file on the remote store.
        dec_fp (str): File path to the decoder's JIT file on the remote store.
        name (str): Name of the model, used for differentiating cache file paths.
        latent_ch (int, optional): Number of latent channels (default is 6).
        is_bf16 (bool, optional): Flag to use Brain Floating Point 16-bit data type (default is True).
        pixel_chunk_duration (int): The duration (in number of frames) of each chunk of video data at the pixel level.
        latent_chunk_duration (int): The duration (in number of frames) of each chunk at the latent representation level.
        max_enc_batch_size (int): The maximum batch size to process in one go to avoid memory overflow.
        level (list[int]): The level defined in FSQ quantizer.
        compression_ratio (list[int]): The compression factor for (T, H, W).
    """

    def __init__(
        self,
        enc_fp: str,
        dec_fp: str,
        name: str,
        latent_ch: int = 6,
        is_bf16: bool = True,
        pixel_chunk_duration: int = 25,
        latent_chunk_duration: int = 4,
        max_enc_batch_size: int = 8,
        max_dec_batch_size: int = 4,
        levels: list[int] = [8, 8, 8, 5, 5, 5],
        compression_ratio: list[int] = [8, 16, 16],
    ):
        super().__init__(
            name,
            latent_ch,
            is_bf16,
            pixel_chunk_duration,
            latent_chunk_duration,
            max_enc_batch_size,
            max_dec_batch_size,
            levels,
            compression_ratio,
        )

        self.load_encoder(enc_fp)
        self.load_decoder(dec_fp)

    def load_encoder(self, enc_fp: str) -> None:
        """
        Load the encoder from the remote store.

        Args:
        - enc_fp (str): File path to the encoder's JIT file on the remote store.
        """
        self.encoder = load_jit_model(enc_fp, device="cuda")
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.to(self.dtype)

    def load_decoder(self, dec_fp: str) -> None:
        """
        Load the decoder from the remote store.

        Args:
        - dec_fp (str): File path to the decoder's JIT file on the remote store.
        """
        self.decoder = load_jit_model(dec_fp, device="cuda")
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.to(self.dtype)


class DiscreteVideoFSQStateDictTokenizer(BaseDiscreteVideoFSQTokenizer):
    """
    A Discrete Video FSQ Tokenizer that loads weights from pre-trained JITed encoder
    into as nn.Module so that encoder can be "torch.compile()" and JITed decoder, so it can be torch.compiled,
    handles data type conversions, and normalization using provided mean and standard deviation values for latent
    space representation.

    Attributes:
        tokenizer_module (Module): Tokenizer module with weights loaded from JIT checkpoints
        encoder (Callable): tokenizer_module's encode method
        decoder (Callable): tokenizer_module's decode method
        dtype (dtype): Data type for model tensors, determined by whether bf16 is enabled.

    Args:
        enc_fp (str): File path to the encoder's JIT file on the remote store.
        dec_fp (str): File path to the decoder's JIT file on the remote store.
        tokenizer_module (Module): Tokenizer module that will have it's weights loaded
        name (str): Name of the model, used for differentiating cache file paths.
        latent_ch (int, optional): Number of latent channels (default is 6).
        is_bf16 (bool, optional): Flag to use Brain Floating Point 16-bit data type (default is True).
        pixel_chunk_duration (int): The duration (in number of frames) of each chunk of video data at the pixel level.
        latent_chunk_duration (int): The duration (in number of frames) of each chunk at the latent representation level.
        max_enc_batch_size (int): The maximum batch size to process in one go to avoid memory overflow.
        level (list[int]): The level defined in FSQ quantizer.
        compression_ratio (list[int]): The compression factor for (T, H, W).
    """

    def __init__(
        self,
        enc_fp: str,
        dec_fp: str,
        tokenizer_module: torch.nn.Module,
        name: str,
        latent_ch: int = 6,
        is_bf16: bool = True,
        pixel_chunk_duration: int = 25,
        latent_chunk_duration: int = 4,
        max_enc_batch_size: int = 8,
        max_dec_batch_size: int = 4,
        levels: list[int] = [8, 8, 8, 5, 5, 5],
        compression_ratio: list[int] = [8, 16, 16],
    ):
        super().__init__(
            name,
            latent_ch,
            is_bf16,
            pixel_chunk_duration,
            latent_chunk_duration,
            max_enc_batch_size,
            max_dec_batch_size,
            levels,
            compression_ratio,
        )

        self.load_encoder_and_decoder(enc_fp, dec_fp, tokenizer_module)

    def load_encoder_and_decoder(self, enc_fp: str, dec_fp: str, tokenizer_module: torch.nn.Module) -> None:
        """
        Load the encoder from the remote store.

        Args:
        - enc_fp (str): File path to the encoder's JIT file on the remote store.
        - def_fp (str): File path to the decoder's JIT file on the remote store.
        - tokenizer_module (Module): Tokenizer module that was used to create JIT checkpoints
        """
        self.decoder = load_jit_model(dec_fp)

        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.to(self.dtype)

        encoder_sd = load_jit_model(enc_fp).state_dict()

        del tokenizer_module.post_quant_conv
        del tokenizer_module.decoder

        state_dict = {
            k: v
            for k, v in (encoder_sd).items()
            #  Variables captured by JIT
            if k
            not in (
                "encoder.patcher3d.wavelets",
                "encoder.patcher3d._arange",
                "encoder.patcher3d.patch_size_buffer",
                "quantizer._levels",
                "quantizer._basis",
                "quantizer.implicit_codebook",
            )
        }

        tokenizer_module.load_state_dict(state_dict)

        tokenizer_module.eval()
        for param in tokenizer_module.parameters():
            param.requires_grad = False
        tokenizer_module.to(self.dtype)

        self.tokenizer_module = tokenizer_module
        self.encoder = self.tokenizer_module.encode

    def reset_dtype(self, *args, **kwargs):
        """
        Resets the data type of the encoder and decoder to the model's default data type.

        Args:
            *args, **kwargs: Unused, present to allow flexibility in method calls.
        """
        del args, kwargs
        self.decoder.to(self.dtype)
        self.tokenizer_module.to(self.dtype)
