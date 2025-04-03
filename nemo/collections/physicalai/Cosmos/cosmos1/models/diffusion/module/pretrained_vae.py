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

import os
from abc import ABC, abstractmethod

import torch
from einops import rearrange
from torch.nn.modules import Module


class BaseVAE(torch.nn.Module, ABC):
    """
    Abstract base class for a Variational Autoencoder (VAE).

    All subclasses should implement the methods to define the behavior for encoding
    and decoding, along with specifying the latent channel size.
    """

    def __init__(self, channel: int = 3, name: str = "vae"):
        super().__init__()
        self.channel = channel
        self.name = name

    @property
    def latent_ch(self) -> int:
        """
        Returns the number of latent channels in the VAE.
        """
        return self.channel

    @abstractmethod
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor into a latent representation.

        Args:
        - state (torch.Tensor): The input tensor to encode.

        Returns:
        - torch.Tensor: The encoded latent tensor.
        """
        pass

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representation back to the original space.

        Args:
        - latent (torch.Tensor): The latent tensor to decode.

        Returns:
        - torch.Tensor: The decoded tensor.
        """
        pass

    @property
    def spatial_compression_factor(self) -> int:
        """
        Returns the spatial reduction factor for the VAE.
        """
        raise NotImplementedError("The spatial_compression_factor property must be implemented in the derived class.")


class BasePretrainedImageVAE(BaseVAE):
    """
    A base class for pretrained Variational Autoencoder (VAE) that loads mean and standard deviation values
    from a remote store, handles data type conversions, and normalization
    using provided mean and standard deviation values for latent space representation.
    Derived classes should load pre-trained encoder and decoder components from a remote store

    Attributes:
        latent_mean (Tensor): The mean used for normalizing the latent representation.
        latent_std (Tensor): The standard deviation used for normalizing the latent representation.
        dtype (dtype): Data type for model tensors, determined by whether bf16 is enabled.

    Args:
        mean_std_fp (str): File path to the pickle file containing mean and std of the latent space.
        latent_ch (int, optional): Number of latent channels (default is 16).
        is_image (bool, optional): Flag to indicate whether the output is an image (default is True).
        is_bf16 (bool, optional): Flag to use Brain Floating Point 16-bit data type (default is True).
    """

    def __init__(
        self,
        name: str,
        latent_ch: int = 16,
        is_image: bool = True,
        is_bf16: bool = True,
    ) -> None:
        super().__init__(latent_ch, name)
        dtype = torch.bfloat16 if is_bf16 else torch.float32
        self.dtype = dtype
        self.is_image = is_image
        self.name = name

    def register_mean_std(self, vae_dir: str) -> None:
        latent_mean, latent_std = torch.load(os.path.join(vae_dir, "image_mean_std.pt"), weights_only=True)

        target_shape = [1, self.latent_ch, 1, 1] if self.is_image else [1, self.latent_ch, 1, 1, 1]

        self.register_buffer(
            "latent_mean",
            latent_mean.to(self.dtype).reshape(*target_shape),
            persistent=False,
        )
        self.register_buffer(
            "latent_std",
            latent_std.to(self.dtype).reshape(*target_shape),
            persistent=False,
        )

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode the input state to latent space; also handle the dtype conversion, mean and std scaling
        """
        in_dtype = state.dtype
        latent_mean = self.latent_mean.to(in_dtype)
        latent_std = self.latent_std.to(in_dtype)
        encoded_state = self.encoder(state.to(self.dtype))
        if isinstance(encoded_state, torch.Tensor):
            pass
        elif isinstance(encoded_state, tuple):
            assert isinstance(encoded_state[0], torch.Tensor)
            encoded_state = encoded_state[0]
        else:
            raise ValueError("Invalid type of encoded state")
        return (encoded_state.to(in_dtype) - latent_mean) / latent_std

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode the input latent to state; also handle the dtype conversion, mean and std scaling
        """
        in_dtype = latent.dtype
        latent = latent * self.latent_std.to(in_dtype) + self.latent_mean.to(in_dtype)
        return self.decoder(latent.to(self.dtype)).to(in_dtype)

    def reset_dtype(self, *args, **kwargs):
        """
        Resets the data type of the encoder and decoder to the model's default data type.

        Args:
            *args, **kwargs: Unused, present to allow flexibility in method calls.
        """
        del args, kwargs
        self.decoder.to(self.dtype)
        self.encoder.to(self.dtype)


class JITVAE(BasePretrainedImageVAE):
    """
    A JIT compiled Variational Autoencoder (VAE) that loads pre-trained encoder
    and decoder components from a remote store, handles data type conversions, and normalization
    using provided mean and standard deviation values for latent space representation.

    Attributes:
        encoder (Module): The JIT compiled encoder loaded from storage.
        decoder (Module): The JIT compiled decoder loaded from storage.
        latent_mean (Tensor): The mean used for normalizing the latent representation.
        latent_std (Tensor): The standard deviation used for normalizing the latent representation.
        dtype (dtype): Data type for model tensors, determined by whether bf16 is enabled.

    Args:
        name (str): Name of the model, used for differentiating cache file paths.
        latent_ch (int, optional): Number of latent channels (default is 16).
        is_image (bool, optional): Flag to indicate whether the output is an image (default is True).
        is_bf16 (bool, optional): Flag to use Brain Floating Point 16-bit data type (default is True).
    """

    def __init__(
        self,
        name: str,
        latent_ch: int = 16,
        is_image: bool = True,
        is_bf16: bool = True,
    ):
        super().__init__(name, latent_ch, is_image, is_bf16)

    def load_encoder(self, vae_dir: str) -> None:
        """
        Load the encoder from the remote store.
        """
        self.encoder = torch.load(os.path.join(vae_dir, "encoder.jit"), weights_only=True)

        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.to(self.dtype)

    def load_decoder(self, vae_dir: str) -> None:
        """
        Load the decoder from the remote store.
        """
        self.decoder = torch.load(os.path.join(vae_dir, "decoder.jit"), weights_only=True)

        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.to(self.dtype)


class BaseVAE(torch.nn.Module, ABC):
    """
    Abstract base class for a Variational Autoencoder (VAE).

    All subclasses should implement the methods to define the behavior for encoding
    and decoding, along with specifying the latent channel size.
    """

    def __init__(self, channel: int = 3, name: str = "vae"):
        super().__init__()
        self.channel = channel
        self.name = name

    @property
    def latent_ch(self) -> int:
        """
        Returns the number of latent channels in the VAE.
        """
        return self.channel

    @abstractmethod
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor into a latent representation.

        Args:
        - state (torch.Tensor): The input tensor to encode.

        Returns:
        - torch.Tensor: The encoded latent tensor.
        """
        pass

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representation back to the original space.

        Args:
        - latent (torch.Tensor): The latent tensor to decode.

        Returns:
        - torch.Tensor: The decoded tensor.
        """
        pass

    @property
    def spatial_compression_factor(self) -> int:
        """
        Returns the spatial reduction factor for the VAE.
        """
        raise NotImplementedError("The spatial_compression_factor property must be implemented in the derived class.")


class VideoTokenizerInterface(ABC):
    @abstractmethod
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        pass

    @abstractmethod
    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        pass

    @property
    @abstractmethod
    def spatial_compression_factor(self):
        pass

    @property
    @abstractmethod
    def temporal_compression_factor(self):
        pass

    @property
    @abstractmethod
    def spatial_resolution(self):
        pass

    @property
    @abstractmethod
    def pixel_chunk_duration(self):
        pass

    @property
    @abstractmethod
    def latent_chunk_duration(self):
        pass


class BasePretrainedVideoTokenizer(ABC):
    """
    Base class for a pretrained video tokenizer that handles chunking of video data for efficient processing.

    Args:
        pixel_chunk_duration (int): The duration (in number of frames) of each chunk of video data at the pixel level.
        temporal_compress_factor (int): The factor by which the video data is temporally compressed during processing.
        max_enc_batch_size (int): The maximum batch size to process in one go during encoding to avoid memory overflow.
        max_dec_batch_size (int): The maximum batch size to process in one go during decoding to avoid memory overflow.

    The class introduces parameters for managing temporal chunks (`pixel_chunk_duration` and `temporal_compress_factor`)
    which define how video data is subdivided and compressed during the encoding and decoding processes. The
    `max_enc_batch_size` and `max_dec_batch_size` parameters allow processing in smaller batches to handle memory
    constraints.
    """

    def __init__(
        self,
        pixel_chunk_duration: int = 17,
        temporal_compress_factor: int = 8,
        max_enc_batch_size: int = 8,
        max_dec_batch_size: int = 4,
    ):
        self._pixel_chunk_duration = pixel_chunk_duration
        self._temporal_compress_factor = temporal_compress_factor
        self.max_enc_batch_size = max_enc_batch_size
        self.max_dec_batch_size = max_dec_batch_size

    def register_mean_std(self, vae_dir: str) -> None:
        latent_mean, latent_std = torch.load(os.path.join(vae_dir, "mean_std.pt"), weights_only=True)

        latent_mean = latent_mean.view(self.latent_ch, -1)[:, : self.latent_chunk_duration]
        latent_std = latent_std.view(self.latent_ch, -1)[:, : self.latent_chunk_duration]

        target_shape = [1, self.latent_ch, self.latent_chunk_duration, 1, 1]

        self.register_buffer(
            "latent_mean",
            latent_mean.to(self.dtype).reshape(*target_shape),
            persistent=False,
        )
        self.register_buffer(
            "latent_std",
            latent_std.to(self.dtype).reshape(*target_shape),
            persistent=False,
        )

    def transform_encode_state_shape(self, state: torch.Tensor) -> torch.Tensor:
        """
        Rearranges the input state tensor to the required shape for encoding video data. Mainly for chunk based encoding
        """
        B, C, T, H, W = state.shape
        assert (
            T % self.pixel_chunk_duration == 0
        ), f"Temporal dimension {T} is not divisible by chunk_length {self.pixel_chunk_duration}"
        return rearrange(state, "b c (n t) h w -> (b n) c t h w", t=self.pixel_chunk_duration)

    def transform_decode_state_shape(self, latent: torch.Tensor) -> torch.Tensor:
        B, _, T, _, _ = latent.shape
        assert (
            T % self.latent_chunk_duration == 0
        ), f"Temporal dimension {T} is not divisible by chunk_length {self.latent_chunk_duration}"
        return rearrange(latent, "b c (n t) h w -> (b n) c t h w", t=self.latent_chunk_duration)

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        if self._temporal_compress_factor == 1:
            _, _, origin_T, _, _ = state.shape
            state = rearrange(state, "b c t h w -> (b t) c 1 h w")
        B, C, T, H, W = state.shape
        state = self.transform_encode_state_shape(state)
        # use max_enc_batch_size to avoid OOM
        if state.shape[0] > self.max_enc_batch_size:
            latent = []
            for i in range(0, state.shape[0], self.max_enc_batch_size):
                latent.append(super().encode(state[i : i + self.max_enc_batch_size]))
            latent = torch.cat(latent, dim=0)
        else:
            latent = super().encode(state)

        latent = rearrange(latent, "(b n) c t h w -> b c (n t) h w", b=B)
        if self._temporal_compress_factor == 1:
            latent = rearrange(latent, "(b t) c 1 h w -> b c t h w", t=origin_T)
        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decodes a batch of latent representations into video frames by applying temporal chunking. Similar to encode,
        it handles video data by processing smaller temporal chunks to reconstruct the original video dimensions.

        It can also decode single frame image data.

        Args:
            latent (torch.Tensor): The latent space tensor containing encoded video data.

        Returns:
            torch.Tensor: The decoded video tensor reconstructed from latent space.
        """
        if self._temporal_compress_factor == 1:
            _, _, origin_T, _, _ = latent.shape
            latent = rearrange(latent, "b c t h w -> (b t) c 1 h w")
        B, _, T, _, _ = latent.shape
        latent = self.transform_decode_state_shape(latent)
        # use max_enc_batch_size to avoid OOM
        if latent.shape[0] > self.max_dec_batch_size:
            state = []
            for i in range(0, latent.shape[0], self.max_dec_batch_size):
                state.append(super().decode(latent[i : i + self.max_dec_batch_size]))
            state = torch.cat(state, dim=0)
        else:
            state = super().decode(latent)
        assert state.shape[2] == self.pixel_chunk_duration
        state = rearrange(state, "(b n) c t h w -> b c (n t) h w", b=B)
        if self._temporal_compress_factor == 1:
            return rearrange(state, "(b t) c 1 h w -> b c t h w", t=origin_T)
        return state

    @property
    def pixel_chunk_duration(self) -> int:
        return self._pixel_chunk_duration

    @property
    def latent_chunk_duration(self) -> int:
        # return self._latent_chunk_duration
        assert (self.pixel_chunk_duration - 1) % self.temporal_compression_factor == 0, (
            f"Pixel chunk duration {self.pixel_chunk_duration} is not divisible by latent chunk duration "
            f"{self.latent_chunk_duration}"
        )
        return (self.pixel_chunk_duration - 1) // self.temporal_compression_factor + 1

    @property
    def temporal_compression_factor(self):
        return self._temporal_compress_factor

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        if num_pixel_frames == 1:
            return 1
        assert (
            num_pixel_frames % self.pixel_chunk_duration == 0
        ), f"Temporal dimension {num_pixel_frames} is not divisible by chunk_length {self.pixel_chunk_duration}"
        return num_pixel_frames // self.pixel_chunk_duration * self.latent_chunk_duration

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        if num_latent_frames == 1:
            return 1
        assert (
            num_latent_frames % self.latent_chunk_duration == 0
        ), f"Temporal dimension {num_latent_frames} is not divisible by chunk_length {self.latent_chunk_duration}"
        return num_latent_frames // self.latent_chunk_duration * self.pixel_chunk_duration


class VideoJITTokenizer(BasePretrainedVideoTokenizer, JITVAE, VideoTokenizerInterface):
    """
    Instance of BasePretrainedVideoVAE that loads encoder and decoder from JIT scripted module file
    """

    def __init__(
        self,
        name: str,
        latent_ch: int = 16,
        is_bf16: bool = True,
        spatial_compression_factor: int = 16,
        temporal_compression_factor: int = 8,
        pixel_chunk_duration: int = 17,
        max_enc_batch_size: int = 8,
        max_dec_batch_size: int = 4,
        spatial_resolution: str = "720",
    ):
        super().__init__(
            pixel_chunk_duration,
            temporal_compression_factor,
            max_enc_batch_size,
            max_dec_batch_size,
        )
        super(BasePretrainedVideoTokenizer, self).__init__(
            name,
            latent_ch,
            False,
            is_bf16,
        )

        self._spatial_compression_factor = spatial_compression_factor
        self._spatial_resolution = spatial_resolution

    @property
    def spatial_compression_factor(self):
        return self._spatial_compression_factor

    @property
    def spatial_resolution(self) -> str:
        return self._spatial_resolution


class JointImageVideoTokenizer(BaseVAE, VideoTokenizerInterface):
    def __init__(
        self,
        image_vae: torch.nn.Module,
        video_vae: torch.nn.Module,
        name: str,
        latent_ch: int = 16,
        squeeze_for_image: bool = True,
    ):
        super().__init__(latent_ch, name)
        self.image_vae = image_vae
        self.video_vae = video_vae
        self.squeeze_for_image = squeeze_for_image

    def encode_image(self, state: torch.Tensor) -> torch.Tensor:
        if self.squeeze_for_image:
            return self.image_vae.encode(state.squeeze(2)).unsqueeze(2)
        return self.image_vae.encode(state)

    def decode_image(self, latent: torch.Tensor) -> torch.Tensor:
        if self.squeeze_for_image:
            return self.image_vae.decode(latent.squeeze(2)).unsqueeze(2)
        return self.image_vae.decode(latent)

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = state.shape
        if T == 1:
            return self.encode_image(state)

        return self.video_vae.encode(state)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = latent.shape
        if T == 1:
            return self.decode_image(latent)
        return self.video_vae.decode(latent)

    def reset_dtype(self, *args, **kwargs):
        """
        Resets the data type of the encoder and decoder to the model's default data type.

        Args:
            *args, **kwargs: Unused, present to allow flexibility in method calls.
        """
        del args, kwargs
        self.video_vae.reset_dtype()

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        if num_pixel_frames == 1:
            return 1
        return self.video_vae.get_latent_num_frames(num_pixel_frames)

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        if num_latent_frames == 1:
            return 1
        return self.video_vae.get_pixel_num_frames(num_latent_frames)

    @property
    def spatial_compression_factor(self):
        return self.video_vae.spatial_compression_factor

    @property
    def temporal_compression_factor(self):
        return self.video_vae.temporal_compression_factor

    @property
    def spatial_resolution(self) -> str:
        return self.video_vae.spatial_resolution

    @property
    def pixel_chunk_duration(self) -> int:
        return self.video_vae.pixel_chunk_duration

    @property
    def latent_chunk_duration(self) -> int:
        return self.video_vae.latent_chunk_duration


class JointImageVideoSharedJITTokenizer(JointImageVideoTokenizer):
    """
    First version of the ImageVideoVAE trained with Fitsum.
    We have to use seperate mean and std for image and video due to non-causal nature of the model.
    """

    def __init__(self, image_vae: Module, video_vae: Module, name: str, latent_ch: int = 16):
        super().__init__(image_vae, video_vae, name, latent_ch, squeeze_for_image=False)
        assert isinstance(image_vae, JITVAE)
        assert isinstance(
            video_vae, VideoJITTokenizer
        ), f"video_vae should be an instance of VideoJITVAE, got {type(video_vae)}"
        # a hack to make the image_vae and video_vae share the same encoder and decoder

    def load_weights(self, vae_dir: str):
        self.video_vae.register_mean_std(vae_dir)

        self.video_vae.load_decoder(vae_dir)
        self.video_vae.load_encoder(vae_dir)
