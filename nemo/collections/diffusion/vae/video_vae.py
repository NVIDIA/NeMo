# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

# pylint: disable=C0115,C0116,C0301

import os
from abc import ABC, abstractmethod

import torch
from einops import rearrange
from huggingface_hub import snapshot_download
from torch.nn.modules import Module

from nemo.collections.diffusion.vae.pretrained_vae import JITVAE, BaseVAE


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
        extension = self.mean_std_fp.split(".")[-1]
        latent_mean, latent_std = torch.load(os.path.join(vae_dir, f"mean_std.{extension}"))

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
        # assert (
        #     T % self.pixel_chunk_duration == 0
        # ), f"Temporal dimension {T} is not divisible by chunk_length {self.pixel_chunk_duration}"
        return rearrange(state, "b c (n t) h w -> (b n) c t h w", t=T)

    def transform_decode_state_shape(self, latent: torch.Tensor) -> torch.Tensor:
        B, _, T, _, _ = latent.shape
        # assert (
        #     T % self.latent_chunk_duration == 0
        # ), f"Temporal dimension {T} is not divisible by chunk_length {self.latent_chunk_duration}"
        return rearrange(latent, "b c (n t) h w -> (b n) c t h w", t=T)

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        origin_T = None
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
        origin_T = None
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
        vae_path: str,
        enc_fp: str,
        dec_fp: str,
        name: str,
        mean_std_fp: str,
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
            enc_fp,
            dec_fp,
            name,
            mean_std_fp,
            latent_ch,
            False,
            is_bf16,
        )

        self._spatial_compression_factor = spatial_compression_factor
        self._spatial_resolution = spatial_resolution

        self.register_mean_std(vae_path)

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
        self.image_vae.reset_dtype()
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
        self.image_vae.register_mean_std(vae_dir)

        self.video_vae.load_decoder(vae_dir)
        self.video_vae.load_encoder(vae_dir)

        self.image_vae.encoder = self.video_vae.encoder
        self.image_vae.decoder = self.video_vae.decoder


def video_vae3_512(
    vae_path: str,
    enc_fp: str = None,
    dec_fp: str = None,
    mean_std_fp: str = None,
    latent_ch: int = 16,
    is_bf16: bool = True,
    video_mean_std_fp=None,
    image_mean_std_fp=None,
    spatial_compression_factor: int = 16,
    temporal_compression_factor: int = 8,
    pixel_chunk_duration: int = 121,
    max_enc_batch_size: int = 8,
    max_dec_batch_size: int = 4,
    spatial_resolution: str = "720",
):
    name = 'cosmos_tokenizer'
    if enc_fp is None:
        enc_fp = os.path.join(vae_path, 'encoder.jit')
    if dec_fp is None:
        dec_fp = os.path.join(vae_path, 'decoder.jit')
    if video_mean_std_fp is None:
        video_mean_std_fp = os.path.join(vae_path, 'mean_std.pt')

    video_vae = VideoJITTokenizer(
        vae_path,
        enc_fp,
        dec_fp,
        name,
        video_mean_std_fp,
        pixel_chunk_duration=pixel_chunk_duration,
        spatial_compression_factor=8,
        temporal_compression_factor=8,
    )

    image_vae = VideoJITTokenizer(
        vae_path,
        enc_fp,
        dec_fp,
        name,
        video_mean_std_fp,
        pixel_chunk_duration=1,
        spatial_compression_factor=8,
        temporal_compression_factor=8,
    )

    video_image_vae = JointImageVideoSharedJITTokenizer(
        image_vae,
        video_vae,
        name,
    )

    return video_image_vae


if __name__ == "__main__":
    tokenizer_dir = snapshot_download("nvidia/Cosmos-1.0-Tokenizer-CV8x8x8")
    vae = video_vae3_512(vae_path=tokenizer_dir)

    image = torch.randn(1, 3, 1, 704, 1280, device="cuda", dtype=torch.bfloat16)
    latent = vae.encode(image)
    print(latent.shape)

    video = torch.randn(1, 3, 121, 704, 1280, device="cuda", dtype=torch.bfloat16)
    latent = vae.encode(video)
    print(latent.shape)
