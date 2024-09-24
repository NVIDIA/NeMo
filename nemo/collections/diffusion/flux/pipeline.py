import torch
from torch import nn
from nemo.collections.diffusion.encoders.conditioner import FrozenCLIPEmbedder, FrozenT5Embedder
from nemo.collections.diffusion.vae.autoencoder import AutoEncoder, AutoEncoderParams
from nemo.collections.diffusion.flux.model import Flux, FluxParams
from nemo.collections.diffusion.schedulers.flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from nemo.collections.diffusion.utils.flux_pipeline_utils import FluxModelParams
from typing import Any, Callable, Dict, List, Optional, Union
from tqdm import tqdm
import numpy as np


class FluxInferencePipeline(nn.Module):
    def __init__(self, params: FluxModelParams):
        super().__init__()
        self.device = params.device
        params.clip_params['device'] = self.device
        params.t5_params['device'] = self.device
        self.vae = AutoEncoder(params.vae_params)
        self.clip_encoder = FrozenCLIPEmbedder(**params.clip_params)
        self.t5_encoder = FrozenT5Embedder(**params.t5_params)
        self.transformer = Flux(params.flux_params)
        self.vae_scale_factor = 2**(len(self.vae.params.ch_mult) - 1)
        self.scheduler = FlowMatchEulerDiscreteScheduler(**params.scheduler_params)

    def encoder_prompt(
            self,
            prompt: Union[str, List[str]],
            num_images_per_prompt: int = 1,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            max_sequence_length: int = 512,
            device: Optional[torch.device] = 'cuda',
            dtype: Optional[torch.dtype] = torch.float,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("Either prompt or prompt_embeds must be provided.")

        if prompt_embeds is None:
            prompt_embeds = self.t5_encoder(prompt, max_sequence_length=max_sequence_length)
        seq_len = prompt_embeds.shape[1]
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)


        if pooled_prompt_embeds is None:
            _, pooled_prompt_embeds = self.clip_encoder(prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size*num_images_per_prompt, -1)

        dtype = self.t5_encoder.dtype if self.t5_encoder is not None else dtype
        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    @staticmethod
    def _prepare_latent_image_ids(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

        return latents

    @staticmethod
    def _calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = FluxInferencePipeline._generate_rand_latents(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        return latents, latent_image_ids

    @staticmethod
    def _generate_rand_latents(
        shape,
        generator,
        device,
        dtype,
    ):
        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            latents = [
                torch.randn(shape, generator=generator[i], device=device, dtype=dtype, layout=layout)
                for i in range(batch_size)
            ]
            latents = torch.cat(latents, dim=0).to(device=device)
        else:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

        return latents
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 28,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        max_sequence_length: int = 512,
        device: Optional[torch.device] = 'cpu',
    ):
        height = height
        width = width
        self._guidance_scale = guidance_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None and isinstance(prompt_embeds, torch.FloatTensor):
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("Either prompt or prompt_embeds must be provided.")

        ## get text prompt embeddings
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encoder_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device = device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length
        )

        ## prepare image latents
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents
        )

        # prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]

        mu = FluxInferencePipeline._calculate_shift(
            image_seq_len,
            self.scheduler.base_image_seq_len,
            self.scheduler.max_image_seq_len,
            self.scheduler.base_shift,
            self.scheduler.max_shift,
        )

        self.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)


        for _ in tqdm(range(num_inference_steps)):
            for i, t in enumerate(timesteps):

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                if self.transformer.config.guidance_embed:
                    guidance = torch.tensor([guidance_scale], device=device).expand(latents.shape[0])
                else:
                    guidance = None

                pred = self.transformer(
                    img = latents,
                    txt = prompt_embeds,
                    y = pooled_prompt_embeds,
                    timesteps = timestep / 1000,
                    img_ids = latent_image_ids,
                    txt_ids = text_ids,
                    guidance = guidance,
                )

                latents = self.scheduler.step(pred, t, latents)

        if output_type == "latent":
            return latents

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.params.scaling_factor) + self.vae.params.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

        return image





