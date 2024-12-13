# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import nn
from tqdm import tqdm

from nemo.collections.diffusion.encoders.conditioner import FrozenCLIPEmbedder, FrozenT5Embedder
from nemo.collections.diffusion.models.flux.model import Flux
from nemo.collections.diffusion.sampler.flow_matching.flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from nemo.collections.diffusion.utils.flux_ckpt_converter import flux_transformer_converter
from nemo.collections.diffusion.utils.flux_pipeline_utils import FluxModelParams
from nemo.collections.diffusion.vae.autoencoder import AutoEncoder
from nemo.collections.diffusion.models.flux_controlnet.model import FluxControlNetConfig, FluxControlNet
from nemo.utils import logging


class FluxInferencePipeline(nn.Module):
    def __init__(self, params: FluxModelParams):
        super().__init__()
        self.device = params.device
        params.clip_params['device'] = self.device
        params.t5_params['device'] = self.device

        self.vae = AutoEncoder(params.vae_params).to(self.device).eval()
        self.clip_encoder = FrozenCLIPEmbedder(**params.clip_params)
        self.t5_encoder = FrozenT5Embedder(**params.t5_params)
        self.transformer = Flux(params.flux_params).to(self.device).eval()
        self.vae_scale_factor = 2 ** (len(self.vae.params.ch_mult))
        self.scheduler = FlowMatchEulerDiscreteScheduler(**params.scheduler_params)
        self.params = params

    def load_from_pretrained(self, ckpt_path, do_convert_from_hf=True, save_converted_model_to=None):
        if do_convert_from_hf:
            ckpt = flux_transformer_converter(ckpt_path, self.transformer.config)
            if save_converted_model_to is not None:
                save_path = os.path.join(save_converted_model_to, 'nemo_flux_transformer.safetensors')
                save_safetensors(ckpt, save_path)
                logging.info(f'saving converted transformer checkpoint to {save_path}')
        else:
            ckpt = load_safetensors(ckpt_path)
        missing, unexpected = self.transformer.load_state_dict(ckpt, strict=False)
        missing = [
            k for k in missing if not k.endswith('_extra_state')
        ]
        # These keys are mcore specific and should not affect the model performance
        if len(missing) > 0:
            logging.info(
                f"The following keys are missing during checkpoint loading, please check the ckpt provided or the image quality may be compromised.\n {missing}"
            )
            logging.info(f"Found unexepected keys: \n {unexpected}")

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
        if prompt is not None:
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("Either prompt or prompt_embeds must be provided.")
        if device == 'cuda' and self.t5_encoder.device != device:
            self.t5_encoder.to(device)
        if prompt_embeds is None:
            prompt_embeds = self.t5_encoder(prompt, max_sequence_length=max_sequence_length)
        seq_len = prompt_embeds.shape[1]
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1).to(dtype=dtype)

        if device == 'cuda' and self.clip_encoder.device != device:
            self.clip_encoder.to(device)
        if pooled_prompt_embeds is None:
            _, pooled_prompt_embeds = self.clip_encoder(prompt)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1).to(dtype=dtype)

        dtype = dtype if dtype is not None else self.t5_encoder.dtype
        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

        return prompt_embeds.transpose(0, 1), pooled_prompt_embeds, text_ids

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
        height = 2 * int(height) // self.vae_scale_factor
        width = 2 * int(width) // self.vae_scale_factor

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

        return latents.transpose(0, 1), latent_image_ids

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

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @staticmethod
    def torch_to_numpy(images):
        numpy_images = images.float().cpu().permute(0, 2, 3, 1).numpy()
        return numpy_images

    @staticmethod
    def denormalize(image):
        return (image / 2 + 0.5).clamp(0, 1)

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
        device: torch.device = 'cuda',
        dtype: torch.dtype = torch.float32,
        save_to_disk: bool = True,
        offload: bool = False,
    ):
        assert device == 'cuda', 'Transformer blocks in Mcore must run on cuda devices'

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
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
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        if offload:
            self.t5_encoder.to('cpu')
            self.clip_encoder.to('cpu')
            torch.cuda.empty_cache()

        ## prepare image latents
        num_channels_latents = self.transformer.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents, height, width, dtype, device, generator, latents
        )
        # prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[0]

        mu = FluxInferencePipeline._calculate_shift(
            image_seq_len,
            self.scheduler.base_image_seq_len,
            self.scheduler.max_image_seq_len,
            self.scheduler.base_shift,
            self.scheduler.max_shift,
        )

        self.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        timesteps = self.scheduler.timesteps

        if device == 'cuda' and device != self.device:
            self.transformer.to(device)
        with torch.no_grad():
            for i, t in tqdm(enumerate(timesteps)):
                timestep = t.expand(latents.shape[1]).to(device=latents.device, dtype=latents.dtype)
                if self.transformer.guidance_embed:
                    guidance = torch.tensor([guidance_scale], device=device).expand(latents.shape[1])
                else:
                    guidance = None
                with torch.autocast(device_type='cuda', dtype=latents.dtype):
                    pred = self.transformer(
                        img=latents,
                        txt=prompt_embeds,
                        y=pooled_prompt_embeds,
                        timesteps=timestep / 1000,
                        img_ids=latent_image_ids,
                        txt_ids=text_ids,
                        guidance=guidance,
                    )
                    latents = self.scheduler.step(pred, t, latents)[0]
            if offload:
                self.transformer.to('cpu')
                torch.cuda.empty_cache()

            if output_type == "latent":
                return latents.transpose(0, 1)
            elif output_type == "pil":
                latents = self._unpack_latents(latents.transpose(0, 1), height, width, self.vae_scale_factor)
                latents = (latents / self.vae.params.scale_factor) + self.vae.params.shift_factor
                if device == 'cuda' and device != self.device:
                    self.vae.to(device)
                with torch.autocast(device_type='cuda', dtype=latents.dtype):
                    image = self.vae.decode(latents)
                if offload:
                    self.vae.to('cpu')
                    torch.cuda.empty_cache()
                image = FluxInferencePipeline.denormalize(image)
                image = FluxInferencePipeline.torch_to_numpy(image)
                image = FluxInferencePipeline.numpy_to_pil(image)
        if save_to_disk:
            print('Saving to disk')
            assert len(image) == int(len(prompt) * num_images_per_prompt)
            prompt = [p[:40] + f'_{idx}' for p in prompt for idx in range(num_images_per_prompt)]
            for file_name, image in zip(prompt, image):
                image.save(f'{file_name}.png')

        return image


class FluxControlNetInferencePipeline(FluxInferencePipeline):
    def __init__(self, params: FluxModelParams, contorlnet_config: FluxControlNetConfig):
        super().__init__(params)
        self.flux_controlnet = FluxControlNet(contorlnet_config)

    def load_from_pretrained(self, flux_ckpt_path, controlnet_ckpt_path, do_convert_from_hf=True, save_converted_model_to=None):
        if do_convert_from_hf:
            flux_ckpt = flux_transformer_converter(flux_ckpt_path, self.transformer.config)
            flux_controlnet_ckpt = flux_transformer_converter(controlnet_ckpt_path, self.flux_controlnet.config)

            if save_converted_model_to is not None:
                save_path = os.path.join(save_converted_model_to, 'nemo_flux_transformer.safetensors')
                save_safetensors(flux_ckpt, save_path)
                logging.info(f'saving converted transformer checkpoint to {save_path}')
                save_path = os.path.join(save_converted_model_to, 'nemo_flux_controlnet_transformer.safetensors')
                save_safetensors(flux_controlnet_ckpt, save_path)
                logging.info(f'saving converted transformer checkpoint to {save_path}')
        else:
            flux_ckpt = load_safetensors(flux_ckpt_path)
            flux_controlnet_ckpt = load_safetensors(controlnet_ckpt_path)
        missing, unexpected = self.transformer.load_state_dict(flux_ckpt, strict=False)
        missing = [
            k for k in missing if not k.endswith('_extra_state')
        ]
        # These keys are mcore specific and should not affect the model performance
        if len(missing) > 0:
            logging.info(
                f"The following keys are missing during flux checkpoint loading, please check the ckpt provided or the image quality may be compromised.\n {missing}"
            )
            logging.info(f"Found unexepected keys: \n {unexpected}")

        missing, unexpected = self.flux_controlnet.load_state_dict(flux_controlnet_ckpt, strict=False)
        missing = [
            k for k in missing if not k.endswith('_extra_state')
        ]
        # These keys are mcore specific and should not affect the model performance
        if len(missing) > 0:
            logging.info(
                f"The following keys are missing during controlnet checkpoint loading, please check the ckpt provided or the image quality may be compromised.\n {missing}"
            )
            logging.info(f"Found unexepected keys: \n {unexpected}")
    def pil_to_numpy(self, images):
        if not isinstance(images, list):
            images = [images]
        images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        images = np.stack(images, axis=0)

        return images

    def numpy_to_pt(self, images: np.ndarray) -> torch.Tensor:
        if images.ndim == 3:
            images = images[..., None]

        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        return images

    def prepare_image(
            self,
            images,
            height,
            width,
            batch_size,
            num_images_per_prompt,
            device,
            dtype,
    ):
        if isinstance(images, torch.Tensor):
            pass
        else:
            orig_height, orig_width = images[0].height, images[0].width
            if height != orig_height or width != orig_width:
                images = [image.resize((width, height), resample=3) for image in images]

            images = self.pil_to_numpy(images)
            images = self.numpy_to_pt(images)
        image_batch_size = images.shape[0]
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_images_per_prompt
        images = images.repeat_interleave(repeat_by, dim=0)

        images = images.to(device=device, dtype=dtype)

        return images



    def __call__(self,
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
        device: torch.device = 'cuda',
        dtype: torch.dtype = torch.float32,
        save_to_disk: bool = True,
        offload: bool = False,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        control_image: Union[Image.Image, torch.FloatTensor] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
    ):
        assert device == 'cuda', 'Transformer blocks in Mcore must run on cuda devices'

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
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
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        if offload:
            self.t5_encoder.to('cpu')
            self.clip_encoder.to('cpu')
            torch.cuda.empty_cache()

        ## prepare image latents
        num_channels_latents = self.transformer.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents, height, width, dtype, device, generator, latents
        )

        # prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[0]

        mu = FluxInferencePipeline._calculate_shift(
            image_seq_len,
            self.scheduler.base_image_seq_len,
            self.scheduler.max_image_seq_len,
            self.scheduler.base_shift,
            self.scheduler.max_shift,
        )

        self.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        timesteps = self.scheduler.timesteps




        control_image = self.prepare_image(
            images=control_image,
            height=height,
            width=width,
            batch_size=batch_size*num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=torch.float32,
        )

        height, width = control_image.shape[-2:]
        if self.flux_controlnet.input_hint_block is None:
            if device == 'cuda' and self.device != device:
                self.vae.to(device)
            with torch.no_grad():
                control_image = self.vae.encode(control_image).to(dtype=dtype)

            height_control_image, width_control_image = control_image.shape[2:]
            control_image = self._pack_latents(
                control_image,
                batch_size*num_images_per_prompt,
                num_channels_latents,
                height_control_image,
                width_control_image,
            ).transpose(0, 1)

        controlnet_keep = []
        for i in range(len(timesteps)):
            controlnet_keep.append(1.0 - float(i / len(timesteps) < control_guidance_start or (i + 1) / len(timesteps) > control_guidance_end))
        if device == 'cuda' and device != self.device:
            self.transformer.to(device)
            self.flux_controlnet.to(device)
        with torch.no_grad():
            for i, t in tqdm(enumerate(timesteps)):
                timestep = t.expand(latents.shape[1]).to(device=latents.device, dtype=latents.dtype)
                if self.transformer.guidance_embed:
                    guidance = torch.tensor([guidance_scale], device=device).expand(latents.shape[1])
                else:
                    guidance = None

                conditioning_scale = controlnet_keep[i] * controlnet_conditioning_scale


                with torch.autocast(device_type='cuda', dtype=latents.dtype):
                    controlnet_double_block_samples, controlnet_single_block_samples = self.flux_controlnet(
                        img=latents,
                        controlnet_cond=control_image,
                        txt=prompt_embeds,
                        y=pooled_prompt_embeds,
                        timesteps=timestep/1000,
                        img_ids=latent_image_ids,
                        txt_ids=text_ids,
                        guidance=guidance,
                        conditioning_scale=conditioning_scale,
                    )
                    pred = self.transformer(
                        img=latents,
                        txt=prompt_embeds,
                        y=pooled_prompt_embeds,
                        timesteps=timestep / 1000,
                        img_ids=latent_image_ids,
                        txt_ids=text_ids,
                        guidance=guidance,
                        controlnet_double_block_samples=controlnet_double_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                    )
                    latents = self.scheduler.step(pred, t, latents)[0]
            if offload:
                self.transformer.to('cpu')
                torch.cuda.empty_cache()

            if output_type == "latent":
                return latents.transpose(0, 1)
            elif output_type == "pil":
                latents = self._unpack_latents(latents.transpose(0, 1), height, width, self.vae_scale_factor)
                latents = (latents / self.vae.params.scale_factor) + self.vae.params.shift_factor
                if device == 'cuda' and device != self.device:
                    self.vae.to(device)
                with torch.autocast(device_type='cuda', dtype=latents.dtype):
                    image = self.vae.decode(latents)
                if offload:
                    self.vae.to('cpu')
                    torch.cuda.empty_cache()
                image = FluxInferencePipeline.denormalize(image)
                image = FluxInferencePipeline.torch_to_numpy(image)
                image = FluxInferencePipeline.numpy_to_pil(image)
        if save_to_disk:
            print('Saving to disk')
            assert len(image) == int(len(prompt) * num_images_per_prompt)
            prompt = [p[:40] + f'_{idx}' for p in prompt for idx in range(num_images_per_prompt)]
            for file_name, image in zip(prompt, image):
                image.save(f'{file_name}.png')

        return image
