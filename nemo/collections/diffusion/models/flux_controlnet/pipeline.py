from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL.Image import Image
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from nemo.collections.diffusion.encoders.conditioner import FrozenCLIPEmbedder, FrozenT5Embedder
from nemo.collections.diffusion.models.flux.model import Flux, FluxConfig
from nemo.collections.diffusion.models.flux_controlnet.model import FluxControlNet, FluxControlNetConfig
from nemo.collections.diffusion.sampler.flow_matching.flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from nemo.collections.diffusion.vae.autoencoder import AutoEncoder, AutoEncoderParams
from nemo.utils import logging


class FluxControlNetPipeline(nn.Module):
    def __init__(
        self,
        flux_transformer: Union[Flux, FluxConfig],
        scheduler: Union[FlowMatchEulerDiscreteScheduler, Dict],
        flux_controlnet: Union[FluxControlNet, FluxControlNetConfig],
        vae: Optional[Union[AutoEncoder, AutoEncoderParams]] = None,
        t5: Optional[Union[FrozenT5Embedder, Dict]] = None,
        clip: Optional[Union[FrozenCLIPEmbedder, Dict]] = None,
    ):
        self.flux_transformer = flux_transformer
        self.scheduler = scheduler
        self.flux_controlnet = flux_controlnet
        self.vae = vae
        self.t5 = t5
        self.clip = clip
        self.configure_modules()

        self.vae_scale_factor = 2 ** (len(self.vae.params.ch_mult))

    def load_from_pretrained(self, ckpt_path, do_convert_from_hf=True, save_converted_model=None):
        if do_convert_from_hf:
            ckpt = flux_transformer_converter(ckpt_path, self.transformer.config)
            if save_converted_model:
                save_path = os.path.join(ckpt_path, 'nemo_flux_controlnet_transformer.safetensors')
                save_safetensors(ckpt, save_path)
                logging.info(f'saving converted transformer checkpoint to {save_path}')
        else:
            ckpt = load_safetensors(ckpt_path)
        missing, unexpected = self.transformer.load_state_dict(ckpt, strict=False)
        missing = [
            k for k in missing if not k.endswith('_extra_state')
        ]  # These keys are mcore specific and should not affect the model performance
        if len(missing) > 0:
            logging.info(
                f"The folloing keys are missing during checkpoint loading, please check the ckpt provided or the image quality may be compromised.\n {missing}"
            )
            logging.info(f"Found unexepected keys: \n {unexpected}")

    def configure_modules(self):
        if isinstance(self.flux_transformer, FluxConfig):
            self.flux_transformer = Flux(self.flux_transformer)
        if isinstance(self.scheduler, dict):
            self.scheduler = FlowMatchEulerDiscreteScheduler(self.scheduler)
        if isinstance(self.flux_controlnet, FluxControlNetConfig):
            self.flux_controlnet = FluxControlNet(self.flux_controlnet)

        if isinstance(self.vae, AutoEncoderParams):
            self.vae_z_channels = self.vae.z_channels
            self.vae = AutoEncoder(self.vae)
        elif isinstance(self.vae, AutoEncoder):
            self.vae_z_channels = self.vae.params.z_channels
        elif self.vae is None:
            self.vae_z_channels = 16
            logging.info('VAE is not setup, assuming image data is precached')

        if isinstance(self.t5, dict):
            self.t5 = FrozenT5Embedder(self.t5)
        elif self.t5 is None:
            logging.info('T5 is not setup, assuming text prompt is precached as embeddings')

        if isinstance(self.clip, dict):
            self.clip = FrozenCLIPEmbedder(self.clip)
        elif self.clip is None:
            logging.info('CLIP is not setup, assuming pooled_text_prompt is precached as embeddings')

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

    def process_image(
        self,
        control_image,
        height,
        width,
    ):
        if not isinstance(control_image, list):
            control_image = [control_image]

        if isinstance(control_image[0], Image):
            orig_h, orig_w = control_image[0].height, control_image[0].width
            if orig_h != height or orig_w != width:
                control_image = [image.resize(height=height, width=width) for image in control_image]

            # pil to numpy
            control_image = [np.array(control_image).astype(np.float32) / 255.0 for image in control_image]
            control_image = np.stack(control_image, axis=0)

            # numpy to torch
            if control_image.ndim == 3:
                control_image = control_image[..., None]

            control_image = torch.from_numpy(control_image.transpose(0, 3, 1, 2))

        elif isinstance(control_image[0], torch.Tensor):
            control_image = (
                torch.cat(control_image, axis=0) if control_image[0].ndim == 4 else torch.stack(control_image, axis=0)
            )
            channel = control_image.shape[1]

            if channel == self.vae_z_channels:
                return control_image

            orig_h, orig_w = control_image[0].shape[2], control_image[0].shape[3]
            if orig_h != height or orig_w != width:
                control_image = [
                    torch.nn.functional.interpolate(
                        image,
                        size=(height, width),
                    )
                    for image in control_image
                ]

        if control_image.min() < 0:
            logging.warning('`image` with value range [{image.min()},{image.max()} is passed')
        else:
            # Do normalization
            control_image = control_image * 2.0 - 1.0

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
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_image: Union[torch.Tensor, Image] = None,
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

        control_image = self.process_image(control_image)
        control_image_bs = control_image.shape[0]

        if control_image_bs == 1:
            repeat = batch_size
        else:
            repeat = num_images_per_prompt

        control_image = torch.repeat_interleave(control_image, repeat, dim=0)

        if self.flux_controlnet.input_hint_block is None:
            if self.vae is None:
                assert (
                    control_image.shape == (1, 2, 3, 4),
                    "no vae is found and no hint blocks in controlnet, the input control image must be packed latents",
                )
            else:
                control_image = self.vae.encode(control_image)

                # pack
                height_control_image, width_control_image = control_image.shape[2:]
                control_image = self._pack_latents(
                    control_image,
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height_control_image,
                    width_control_image,
                )

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

        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(self.controlnet, FluxControlNetModel) else keeps)

        with torch.no_grad():
            for i, t in tqdm(enumerate(timesteps)):
                timestep = t.expand(latents.shape[1]).to(device=latents.device, dtype=latents.dtype)

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                with torch.autocast(device_type='cuda', dtype=latents.dtype):
                    if self.flux_controlnet.guidance_embed:
                        guidance = torch.tensor([guidance_scale], device=device).expand(latents.shape[1])
                    else:
                        guidance = None

                    controlnet_double_block_samples, controlnet_single_block_samples = self.flux_controlnet(
                        img=packed_noisy_model_input,
                        controlnet_cond=control_image,
                        txt=prompt_embeds,
                        y=pooled_prompt_embeds,
                        timesteps=timesteps / 1000,
                        img_ids=latent_image_ids,
                        txt_ids=text_ids,
                        guidance=guidance_vec,
                    )

                    if self.flux_transformer.guidance_embed:
                        guidance = torch.tensor([guidance_scale], device=device).expand(latents.shape[1])
                    else:
                        guidance = None
                    noise_pred = self.forward(
                        img=packed_noisy_model_input,
                        txt=prompt_embeds,
                        y=pooled_prompt_embeds,
                        timesteps=timesteps / 1000,
                        img_ids=latent_image_ids,
                        txt_ids=text_ids,
                        guidance=guidance_vec,
                        controlnet_double_block_samples=controlnet_double_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                    )
                    latents = self.scheduler.step(pred, t, latents)[0]

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
                    image = FluxControlNetPipeline.denormalize(image)
                    image = FluxControlNetPipeline.torch_to_numpy(image)
                    image = FluxControlNetPipeline.numpy_to_pil(image)
                if save_to_disk:
                    print('Saving to disk')
                    assert len(image) == int(len(prompt) * num_images_per_prompt)
                    prompt = [p[:40] + f'_{idx}' for p in prompt for idx in range(num_images_per_prompt)]
                    for file_name, image in zip(prompt, image):
                        image.save(f'{file_name}.png')

                return image
