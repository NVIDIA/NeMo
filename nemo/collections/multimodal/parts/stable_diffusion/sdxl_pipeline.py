from nemo.core.classes.common import Serialization
from nemo.core.config import hydra_runner
from typing import List, Optional, Tuple, Union
import torch
from tqdm import tqdm
from nemo.collections.multimodal.models.stable_diffusion.diffusion_engine import DiffusionEngine

class StableDiffusionXL(Serialization):
    def __init__(self, cfg):
        super().__init__()
        model = StableDiffusionXL.from_config_dict(cfg.model)
        # self.device = torch.device('cuda') if cfg.device != 'cpu' else torch.device('cpu')
        # self.vae = StableDiffusionXL.from_config_dict(cfg.model.first_stage_config).to(self.device)
        # self.text_encoder = StableDiffusionXL.from_config_dict(cfg.model.text_encoder).to(self.device)
        # self.text_encoder2 = StableDiffusionXL.from_config_dict(cfg.model.text_encoder2).to(self.device)
        # self.unet = StableDiffusionXL.from_config_dict(cfg.model.unet_config).to(self.device)
        # self.scheduler = StableDiffusionXL.from_config_dict(cfg.model.scheduler)
        # self.vae_scale_factor = 2**(len(cfg.model.first_stage_config.ddconfig.ch_mult) - 1)
        # self.num_channels_latents = cfg.model.unet_config.in_channels


    def encode_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance=True):
        prompt_embeds = None
        uncond_prompt_embeds = None

        prompt_embeds_list = []
        text_encoders = [self.text_encoder, self.text_encoder2]
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(promptm, list):
            batch_size = len(prompt)


        for text_encoder in text_encoders:
            prompt_embeds = text_encoder.encode(prompt)

            prompt_embed, pooled_prompt_embed = prompt_embeds

            bs_embed, seq_len, _ = prompt_embed.shape
            prompt_embed = prompt_embed.repeat(1, num_images_per_prompt, 1)
            prompt_embed = prompt_embed.view(bs_embed * num_images_per_prompt, seq_len, -1)

            prompt_embeds_list.append(prompt_embed)

        prompt_embeds = torch.concat(prompt_embeds_list, dim = -1)

        if do_classifier_free_guidance:
            uc_prompt = ""
            uncond_prompt_embeds_list = []
            for text_encoder in text_encoders:
                uncond_prompt_embeds = text_encoder.encode(uc_prompt)

                uncond_prompt_embed, uncond_pooled_prompt_embed = uncond_prompt_embeds


                seq_len = uncond_prompt_embed.shape[1]
                uncond_prompt_embed = uncond_prompt_embed.repeat(1, num_images_per_prompt, 1)
                uncond_prompt_embed = uncond_prompt_embed.view(
                    batch_size * num_images_per_prompt, seq_len, -1
                )
                uncond_prompt_embeds_list.append(uncond_prompt_embed)

        uncond_prompt_embeds = torch.concat(uncond_prompt_embeds_list, dim = -1)

        bs_embed = pooled_prompt_embed.shape[0]
        pooled_prompt_embeds = pooled_prompt_embed.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        uncond_pooled_prompt_embeds = uncond_pooled_prompt_embed.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        return prompt_embeds, uncond_prompt_embeds, pooled_prompt_embeds, uncond_pooled_prompt_embeds

    def rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
                self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder2.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def __call__(self, prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil"):

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        height = height
        width = width
        do_classifier_free_guidance = (guidance_scale > 1.0)

        generator = None

        prompt_embeds, uncond_prompt_embeds, pooled_prompt_embeds, uncond_pooled_prompt_embeds = self.encode_prompt(
            prompt,
            num_images_per_prompt = num_images_per_prompt,
            do_classifier_free_guidance = do_classifier_free_guidance)

        self.scheduler.set_timesteps(num_inference_steps, device = self.device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.num_channels_latents
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
        )

        iterator = tqdm(timesteps, desc='Decoding image', total=num_inference_steps)
        for i, t in enumerate(iterator):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t
            )

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            latents = self.scheduler.step(noise_pred, t, latents, eta = eta)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae_scaling_factor, return_dict=False)[0]
        else:
            image = latents

        return image




@hydra_runner(config_path='/opt/NeMo/examples/multimodal/generative/stable_diffusion/conf', config_name='sdxl_infer')
def main(cfg):
    x = DiffusionEngine(cfg.model)
    import pdb;pdb.set_trace();

if __name__ == "__main__":
    main()
