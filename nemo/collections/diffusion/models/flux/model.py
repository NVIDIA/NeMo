# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Callable

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import openai_gelu
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.optimizer import OptimizerConfig
from torch import nn
from torch.nn import functional as F

from nemo.collections.diffusion.models.dit.dit_layer_spec import (
    AdaLNContinuous,
    FluxSingleTransformerBlock,
    MMDiTLayer,
    get_flux_double_transformer_engine_spec,
    get_flux_single_transformer_engine_spec,
)
from nemo.collections.diffusion.models.flux.layers import EmbedND, MLPEmbedder, TimeStepEmbedder


##############

import pytorch_lightning as L
from nemo.collections.llm import fn
from nemo.lightning import io
from megatron.core.transformer.transformer_config import TransformerConfig
from typing import Optional
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.collections.diffusion.vae.autoencoder import AutoEncoderParams,AutoEncoder
from nemo.utils import logging
from nemo.collections.diffusion.encoders.conditioner import FrozenCLIPEmbedder, FrozenT5Embedder
from nemo.collections.diffusion.sampler.flow_matching.flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from nemo.lightning import megatron_parallel as mp
from megatron.core.transformer.enums import ModelType
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
import numpy as np



def flux_data_step(dataloader_iter):
    # latents = torch.randn(4096, b, 64)
    # prompt_embeds = torch.randn(256, b, 4096)
    # pooled_prompt_embeds = torch.randn(b, 768)
    # timestep = torch.randn(b)
    # latent_image_ids = torch.randn(b, 4096, 3)
    # text_ids = torch.randn(b, 256, 3)
    # guidance = torch.randn(b)
    #
    # return latents, prompt_embeds, pooled_prompt_embeds, timestep, latent_image_ids, text_ids, guidance
    batch = next(dataloader_iter)
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    _batch['loss_mask'] = torch.Tensor([1.0]).cuda(non_blocking=True)
    return _batch


@dataclass
class FluxConfig(TransformerConfig, io.IOMixin):
    ## transformer related
    num_layers: int = 1 # dummy setting
    num_joint_layers: int = 19
    num_single_layers: int = 38
    hidden_size: int = 3072
    num_attention_heads: int = 24
    activation_func: Callable = openai_gelu
    add_qkv_bias: bool = True
    in_channels: int = 64
    context_dim: int = 4096
    model_channels: int = 256
    patch_size: int = 1
    guidance_embed: bool = False
    vec_in_dim: int = 768
    rotary_interleaved: bool = True
    layernorm_epsilon: float = 1e-06
    hidden_dropout: float = 0
    attention_dropout: float = 0
    use_cpu_initialization: bool = True

    guidance_scale: float = 3.5
    data_step_fn: Callable = flux_data_step

    def configure_model(self):
        model = Flux(
            config = self
        )
        return model

@dataclass
class FluxModelParams:
    flux_params: FluxConfig
    vae_params: AutoEncoderParams
    clip_params: dict | None
    t5_params: dict | None
    scheduler_params: dict | None
    device: str | torch.device






class Flux(VisionModule):
    def __init__(self, config: FluxConfig):

        super().__init__(config)
        self.out_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.guidance_embed = config.guidance_embed


        self.pos_embed = EmbedND(dim=self.hidden_size, theta=10000, axes_dim=[16, 56, 56])
        self.img_embed = nn.Linear(config.in_channels, self.hidden_size)
        self.txt_embed = nn.Linear(config.context_dim, self.hidden_size)
        self.timestep_embedding = TimeStepEmbedder(config.model_channels, self.hidden_size)
        self.vector_embedding = MLPEmbedder(in_dim=config.vec_in_dim, hidden_dim=self.hidden_size)
        if config.guidance_embed:
            self.guidance_embedding = (
                MLPEmbedder(in_dim=config.model_channels, hidden_dim=self.hidden_size)
                if config.guidance_embed
                else nn.Identity()
            )

        self.double_blocks = nn.ModuleList(
            [
                MMDiTLayer(
                    config=config,
                    submodules=get_flux_double_transformer_engine_spec().submodules,
                    layer_number=i,
                    context_pre_only=False,
                )
                for i in range(config.num_joint_layers)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    config=config,
                    submodules=get_flux_single_transformer_engine_spec().submodules,
                    layer_number=i,
                )
                for i in range(config.num_single_layers)
            ]
        )

        self.norm_out = AdaLNContinuous(config=config, conditioning_embedding_dim=self.hidden_size)
        self.proj_out = nn.Linear(self.hidden_size, self.patch_size * self.patch_size * self.out_channels, bias=True)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor = None,
        y: torch.Tensor = None,
        timesteps: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        controlnet_double_block_samples: torch.Tensor = None,
        controlnet_single_block_samples: torch.Tensor = None,
    ):
        hidden_states = self.img_embed(img)
        encoder_hidden_states = self.txt_embed(txt)

        timesteps = timesteps.to(img.dtype) * 1000
        vec_emb = self.timestep_embedding(timesteps)

        if guidance is not None:
            vec_emb = vec_emb + self.guidance_embedding(self.timestep_embedding.time_proj(guidance * 1000))
        vec_emb = vec_emb + self.vector_embedding(y)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        rotary_pos_emb = self.pos_embed(ids)
        for id_block, block in enumerate(self.double_blocks):
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                emb=vec_emb,
            )

            if controlnet_double_block_samples is not None:
                interval_control = len(self.double_blocks) / len(controlnet_double_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_double_block_samples[id_block // interval_control]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)

        for id_block, block in enumerate(self.single_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                emb=vec_emb,
            )

            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = torch.cat([
                    hidden_states[:encoder_hidden_states.shape[0]],
                    hidden_states[encoder_hidden_states.shape[0]:] + controlnet_single_block_samples[id_block // interval_control]
                ])

        hidden_states = hidden_states[encoder_hidden_states.shape[0]:, ...]

        hidden_states = self.norm_out(hidden_states, vec_emb)
        output = self.proj_out(hidden_states)

        return output






class MegatronFluxModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    def __init__(
        self,
        config: FluxModelParams,
        optim: Optional[OptimizerModule] = None,
    ):

        self.config = config.flux_params
        super().__init__()
        self._training_loss_reduction = None
        self._validation_loss_reduction = None

        self.vae_params = config.vae_params
        self.clip_params = config.clip_params
        self.t5_params = config.t5_params
        self.scheduler_params = config.scheduler_params
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=False))
        self.optim.connect(self)
        self.model_type = ModelType.encoder_or_decoder
        self.text_precached = (self.t5_params is None or self.clip_params is None)
        self.image_precached = (self.vae_params is None)




    def configure_model(self):
        if not hasattr(self, "module"):
            self.module = self.config.configure_model()
        self.configure_vae(self.vae_params)
        self.configure_scheduler(self.scheduler_params)
        self.configure_text_encoders(self.clip_params, self.t5_params)
        for name, param in self.module.named_parameters():
            if self.config.num_single_layers == 0:
                if 'context' in name or 'added' in name:
                    param.requires_grad = False
            # When getting rid of concat, the projection bias in attention and mlp bias are identical
            # So this bias is skipped and not included in the computation graph
            if 'single_blocks' in name and 'self_attention.linear_proj.bias' in name:
                param.requires_grad = False


    def configure_scheduler(self, scheduler):
        self.scheduler = FlowMatchEulerDiscreteScheduler(**scheduler)


    def configure_vae(self, vae):
        if isinstance(vae, nn.Module):
            self.vae = vae.eval().cuda()
            self.vae_scale_factor = 2 ** (len(self.vae.params.ch_mult))
            for param in self.vae.parameters():
                param.requires_grad = False
        elif isinstance(vae, AutoEncoderParams):
            self.vae = AutoEncoder(vae).eval().cuda()
            self.vae_scale_factor = 2 ** (len(vae.ch_mult))
            for param in self.vae.parameters():
                param.requires_grad = False
        else:
            logging.info("Vae not provided, assuming the image input is precached...")
            self.vae = None
            self.vae_scale_factor = 16



    def configure_text_encoders(self, clip, t5):
        if isinstance(clip, nn.Module):
            self.clip = clip
        elif isinstance(clip, dict):
            self.clip = FrozenCLIPEmbedder(**clip, device=torch.cuda.current_device())
        else:
            logging.info("CLIP encoder not provided, assuming the text embeddings is precached...")
            self.clip = None

        if isinstance(t5, nn.Module):
            self.t5 = t5
        elif isinstance(t5, dict):
            self.t5 = FrozenT5Embedder(**t5, device=torch.cuda.current_device())
        else:
            logging.info("T5 encoder not provided, assuming the text embeddings is precached...")
            self.t5 = None

    def data_step(self, dataloader_iter):
        return self.config.data_step_fn(dataloader_iter)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)
    def forward_step(self, batch) -> torch.Tensor:
        if self.optim.config.bf16:
            self.autocast_dtype = torch.bfloat16
        elif self.optim.config.fp16:
            self.autocast_dtype = torch.float
        else:
            self.autocast_dtype = torch.float32

        if self.image_precached:
            latents = batch['latents'].cuda(non_blocking=True)
        else:
            img = batch['images'].cuda(non_blocking=True)
            latents = self.vae.encode(img).to(dtype=self.autocast_dtype)
        latents, noise, packed_noisy_model_input, latent_image_ids, guidance_vec, timesteps = self.prepare_image_latent(latents)
        if self.text_precached:
            prompt_embeds = batch['prompt_embeds'].cuda(non_blocking=True).transpose(0, 1)
            pooled_prompt_embeds = batch['pooled_prompt_embeds'].cuda(non_blocking=True)
            text_ids = batch['text_ids'].cuda(non_blocking=True)
        else:
            txt = batch['txt']
            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(txt, device = latents.device, dtype=latents.dtype)
        with torch.cuda.amp.autocast(
                self.autocast_dtype in (torch.half, torch.bfloat16),
                dtype=self.autocast_dtype,
            ):
            noise_pred = self.forward(
                img=packed_noisy_model_input,
                txt=prompt_embeds,
                y=pooled_prompt_embeds,
                timesteps = timesteps/1000,
                img_ids = latent_image_ids,
                txt_ids = text_ids,
                guidance = guidance_vec,
            )

        noise_pred = self._unpack_latents(
            noise_pred.transpose(0, 1),
            int(latents.shape[2] * self.vae_scale_factor // 2),
            int(latents.shape[3] * self.vae_scale_factor // 2),
            vae_scale_factor=self.vae_scale_factor).transpose(0, 1)

        target = noise - latents
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        return loss

    def encode_prompt(self, prompt, device='cuda', dtype=torch.float32):
        prompt_embeds = self.t5(prompt).transpose(0, 1)
        _, pooled_prompt_embeds = self.clip(prompt)
        text_ids = torch.zeros(prompt_embeds.shape[1], prompt_embeds.shape[0], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds.to(dtype=dtype), text_ids

    def compute_density_for_timestep_sampling(
            self,
            weighting_scheme: str,
            batch_size: int,
            logit_mean: float = 0.0,
            logit_std: float = 1.0,
            mode_scale: float = 1.29
    ):
        """
        Compute the density for sampling the timesteps when doing SD3 training.

        Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

        SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
        """
        if weighting_scheme == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
            u = torch.nn.functional.sigmoid(u)
        elif weighting_scheme == "mode":
            u = torch.rand(size=(batch_size,), device="cpu")
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=(batch_size,), device="cpu")
        return u
    def prepare_image_latent(self, latents):
        latent_image_ids = self._prepare_latent_image_ids(
            latents.shape[0],
            latents.shape[2],
            latents.shape[3],
            latents.device,
            latents.dtype,
        )

        noise = torch.randn_like(latents, device=latents.device, dtype=latents.dtype)
        batch_size = latents.shape[0]
        u = self.compute_density_for_timestep_sampling(
            "logit_normal",
            batch_size,
        )
        indices = (u * self.scheduler.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(device=latents.device)

        sigmas = self.scheduler.sigmas.to(device=latents.device, dtype=latents.dtype)
        schduler_timesteps = self.scheduler.timesteps.to(device=latents.device)
        step_indices = [(schduler_timesteps == t).nonzero().item() for t in timesteps]
        timesteps = timesteps.to(dtype=latents.dtype)
        sigma = sigmas[step_indices].flatten()

        if len(sigma.shape) < latents.ndim:
            sigma = sigma.unsqueeze(-1)

        noisy_model_input = (1.0 - sigma) * latents + sigma * noise
        packed_noisy_model_input = self._pack_latents(
            noisy_model_input,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )


        if self.config.guidance_embed:
            guidance_vec = torch.full(
                            (noisy_model_input.shape[0],),
                            self.config.guidance_scale,
                            device=latents.device,
                            dtype=latents.dtype,
                        )
        else:
            guidance_vec = None

        return latents.transpose(0,1), noise.transpose(0,1), packed_noisy_model_input.transpose(0,1), latent_image_ids, guidance_vec, timesteps

    def _unpack_latents(self, latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

        return latents

    def _prepare_latent_image_ids(self, batch_size: int, height: int, width: int, device: torch.device,
                                  dtype: torch.dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    def _pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    def set_input_tensor(self, tensor):
        pass

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReduction()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MaskedTokenLossReduction(validation_step=True)

        return self._validation_loss_reduction



