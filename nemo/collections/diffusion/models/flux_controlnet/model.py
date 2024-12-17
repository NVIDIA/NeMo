from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.nn import functional as F

from nemo.collections.diffusion.models.dit.dit_layer_spec import (
    AdaLNContinuous,
    FluxSingleTransformerBlock,
    MMDiTLayer,
    get_flux_double_transformer_engine_spec,
    get_flux_single_transformer_engine_spec,
)
from nemo.collections.diffusion.models.flux.layers import EmbedND, MLPEmbedder, TimeStepEmbedder
from nemo.collections.diffusion.models.flux.model import FluxModelParams, MegatronFluxModel
from nemo.collections.diffusion.models.flux_controlnet.layers import ControlNetConditioningEmbedding
from nemo.lightning import io


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def flux_controlnet_data_step(dataloader_iter):
    batch = next(dataloader_iter)
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    _batch['loss_mask'] = torch.Tensor([1.0]).cuda(non_blocking=True)
    return _batch


@dataclass
class FluxControlNetConfig(TransformerConfig, io.IOMixin):
    num_layers: int = 1  # dummy setting
    patch_size: int = 1
    in_channels: int = 64
    num_joint_layers: int = 4
    num_single_layers: int = 10
    hidden_size: int = 3072
    num_attention_heads: int = 24
    vec_in_dim: int = 768
    context_dim: int = 4096
    guidance_embed: bool = True
    num_mode: int = None
    model_channels: int = 256
    conditioning_embedding_channels: int = None
    rotary_interleaved: bool = True
    layernorm_epsilon: float = 1e-06
    hidden_dropout: float = 0
    attention_dropout: float = 0
    add_qkv_bias: bool = True
    use_cpu_initialization: bool = True

    load_from_flux_transformer: bool = True
    guidance_scale: float = 3.5

    data_step_fn: Callable = flux_controlnet_data_step


class FluxControlNet(VisionModule):
    def __init__(self, config: FluxControlNetConfig):
        super().__init__(config)
        self.out_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size

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

        # ContolNet Blocks
        self.controlnet_double_blocks = nn.ModuleList()
        for _ in range(config.num_joint_layers):
            self.controlnet_double_blocks.append(zero_module(nn.Linear(self.hidden_size, self.hidden_size)))

        self.controlnet_single_blocks = nn.ModuleList()
        for _ in range(config.num_single_layers):
            self.controlnet_single_blocks.append(zero_module(nn.Linear(self.hidden_size, self.hidden_size)))

        if config.conditioning_embedding_channels is not None:
            self.input_hint_block = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=conditioning_embedding_channels, block_out_channels=(16, 16, 16, 16)
            )
            self.controlnet_x_embedder = torch.nn.Linear(config.in_channels, self.hidden_size)
        else:
            self.input_hint_block = None
            self.controlnet_x_embedder = zero_module(torch.nn.Linear(config.in_channels, self.hidden_size))

    def load_from_flux_transformer(self, flux):
        self.pos_embed.load_state_dict(flux.pos_embed.state_dict())
        self.img_embed.load_state_dict(flux.img_embed.state_dict())
        self.txt_embed.load_state_dict(flux.txt_embed.state_dict())
        self.timestep_embedding.load_state_dict(flux.timestep_embedding.state_dict())
        self.vector_embedding.load_state_dict(flux.vector_embedding.state_dict())
        self.double_blocks.load_state_dict(flux.double_blocks.state_dict(), strict=False)
        self.single_blocks.load_state_dict(flux.single_blocks.state_dict(), strict=False)

    def forward(
        self,
        img: torch.Tensor,
        controlnet_cond: torch.Tensor,
        txt: torch.Tensor = None,
        y: torch.Tensor = None,
        timesteps: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        conditioning_scale: float = 1.0,
    ):
        hidden_states = self.img_embed(img)
        encoder_hidden_states = self.txt_embed(txt)
        if self.input_hint_block is not None:
            controlnet_cond = self.input_hint_block(controlnet_cond)
            batch_size, channels, height_pw, width_pw = controlnet_cond.shape
            height = height_pw // self.config.patch_size
            width = width_pw // self.config.patch_size
            controlnet_cond = controlnet_cond.reshape(
                batch_size, channels, height, self.patch_size, width, self.patch_size
            )
            controlnet_cond = controlnet_cond.permute(0, 2, 4, 1, 3, 5)
            controlnet_cond = controlnet_cond.reshape(batch_size, height * width, -1)

        hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_cond)

        timesteps = timesteps.to(img.dtype) * 1000
        vec_emb = self.timestep_embedding(timesteps)
        if guidance is not None:
            vec_emb = vec_emb + self.guidance_embedding(self.timestep_embedding.time_proj(guidance * 1000))
        vec_emb = vec_emb + self.vector_embedding(y)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        rotary_pos_emb = self.pos_embed(ids)

        double_block_samples = ()
        for id_block, block in enumerate(self.double_blocks):
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                emb=vec_emb,
            )
            double_block_samples = double_block_samples + (hidden_states,)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)

        single_block_samples = ()
        for id_block, block in enumerate(self.single_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                emb=vec_emb,
            )
            single_block_samples = single_block_samples + (hidden_states[encoder_hidden_states.shape[0] :, ...],)

        controlnet_double_block_samples = ()
        for double_block_sample, control_block in zip(double_block_samples, self.controlnet_double_blocks):
            double_block_sample = control_block(double_block_sample)
            controlnet_double_block_samples += (double_block_sample,)

        controlnet_single_block_samples = ()
        for single_block_sample, control_block in zip(single_block_samples, self.controlnet_single_blocks):
            single_block_sample = control_block(single_block_sample)
            controlnet_single_block_samples += (single_block_sample,)

        controlnet_double_block_samples = [sample * conditioning_scale for sample in controlnet_double_block_samples]
        controlnet_single_block_samples = [sample * conditioning_scale for sample in controlnet_single_block_samples]

        controlnet_double_block_samples = (
            None if len(controlnet_double_block_samples) == 0 else controlnet_double_block_samples
        )
        controlnet_single_block_samples = (
            None if len(controlnet_single_block_samples) == 0 else controlnet_single_block_samples
        )

        return controlnet_double_block_samples, controlnet_single_block_samples


class FluxControlnetForwardWrapper(VisionModule):
    def __init__(self, flux_config: FluxModelParams, flux_controlnet_config: FluxControlNetConfig):
        super().__init__(flux_config)

        self.flux = self.config.configure_model()
        for param in self.flux.parameters():
            param.requires_grad = False

        self.flux_controlnet = FluxControlNet(flux_controlnet_config)
        if flux_controlnet_config.load_from_flux_transformer:
            self.flux_controlnet.load_from_flux_transformer(self.flux)


class MegatronFluxControlNetModel(MegatronFluxModel):
    def __init__(self, flux_config: FluxModelParams, flux_controlnet_config: FluxControlNetConfig):
        super().__init__(flux_config)
        self.flux_controlnet_config = flux_controlnet_config
        self.optim.connect(self)

    def configure_model(self):
        if not hasattr(self, "module"):
            self.module = FluxControlnetForwardWrapper(self.config, self.flux_controlnet_config)

            self.configure_vae(self.vae_params)
            self.configure_scheduler(self.scheduler_params)
            self.configure_text_encoders(self.clip_params, self.t5_params)
            ## Have to disable requiring grads for those params not getting one, otherwise custom fsdp fails at assert
            ## when there is no single layer, encoder_hidden_states related params are not included in computation graph
            for name, param in self.module.named_parameters():
                if self.flux_controlnet_config.num_single_layers == 0:
                    if 'context' in name or 'added' in name:
                        param.requires_grad = False
                # When getting rid of concat, the projection bias in attention and mlp bias are identical
                # So this bias is skipped and not included in the computation graph
                if 'single_blocks' in name and 'self_attention.linear_proj.bias' in name:
                    param.requires_grad = False

    def data_step(self, dataloader_iter):
        return self.flux_controlnet_config.data_step_fn(dataloader_iter)

    def forward(self, *args, **kwargs):
        # FSDP module -> Bfloat16 module -> ForwardWrapper -> flux controlnet
        return self.module.module.module.flux_controlnet(*args, **kwargs)

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
            control_latents = batch['control_latents'].cuda(non_blocking=True)
        else:
            img = batch['images'].cuda(non_blocking=True)
            latents = self.vae.encode(img).to(dtype=self.autocast_dtype)
            hint = batch['hint'].cuda(non_blocking=True)
            control_latents = self.vae.encode(hint).to(dtype=self.autocast_dtype)

        latents, noise, packed_noisy_model_input, latent_image_ids, guidance_vec, timesteps = (
            self.prepare_image_latent(latents)
        )
        control_image = self._pack_latents(
            control_latents,
            batch_size=control_latents.shape[0],
            num_channels_latents=control_latents.shape[1],
            height=control_latents.shape[2],
            width=control_latents.shape[3],
        ).transpose(0, 1)
        if self.text_precached:
            prompt_embeds = batch['prompt_embeds'].cuda(non_blocking=True).transpose(0, 1)
            pooled_prompt_embeds = batch['pooled_prompt_embeds'].cuda(non_blocking=True)
            text_ids = batch['text_ids'].cuda(non_blocking=True)
        else:
            txt = batch['txt']
            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                txt, device=latents.device, dtype=latents.dtype
            )

        with torch.cuda.amp.autocast(
            self.autocast_dtype in (torch.half, torch.bfloat16),
            dtype=self.autocast_dtype,
        ):
            controlnet_double_block_samples, controlnet_single_block_samples = self.forward(
                img=packed_noisy_model_input,
                controlnet_cond=control_image,
                txt=prompt_embeds,
                y=pooled_prompt_embeds,
                timesteps=timesteps / 1000,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=guidance_vec,
            )
            noise_pred = self.module.module.module.flux(
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
            noise_pred = self._unpack_latents(
                noise_pred.transpose(0, 1),
                int(latents.shape[2] * self.vae_scale_factor // 2),
                int(latents.shape[3] * self.vae_scale_factor // 2),
                vae_scale_factor=self.vae_scale_factor,
            ).transpose(0, 1)
            target = noise - latents
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
            return loss
