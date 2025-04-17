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

# pylint: disable=C0115,C0116,C0301

import importlib
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn
from typing_extensions import override

from nemo.collections.diffusion.models.dit_llama.dit_llama_model import DiTLlamaModel
from nemo.collections.diffusion.sampler.conditioner import (
    VideoConditioner,
    VideoExtendConditioner,
    VideoExtendConditionerControl,
)
from nemo.collections.diffusion.sampler.conditioner_configs import (
    ActionControlConfig,
    FPSConfig,
    ImageSizeConfig,
    NumFramesConfig,
    PaddingMaskConfig,
    TextConfig,
    VideoCondBoolConfig,
)
from nemo.collections.diffusion.sampler.cosmos.cosmos_control_diffusion_pipeline import CosmosControlDiffusionPipeline
from nemo.collections.diffusion.sampler.cosmos.cosmos_diffusion_pipeline import CosmosDiffusionPipeline
from nemo.collections.diffusion.sampler.cosmos.cosmos_extended_diffusion_pipeline import ExtendedDiffusionPipeline
from nemo.collections.diffusion.sampler.edm.edm_pipeline import EDMPipeline
from nemo.collections.llm.gpt.model.base import GPTModel
from nemo.lightning import io
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction, MegatronLossReduction
from nemo.lightning.pytorch.optim import OptimizerModule

from .dit.dit_model import DiTCrossAttentionModel
from .dit.dit_model_7b import (
    DiTControl7B,
    DiTCrossAttentionActionExtendModel7B,
    DiTCrossAttentionExtendModel7B,
    DiTCrossAttentionModel7B,
    PatchEmbed,
)
from .dit.dit_model_14b import (
    DiTCrossAttentionActionExtendModel14B,
    DiTCrossAttentionExtendModel14B,
    DiTCrossAttentionModel14B,
)


def dit_forward_step(model, batch) -> torch.Tensor:
    return model(**batch)


def dit_data_step(module, dataloader_iter):
    batch = next(dataloader_iter)[0]
    batch = get_batch_on_this_cp_rank(batch)
    batch = {k: v.to(device='cuda', non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    batch["is_preprocessed"] = True  # assume data is preprocessed

    if ('seq_len_q' in batch) and ('seq_len_kv' in batch):
        cu_seqlens = batch['seq_len_q'].cumsum(dim=0).to(torch.int32)
        zero = torch.zeros(1, dtype=torch.int32, device="cuda")
        cu_seqlens = torch.cat((zero, cu_seqlens))

        cu_seqlens_kv = batch['seq_len_kv'].cumsum(dim=0).to(torch.int32)
        cu_seqlens_kv = torch.cat((zero, cu_seqlens_kv))

        batch['packed_seq_params'] = {
            'self_attention': PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                qkv_format=module.qkv_format,
            ),
            'cross_attention': PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens_kv,
                qkv_format=module.qkv_format,
            ),
        }

    return batch


def get_batch_on_this_cp_rank(data: Dict):
    """Split the data for context parallelism."""
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    t = 16
    if cp_size > 1:
        # cp split on seq_length, for video_latent, noise_latent and pos_ids
        assert t % cp_size == 0, "t must divisibly by cp_size"
        num_valid_tokens_in_ub = None
        if 'loss_mask' in data and data['loss_mask'] is not None:
            num_valid_tokens_in_ub = data['loss_mask'].sum()

        for key, value in data.items():
            if (value is not None) and (key in ['video', 'video_latent', 'noise_latent', 'pos_ids']):
                if len(value.shape) > 5:
                    value = value.squeeze(0)
                B, C, T, H, W = value.shape
                if T % cp_size == 0:
                    # FIXME packed sequencing
                    data[key] = value.view(B, C, cp_size, T // cp_size, H, W)[:, :, cp_rank, ...].contiguous()
                else:
                    # FIXME packed sequencing
                    data[key] = value.view(B, C, T, cp_size, H // cp_size, W)[:, :, :, cp_rank, ...].contiguous()
        loss_mask = data["loss_mask"]
        data["loss_mask"] = loss_mask.view(loss_mask.shape[0], cp_size, loss_mask.shape[1] // cp_size)[
            :, cp_rank, ...
        ].contiguous()
        data['num_valid_tokens_in_ub'] = num_valid_tokens_in_ub

    return data


@dataclass
class DiTConfig(TransformerConfig, io.IOMixin):
    """
    Config for DiT-S model
    """

    crossattn_emb_size: int = 1024
    add_bias_linear: bool = False
    gated_linear_unit: bool = False

    num_layers: int = 12
    hidden_size: int = 384
    max_img_h: int = 80
    max_img_w: int = 80
    max_frames: int = 34
    patch_spatial: int = 2
    num_attention_heads: int = 6
    layernorm_epsilon = 1e-6
    normalization = "RMSNorm"
    add_bias_linear = False
    qk_layernorm_per_head = True
    layernorm_zero_centered_gamma = False

    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True

    # max_position_embeddings: int = 5400
    hidden_dropout: float = 0
    attention_dropout: float = 0

    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16

    vae_module: str = 'nemo.collections.diffusion.vae.diffusers_vae.AutoencoderKLVAE'
    vae_path: str = None
    sigma_data: float = 0.5

    in_channels: int = 16

    data_step_fn = dit_data_step
    forward_step_fn = dit_forward_step

    replicated_t_embedder = True

    @override
    def configure_model(self, tokenizer=None) -> DiTCrossAttentionModel:
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        if isinstance(self, DiTLlama30BConfig):
            model = DiTLlamaModel
        elif isinstance(self, DiT14BConfig):
            model = DiTCrossAttentionModel14B
            return model(
                self,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
                parallel_output=self.parallel_output,
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
            )
        elif isinstance(self, DiT7BConfig):
            model = DiTCrossAttentionModel7B
            return model(
                self,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
                parallel_output=self.parallel_output,
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
            )
        else:
            model = DiTCrossAttentionModel

        return model(
            self,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            max_img_h=self.max_img_h,
            max_img_w=self.max_img_w,
            max_frames=self.max_frames,
            patch_spatial=self.patch_spatial,
        )

    def configure_vae(self):
        return dynamic_import(self.vae_module)(self.vae_path)


@dataclass
class DiTBConfig(DiTConfig):
    num_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 12


@dataclass
class DiTLConfig(DiTConfig):
    num_layers: int = 24
    hidden_size: int = 1024
    num_attention_heads: int = 16


@dataclass
class DiTXLConfig(DiTConfig):
    num_layers: int = 28
    hidden_size: int = 1152
    num_attention_heads: int = 16


@dataclass
class DiT7BConfig(DiTConfig):
    num_layers: int = 28
    hidden_size: int = 4096
    crossattn_emb_size: int = 1024
    max_img_h: int = 240
    max_img_w: int = 240
    max_frames: int = 128
    patch_spatial: int = 2
    num_attention_heads: int = 32
    layernorm_epsilon = 1e-6
    normalization = "RMSNorm"
    gated_linear_unit: bool = False
    add_bias_linear: bool = False
    qk_layernorm_per_head: bool = True
    apply_rope_fusion: bool = True
    layernorm_zero_centered_gamma: bool = False
    additional_timestamp_channels = None
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True
    hidden_dropout: float = 0
    attention_dropout: float = 0
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    vae_module: str = 'nemo.collections.diffusion.vae.video_vae.video_vae3_512'
    vae_path: str = None
    sigma_data: float = 0.5
    loss_add_logvar: bool = True
    data_step_fn = dit_data_step
    forward_step_fn = dit_forward_step
    model_name: str = 'cosmos_7b_text2world'


@dataclass
class DiT7BVideo2WorldConfig(DiT7BConfig):
    model_name: str = 'cosmos_7b_video2world'

    @override
    def configure_model(self, tokenizer=None) -> DiTCrossAttentionExtendModel7B:
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        model = DiTCrossAttentionExtendModel7B
        return model(
            self,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        )


@dataclass
class DiT7BControl2WorldConfig(DiT7BConfig):
    model_name: str = 'cosmos_7b_control2world'

    # num_layers: int =
    @override
    def configure_model(self, tokenizer=None) -> DiTControl7B:
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        model = DiTControl7B
        return model(
            self,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        )


@dataclass
class DiT7BCameraCtrlConfig(DiT7BConfig):
    model_name: str = 'cosmos_7b_video2world_camera_ctrl'

    @override
    def configure_vae(self):
        return dynamic_import(self.vae_module)(self.vae_path, pixel_chunk_duration=57)

    @override
    def configure_model(self, tokenizer=None) -> DiTCrossAttentionExtendModel7B:
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        dit_model = DiTCrossAttentionExtendModel7B(
            self,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        )
        dit_model.x_embedder = (
            PatchEmbed(
                spatial_patch_size=dit_model.patch_spatial,
                temporal_patch_size=dit_model.patch_temporal,
                in_channels=24,
                out_channels=dit_model.config.hidden_size,
                bias=False,
                keep_spatio=True,
                legacy_patch_emb=dit_model.legacy_patch_emb,
            )
            .cuda()
            .to(dtype=torch.bfloat16)
        )
        dit_model.extra_per_block_abs_pos_emb = True
        return dit_model


@dataclass
class DiT7BActionConfig(DiTConfig):
    num_layers: int = int(os.getenv("NUM_LAYERS", 28))
    hidden_size: int = 4096
    crossattn_emb_size: int = 1024
    max_img_h: int = 240
    max_img_w: int = 240
    max_frames: int = 128
    patch_spatial: int = 2
    num_attention_heads: int = 32
    layernorm_epsilon = 1e-6
    normalization = "RMSNorm"
    gated_linear_unit: bool = False
    add_bias_linear: bool = False
    qk_layernorm_per_head: bool = True
    apply_rope_fusion: bool = True
    layernorm_zero_centered_gamma: bool = False
    additional_timestamp_channels = None
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True
    hidden_dropout: float = 0
    attention_dropout: float = 0
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    vae_module: str = 'nemo.collections.diffusion.vae.video_vae.video_vae3_512'
    vae_path: str = None
    sigma_data: float = 0.5
    loss_add_logvar: bool = True
    data_step_fn = dit_data_step
    forward_step_fn = dit_forward_step
    model_name: str = 'cosmos_7b_video2world_action_control'
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    action_emb_dim: int = 7


@dataclass
class DiT7BVideo2WorldActionConfig(DiT7BActionConfig):
    @override
    def configure_model(self, tokenizer=None) -> DiTCrossAttentionActionExtendModel7B:
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        return DiTCrossAttentionActionExtendModel7B(
            self,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        )


@dataclass
class DiT14BConfig(DiTConfig):
    num_layers: int = 36
    hidden_size: int = 5120
    crossattn_emb_size: int = 1024
    max_img_h: int = 240
    max_img_w: int = 240
    max_frames: int = 128
    patch_spatial: int = 2
    num_attention_heads: int = 40
    layernorm_epsilon = 1e-6
    normalization = "RMSNorm"
    gated_linear_unit: bool = False
    add_bias_linear: bool = False
    qk_layernorm_per_head: bool = True
    apply_rope_fusion: bool = True
    layernorm_zero_centered_gamma: bool = False
    additional_timestamp_channels = None
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True
    hidden_dropout: float = 0
    attention_dropout: float = 0
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    vae_module: str = 'nemo.collections.diffusion.vae.video_vae.video_vae3_512'
    vae_path: str = None
    sigma_data: float = 0.5
    loss_add_logvar: bool = True
    data_step_fn = dit_data_step
    forward_step_fn = dit_forward_step
    model_name = 'cosmos_14b_text2world'


@dataclass
class DiT14BVideo2WorldConfig(DiT14BConfig):
    model_name = 'cosmos_14b_video2world'

    @override
    def configure_model(self, tokenizer=None) -> DiTCrossAttentionExtendModel14B:
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        model = DiTCrossAttentionExtendModel14B
        return model(
            self,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        )


@dataclass
class DiT14BActionConfig(DiTConfig):
    num_layers: int = int(os.getenv("NUM_LAYERS", 36))
    hidden_size: int = 5120
    crossattn_emb_size: int = 1024
    max_img_h: int = 240
    max_img_w: int = 240
    max_frames: int = 128
    patch_spatial: int = 2
    num_attention_heads: int = 40
    layernorm_epsilon = 1e-6
    normalization = "RMSNorm"
    gated_linear_unit: bool = False
    add_bias_linear: bool = False
    qk_layernorm_per_head: bool = True
    apply_rope_fusion: bool = True
    layernorm_zero_centered_gamma: bool = False
    additional_timestamp_channels = None
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True
    hidden_dropout: float = 0
    attention_dropout: float = 0
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    vae_module: str = 'nemo.collections.diffusion.vae.video_vae.video_vae3_512'
    vae_path: str = None
    sigma_data: float = 0.5
    loss_add_logvar: bool = True
    data_step_fn = dit_data_step
    forward_step_fn = dit_forward_step
    model_name: str = 'cosmos_14b_video2world_action_control'
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    action_emb_dim: int = 7


@dataclass
class DiT14BVideo2WorldActionConfig(DiT14BActionConfig):
    @override
    def configure_model(self, tokenizer=None) -> DiTCrossAttentionActionExtendModel14B:
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        return DiTCrossAttentionActionExtendModel14B(
            self,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        )


@dataclass
class DiTLlama30BConfig(DiTConfig):
    num_layers: int = 48
    hidden_size: int = 6144
    ffn_hidden_size: int = 16384
    num_attention_heads: int = 48
    num_query_groups: int = 8
    gated_linear_unit: int = True
    bias_activation_fusion: int = True
    activation_func: Callable = F.silu
    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-5
    max_frames: int = 128
    max_img_h: int = 240
    max_img_w: int = 240
    patch_spatial: int = 2

    init_method_std: float = 0.01
    add_bias_linear: bool = False
    seq_length: int = 256

    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True


@dataclass
class DiTLlama5BConfig(DiTLlama30BConfig):
    num_layers: int = 32
    hidden_size: int = 3072
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 24


class DiTModel(GPTModel):
    def __init__(
        self,
        config: Optional[DiTConfig] = None,
        optim: Optional[OptimizerModule] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
        tokenizer: Optional[Any] = None,
    ):
        super().__init__(config or DiTConfig(), optim=optim, model_transform=model_transform)

        self.vae = None
        self._training_loss_reduction = None
        self._validation_loss_reduction = None

        if hasattr(config, 'model_name'):
            model_name = getattr(config, 'model_name')
            print(f"Initializing the DiT model with configuration model name: [{model_name}]")
            if 'cosmos' in model_name and 'text2world' in model_name:
                self.conditioner = VideoConditioner(
                    text=TextConfig(),
                    fps=FPSConfig(),
                    num_frames=NumFramesConfig(),
                    image_size=ImageSizeConfig(),
                    padding_mask=PaddingMaskConfig(),
                )
                self.diffusion_pipeline = CosmosDiffusionPipeline(
                    net=self,
                    conditioner=self.conditioner,
                    loss_add_logvar=self.config.loss_add_logvar,
                )
            elif 'cosmos' in model_name and 'video2world' in model_name:
                # Parse conditioner configs.
                conditioner_configs = {
                    'text': TextConfig(),
                    'fps': FPSConfig(),
                    'num_frames': NumFramesConfig(),
                    'image_size': ImageSizeConfig(),
                    'padding_mask': PaddingMaskConfig(),
                    'video_cond_bool': VideoCondBoolConfig(
                        condition_location="first_random_n",
                        apply_corruption_to_condition_region="noise_with_sigma",
                    ),
                }
                if 'action_control' in model_name:
                    # Add an action control tensor input.
                    conditioner_configs['action_ctrl'] = ActionControlConfig()
                elif 'camera_ctrl' in model_name:
                    # Overwrite the default 'video_cond_bool' condition
                    conditioner_configs['video_cond_bool'] = VideoCondBoolConfig(
                        condition_location="first_random_n",
                        apply_corruption_to_condition_region="noise_with_sigma",
                        add_pose_condition=True,
                        first_random_n_num_condition_t_max=1,
                    )
                # Instantiate VideoExtendConditioner and ExtendedDiffusionPipeline.
                self.conditioner = VideoExtendConditioner(**conditioner_configs)
                self.diffusion_pipeline = ExtendedDiffusionPipeline(
                    net=self,
                    conditioner=self.conditioner,
                    loss_add_logvar=self.config.loss_add_logvar,
                )
            elif 'cosmos' in model_name and 'control2world' in model_name:
                conditioner_configs = {
                    'text': TextConfig(),
                    'fps': FPSConfig(),
                    'num_frames': NumFramesConfig(),
                    'image_size': ImageSizeConfig(),
                    'padding_mask': PaddingMaskConfig(),
                    'video_cond_bool': VideoCondBoolConfig(
                        condition_location="first_random_n",
                        apply_corruption_to_condition_region="noise_with_sigma",
                    ),
                }
                self.conditioner = VideoExtendConditionerControl(**conditioner_configs)
                self.diffusion_pipeline = CosmosControlDiffusionPipeline(
                    net=self,
                    base_model=None,  # initialize as empty for now
                    conditioner=self.conditioner,
                    loss_add_logvar=self.config.loss_add_logvar,
                )
        else:
            self.diffusion_pipeline = EDMPipeline(net=self, sigma_data=self.config.sigma_data)

    def compute_logvar(self, c_noise):
        logvar = self.module.module.module.logvar
        return logvar(c_noise)

    def data_step(self, dataloader_iter) -> Dict[str, Any]:
        return self.config.data_step_fn(dataloader_iter)

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def forward_step(self, batch) -> torch.Tensor:
        if hasattr(self.config, 'model_name'):
            if 'cosmos' in getattr(self.config, 'model_name'):
                data_batch = {
                    k: v.to(device='cuda', non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()
                }
                data_batch["is_preprocessed"] = True  # assume data is preprocessed
                if parallel_state.is_pipeline_last_stage():
                    output_batch, kendall_loss = self.diffusion_pipeline.training_step(batch, 0)
                    if torch.distributed.get_rank() == 0 and wandb.run:
                        wandb.log(
                            {k: output_batch[k] for k in ['edm_loss'] if k in output_batch}, step=self.global_step
                        )
                    kendall_loss = torch.mean(kendall_loss, dim=1)
                    return kendall_loss
                else:
                    output_tensor = self.diffusion_pipeline.training_step(batch, 0)
                    return output_tensor

        else:
            if parallel_state.is_pipeline_last_stage():
                output_batch, loss = self.diffusion_pipeline.training_step(batch, 0)
                loss = torch.mean(loss, dim=1)
                return loss
            else:
                output_tensor = self.diffusion_pipeline.training_step(batch, 0)
                return output_tensor

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def on_validation_start(self):
        if self.vae is None:
            if self.config.vae_path is None:
                warnings.warn('vae_path not specified skipping validation')
                return None
            self.vae = self.config.configure_vae()
        self.vae.to('cuda')

    def on_validation_end(self):
        if self.vae is not None:
            self.vae.to('cpu')

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        state_shape = batch['video'].shape
        sample = self.diffusion_pipeline.generate_samples_from_batch(
            batch,
            guidance=7,
            state_shape=state_shape,
            num_steps=35,
            is_negative_prompt=True if 'neg_t5_text_embeddings' in batch else False,
        )

        # TODO visualize more than 1 sample
        sample = sample[0, None]
        C, T, H, W = batch['latent_shape'][0]
        seq_len_q = batch['seq_len_q'][0]

        sample = rearrange(
            sample[:, :seq_len_q],
            'B (T H W) (ph pw pt C) -> B C (T pt) (H ph) (W pw)',
            ph=self.config.patch_spatial,
            pw=self.config.patch_spatial,
            C=C,
            T=T,
            H=H // self.config.patch_spatial,
            W=W // self.config.patch_spatial,
        )

        video = (1.0 + self.vae.decode(sample / self.config.sigma_data)).clamp(0, 2) / 2  # [B, 3, T, H, W]

        video = (video * 255).to(torch.uint8).cpu().numpy().astype(np.uint8)

        T = video.shape[2]
        if T == 1:
            image = rearrange(video, 'b c t h w -> (b t h) w c')
            result = image
        else:
            # result = wandb.Video(video, fps=float(batch['fps'])) # (batch, time, channel, height width)
            result = video

        # wandb is on the last rank for megatron, first rank for nemo
        wandb_rank = 0

        if parallel_state.get_data_parallel_src_rank() == wandb_rank:
            if torch.distributed.get_rank() == wandb_rank:
                gather_list = [None for _ in range(parallel_state.get_data_parallel_world_size())]
            else:
                gather_list = None
            torch.distributed.gather_object(
                result, gather_list, wandb_rank, group=parallel_state.get_data_parallel_group()
            )
            if gather_list is not None:
                videos = []
                for video in gather_list:
                    if len(video.shape) == 3:
                        videos.append(wandb.Image(video))
                    else:
                        videos.append(wandb.Video(video, fps=30))
                wandb.log({'prediction': videos}, step=self.global_step)

        return None

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReduction()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = DummyLossReduction()

        return self._validation_loss_reduction

    def on_validation_model_zero_grad(self) -> None:
        '''
        Small hack to avoid first validation on resume.
        This will NOT work if the gradient accumulation step should be performed at this point.
        https://github.com/Lightning-AI/pytorch-lightning/discussions/18110
        '''
        super().on_validation_model_zero_grad()
        if self.trainer.ckpt_path is not None and getattr(self, '_restarting_skip_val_flag', True):
            self.trainer.sanity_checking = True
            self._restarting_skip_val_flag = False


class DummyLossReduction(MegatronLossReduction):
    def __init__(self, validation_step: bool = False, val_drop_last: bool = True) -> None:
        super().__init__()
        self.validation_step = validation_step
        self.val_drop_last = val_drop_last

    def forward(
        self, batch: Dict[str, torch.Tensor], forward_out: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return torch.tensor(0.0, device=torch.cuda.current_device()), {
            "avg": torch.tensor(0.0, device=torch.cuda.current_device())
        }

    def reduce(self, losses_reduced_per_micro_batch) -> torch.Tensor:
        return torch.tensor(0.0, device=torch.cuda.current_device())


def dynamic_import(full_path):
    """
    Dynamically import a class or function from a given full path.

    :param full_path: The full path to the class or function (e.g., "package.module.ClassName")
    :return: The imported class or function
    :raises ImportError: If the module or attribute cannot be imported
    :raises AttributeError: If the attribute does not exist in the module
    """
    try:
        # Split the full path into module path and attribute name
        module_path, attribute_name = full_path.rsplit('.', 1)
    except ValueError as e:
        raise ImportError(
            f"Invalid full path '{full_path}'. It should contain both module and attribute names."
        ) from e

    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_path}'.") from e

    # Retrieve the attribute from the module
    try:
        attribute = getattr(module, attribute_name)
    except AttributeError as e:
        raise AttributeError(f"Module '{module_path}' does not have an attribute '{attribute_name}'.") from e

    return attribute
