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

# pylint: disable=C0115,C0116,C0301

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional

import torch
import torch.nn.functional as F
from cosmos1.utils import log
from einops import rearrange, repeat
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
from megatron.core.transformer.enums import AttnBackend
from torch import Tensor, nn

from nemo.collections.llm.gpt.model.llama import Llama3Config, LlamaModel
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io
from nemo.lightning.base import teardown


class RotaryEmbedding3D(RotaryEmbedding):
    """Rotary Embedding3D for Cosmos Language model.
    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained
            from transformer config
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to
            10000.
        use_cpu_initialization (bool, optional): If False, initialize the inv_freq directly
            on the GPU. Defaults to False
        latent_shape: The shape of the latents produced by the video after being tokenized
    """

    def __init__(
        self,
        seq_len: int,
        kv_channels: int,
        training_type: str | None = None,
        rotary_base: int = 10000,
        use_cpu_initialization: bool = False,
        latent_shape=[5, 40, 64],
        apply_yarn=False,
        original_latent_shape=None,
        beta_fast=32,
        beta_slow=1,
        scale=None,
        max_position_embeddings=None,
        original_max_position_embeddings=None,
        extrapolation_factor=1,
        attn_factor=1,
        pad_to_multiple_of=None,
    ) -> None:
        super().__init__(
            kv_channels=kv_channels,
            rotary_base=rotary_base,
            rotary_percent=1.0,
            use_cpu_initialization=use_cpu_initialization,
        )
        self.training_type = training_type
        self.latent_shape = latent_shape
        self.device = "cpu" if use_cpu_initialization else torch.cuda.current_device()
        self.dim = kv_channels
        self.rope_theta = rotary_base
        self.apply_yarn = apply_yarn
        self.original_latent_shape = original_latent_shape
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.scale = scale
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.attn_factor = attn_factor
        dim_h = self.dim // 6 * 2
        dim_t = self.dim - 2 * dim_h
        self.dim_spatial_range = torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(self.device) / dim_h
        spatial_inv_freq = 1.0 / (self.rope_theta**self.dim_spatial_range)
        self.dim_temporal_range = torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(self.device) / dim_t
        temporal_inv_freq = 1.0 / (self.rope_theta**self.dim_temporal_range)
        if self.apply_yarn:
            assert self.original_latent_shape is not None, "Original latent shape required."
            assert self.beta_slow is not None, "Beta slow value required."
            assert self.beta_fast is not None, "Beta fast value required."
            scale_factors_spatial = self.get_scale_factors(spatial_inv_freq, self.original_latent_shape[1])
            spatial_inv_freq = spatial_inv_freq * scale_factors_spatial
            scale_factors_temporal = self.get_scale_factors(temporal_inv_freq, self.original_latent_shape[0])
            temporal_inv_freq = temporal_inv_freq * scale_factors_temporal
            self.mscale = float(self.get_mscale(self.scale) * self.attn_factor)
        self.spatial_inv_freq = spatial_inv_freq
        self.temporal_inv_freq = temporal_inv_freq
        max_seq_len_cached = max(self.latent_shape)
        if self.apply_yarn and seq_len > max_seq_len_cached:
            max_seq_len_cached = seq_len
        self.max_seq_len_cached = max_seq_len_cached
        self.pad_to_multiple_of = pad_to_multiple_of
        self.freqs = self.get_freqs_non_repeated(self.max_seq_len_cached)

    def get_mscale(self, scale: float = 1.0) -> float:
        """Get the magnitude scaling factor for YaRN."""
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def get_scale_factors(self, inv_freq: torch.Tensor, original_seq_len: int) -> torch.Tensor:
        """Get the scale factors for YaRN."""
        # Calculate the high and low frequency cutoffs for YaRN. Note: `beta_fast` and `beta_slow` are called
        # `high_freq_factor` and `low_freq_factor` in the Llama 3.1 RoPE scaling code.
        high_freq_cutoff = 2 * math.pi * self.beta_fast / original_seq_len
        low_freq_cutoff = 2 * math.pi * self.beta_slow / original_seq_len
        # Obtain a smooth mask that has a value of 0 for low frequencies and 1 for high frequencies, with linear
        # interpolation in between.
        smooth_mask = torch.clamp((inv_freq - low_freq_cutoff) / (high_freq_cutoff - low_freq_cutoff), min=0, max=1)
        # For low frequencies, we scale the frequency by 1/self.scale. For high frequencies, we keep the frequency.
        scale_factors = (1 - smooth_mask) / self.scale + smooth_mask
        return scale_factors

    def get_freqs_non_repeated(self, max_seq_len_cached: int, offset: int = 0) -> Tensor:
        dtype = self.spatial_inv_freq.dtype
        device = self.spatial_inv_freq.device

        self.seq = (torch.arange(max_seq_len_cached, device=device, dtype=dtype) + offset).cuda()

        assert hasattr(
            self, "latent_shape"
        ), "Latent shape is not set. Please run set_latent_shape() method on rope embedding. "
        T, H, W = self.latent_shape
        half_emb_t = torch.outer(self.seq[:T], self.temporal_inv_freq.cuda())
        half_emb_h = torch.outer(self.seq[:H], self.spatial_inv_freq.cuda())
        half_emb_w = torch.outer(self.seq[:W], self.spatial_inv_freq.cuda())
        emb = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )
        emb = rearrange(emb, "t h w d -> (t h w) 1 1 d").float()

        if self.training_type == "text_to_video":
            bov_pe = torch.zeros((1, *emb.shape[1:]), device=emb.device)
            emb = torch.cat((bov_pe, emb), dim=0)

        if self.pad_to_multiple_of is not None and emb.shape[0] % self.pad_to_multiple_of != 0:
            # Round up to the nearest multiple of pad_to_multiple_of
            pad_len = self.pad_to_multiple_of - emb.shape[0] % self.pad_to_multiple_of
            emb = torch.cat((emb, torch.zeros((pad_len, *emb.shape[1:]), device=emb.device)), dim=0)

        return emb

    @lru_cache(maxsize=32)
    def forward(self, seq_len: int, offset: int = 0, packed_seq: bool = False) -> Tensor:
        if self.spatial_inv_freq.device.type == "cpu":
            # move `inv_freq` to GPU once at the first micro-batch forward pass
            self.spatial_inv_freq = self.spatial_inv_freq.to(device=torch.cuda.current_device())

        max_seq_len_cached = self.max_seq_len_cached
        if self.apply_yarn and seq_len > max_seq_len_cached:
            max_seq_len_cached = seq_len
        self.max_seq_len_cached = max_seq_len_cached
        emb = self.get_freqs_non_repeated(self.max_seq_len_cached)
        return emb


if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class CosmosConfig(Llama3Config):
    qk_layernorm: bool = True
    rope_dim: str = "3D"
    vocab_size: int = 64000
    activation_func = F.silu
    attention_backend: AttnBackend = AttnBackend.flash

    def configure_model(self, tokenizer) -> "MCoreGPTModel":
        model = super().configure_model(tokenizer)
        if self.rope_dim == "3D":
            model.rotary_pos_emb = RotaryEmbedding3D(
                seq_len=self.seq_length,
                training_type=None,
                kv_channels=self.kv_channels,
                max_position_embeddings=self.seq_length,
                original_max_position_embeddings=self.original_seq_len if hasattr(self, "original_seq_len") else None,
                rotary_base=self.rotary_base,
                apply_yarn=True if hasattr(self, "apply_yarn") else False,
                scale=self.yarn_scale if hasattr(self, "yarn_scale") else None,
                extrapolation_factor=1,
                attn_factor=1,
                beta_fast=self.yarn_beta_fast if hasattr(self, "yarn_beta_fast") else 32,
                beta_slow=self.yarn_beta_slow if hasattr(self, "yarn_beta_slow") else 1,
                latent_shape=[5, 40, 64],
                original_latent_shape=self.original_latent_shape if hasattr(self, "original_latent_shape") else None,
            )
        return model


@dataclass
class CosmosConfig4B(CosmosConfig):
    rotary_base: int = 500_000
    seq_length: int = 15360
    num_layers: int = 16
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32
    num_query_groups: int = 8
    layernorm_epsilon: float = 1e-5
    use_cpu_initialization: bool = True
    make_vocab_size_divisible_by: int = 128
    kv_channels: int = 128


@dataclass
class CosmosConfig12B(CosmosConfig):
    rotary_base: int = 500_000
    seq_length: int = 15360
    num_layers: int = 40
    hidden_size: int = 5120
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32
    num_query_groups: int = 8
    layernorm_epsilon: float = 1e-5
    use_cpu_initialization: bool = True
    make_vocab_size_divisible_by: int = 128
    kv_channels: int = 128
    original_latent_shape = [3, 40, 64]
    apply_yarn: bool = True
    yarn_beta_fast: int = 4
    yarn_beta_slow: int = 1
    yarn_scale: int = 2
    original_seq_len = 8192


class CosmosModel(LlamaModel):
    def __init__(
        self,
        config: Annotated[Optional[CosmosConfig], Config[CosmosConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or CosmosConfig4B(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)
        self.config = config


@io.state_transform(
    source_key=(
        "model.layers.*.feed_forward.w1.weight",
        "model.layers.*.feed_forward.w3.weight",
    ),
    target_key="decoder.layers.*.mlp.linear_fc1.weight",
)
def _mlp_glu(ctx: io.TransformCTX, w1, w3):
    return torch.cat((w1, w3), axis=0)


@io.state_transform(
    source_key=(
        "model.layers.*.attention.wq.weight",
        "model.layers.*.attention.wk.weight",
        "model.layers.*.attention.wv.weight",
    ),
    target_key="decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv_cosmos(ctx: io.TransformCTX, q, k, v):
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
    new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    qkv_weights_l = []
    for i in range(num_query_groups):
        qkv_weights_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
        qkv_weights_l.append(k[i : i + 1, :, :])
        qkv_weights_l.append(v[i : i + 1, :, :])
    qkv_weights = torch.cat(qkv_weights_l)
    assert qkv_weights.ndim == 3, qkv_weights.shape
    assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
    assert qkv_weights.shape[1] == head_size, qkv_weights.shape
    assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

    qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

    return qkv_weights


@io.model_importer(CosmosModel, "pt")
class PTCosmosImporter(io.ModelConnector["PTCosmosModel", CosmosModel]):
    def init(self) -> CosmosModel:
        return CosmosModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        pt_model_path = str(self)
        cosmos_model_state_dict = torch.load(pt_model_path, map_location="cpu")
        for k, v in cosmos_model_state_dict.items():
            # convert to float 32 (for cpu conversion) (Original model is bf16)
            cosmos_model_state_dict[k] = v.float()

        # Small wrapper since nemo calls source.state_dict() , to get state dict
        class WrapperCosmos:
            def __init__(self, model_state_dict):
                self.model_state_dict = model_state_dict

            def state_dict(self):
                return self.model_state_dict

        source = WrapperCosmos(cosmos_model_state_dict)
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        log.info(f"Converted PT Cosmos model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "model.tok_embeddings.weight": "embedding.word_embeddings.weight",
            "model.layers.*.attention.wo.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.attention.q_norm.weight": "decoder.layers.*.self_attention.q_layernorm.weight",
            "model.layers.*.attention.k_norm.weight": "decoder.layers.*.self_attention.k_layernorm.weight",
            "model.layers.*.attention_norm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.feed_forward.w2.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.ffn_norm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "model.output.weight": "output_layer.weight",
        }

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_import_qkv_cosmos, _mlp_glu])

    @property
    def tokenizer(self):
        return None

    @property
    def config(self):
        if "4B" in str(self) or "4b" in str(self):
            return CosmosConfig4B()
        elif "12B" in str(self) or "12b" in str(self):
            return CosmosConfig12B()
        else:
            raise ValueError("Unable to infer model size from checkpoint")
