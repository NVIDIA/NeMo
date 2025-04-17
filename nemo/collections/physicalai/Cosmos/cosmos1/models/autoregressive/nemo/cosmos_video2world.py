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

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Dict, Optional, Union

import torch
from cosmos1.models.autoregressive.nemo.cosmos import (
    CosmosConfig,
    CosmosConfig4B,
    CosmosConfig12B,
    CosmosModel,
    RotaryEmbedding3D,
)
from cosmos1.models.autoregressive.nemo.inference.inference_controller import CosmosInferenceWrapper
from cosmos1.utils import log
from megatron.core import tensor_parallel
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
from megatron.core.transformer.attention import (
    Attention,
    CrossAttentionSubmodules,
    SelfAttention,
    SelfAttentionSubmodules,
)
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor
from torch import nn

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_block import TransformerBlock

from nemo.collections.llm.gpt.model.base import get_batch_on_this_context_parallel_rank
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io
from nemo.lightning.base import teardown


class CosmosTransformerBlock(TransformerBlock):
    def forward(
        self,
        *args,
        packed_seq_params: PackedSeqParams = None,
        extra_positional_embeddings=None,
        **kwargs,
    ):
        packed_seq_params = {"abs_pos_embed": extra_positional_embeddings}
        return super().forward(
            *args,
            packed_seq_params=packed_seq_params,
            **kwargs,
        )


@dataclass
class CosmosCrossAttentionSubmodules(CrossAttentionSubmodules):
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class CrossAttention(Attention):
    """Cross-attention layer class

    Cross-attention layer takes input with size [s, b, h] and context with size
    [s, b, h] and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CosmosCrossAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
    ):
        config = copy.deepcopy(config)
        # config.num_query_groups =  config.num_attention_heads

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="cross",
        )

        self.linear_q = build_module(
            submodules.linear_q,
            self.config.hidden_size,
            self.query_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

        self.linear_kv = build_module(
            submodules.linear_kv,
            self.config.crossattn_emb_size,
            2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=self.hidden_size_per_attention_head,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.k_layernorm = build_module(
            submodules.k_layernorm,
            hidden_size=self.hidden_size_per_attention_head,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensors
        from `key_value_states`.
        """
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv, _ = self.linear_kv(key_value_states)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv.size()[:-1] + (
            self.num_query_groups_per_partition,
            2 * self.hidden_size_per_attention_head,
        )
        mixed_kv = mixed_kv.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key, value) = tensor_parallel.split_tensor_along_last_dim(mixed_kv, 2)

        # Attention head [sq, b, h] --> [sq, b, hp]
        query, _ = self.linear_q(hidden_states)

        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = query.size()[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        query = query.view(*new_tensor_shape)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        return query, key, value


class CosmosVideo2WorldTransformerLayer(TransformerLayer):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__(
            config=config, submodules=submodules, layer_number=layer_number, hidden_dropout=hidden_dropout
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
        **kwargs,
    ):
        assert "abs_pos_embed" in packed_seq_params
        abs_pos_embed = packed_seq_params["abs_pos_embed"]
        assert (
            abs_pos_embed.shape[0] == hidden_states.shape[0]
        ), f"Abs pos embed shape : {abs_pos_embed.shape}, hidden states shape : {hidden_states.shape}. They should match at the zeroth dimension"
        packed_seq_params = None

        hidden_states = abs_pos_embed + hidden_states

        rotary_pos_emb
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=None,
        )
        # print(f'After attn : {attention_output_with_bias[0].sum().item()}')

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)
        # print(f'After ffn : {mlp_output_with_bias[0].sum().item()}')

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        # print(f'Final out : {hidden_states.sum().item()}')
        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)
        # CUDA graph requires returned values to be Tensors
        if self.config.external_cuda_graph and self.training:
            return output
        return output, context


def get_cosmos_video2world_spec() -> ModuleSpec:
    return ModuleSpec(
        module=CosmosVideo2WorldTransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_cross_attn_layernorm=TENorm,
            cross_attention=ModuleSpec(
                module=CrossAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=CosmosCrossAttentionSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
            cross_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


def cosmos_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
    from megatron.core import parallel_state

    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842

    batch = next(dataloader_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_device_keys = set()
    required_host_keys = set()

    required_device_keys.add("attention_mask")
    if "cu_seqlens" in _batch:
        raise ValueError("Packed sequence cu_seqlens not supported")

    required_device_keys.update(("context", "abs_pos_embed", "action"))
    if parallel_state.is_pipeline_first_stage():
        required_device_keys.update(("tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_device_keys.update(("labels", "loss_mask"))

    _batch_required_keys = {}
    for key, val in _batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True)
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu()
        else:
            _batch_required_keys[key] = None

    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch_required_keys)

    return output


def cosmos_forward_step(model, batch) -> torch.Tensor:
    forward_args = {}
    forward_args["input_ids"] = batch["tokens"]
    forward_args["position_ids"] = batch["position_ids"]
    forward_args["attention_mask"] = batch["attention_mask"]
    forward_args["labels"] = batch["labels"]
    forward_args["extra_block_kwargs"] = {
        "context": batch["context"],
        "extra_positional_embeddings": batch["abs_pos_embed"],
    }
    forward_args["packed_seq_params"] = None

    return model(**forward_args)


@dataclass
class CosmosVideo2WorldConfig:
    vocab_size: int = 64064
    output_layer_vocab_size: int = 64000
    seq_length: int = 12864
    latent_shape = [5, 40, 64]
    pad_to_multiple_of = 64
    forward_step_fn: Callable = cosmos_forward_step
    transformer_layer_spec = get_cosmos_video2world_spec()
    data_step_fn: Callable = cosmos_data_step
    attention_backend: AttnBackend = AttnBackend.flash
    crossattn_emb_size: int = 1024
    kv_channels: int = 128
    training_type: str | None = "text_to_video"

    def configure_model(self, tokenizer) -> "MCoreGPTModel":
        self.transformer_layer_spec = get_cosmos_video2world_spec()
        model = super().configure_model(tokenizer)
        if self.rope_dim == "3D":
            model.rotary_pos_emb = RotaryEmbedding3D(
                seq_len=self.seq_length,
                training_type=self.training_type,
                pad_to_multiple_of=self.pad_to_multiple_of,
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
                latent_shape=self.latent_shape,
                original_latent_shape=self.original_latent_shape if hasattr(self, "original_latent_shape") else None,
            )
        model.output_layer = tensor_parallel.ColumnParallelLinear(
            self.hidden_size,
            self.output_layer_vocab_size,
            config=self,
            init_method=self.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=False,
            skip_weight_param_allocation=False,
            embedding_activation_buffer=None,
            grad_output_buffer=None,
        )

        model.decoder = CosmosTransformerBlock(
            config=self,
            spec=self.transformer_layer_spec,
            pre_process=model.pre_process,
            post_process=model.post_process,
        )
        return model


@dataclass
class CosmosConfigVideo2World5B(CosmosVideo2WorldConfig, CosmosConfig4B):
    make_vocab_size_divisible_by: int = 64


@dataclass
class CosmosConfigVideo2World13B(CosmosVideo2WorldConfig, CosmosConfig12B):
    make_vocab_size_divisible_by: int = 128


class CosmosVideo2WorldModel(CosmosModel):
    def __init__(
        self,
        config: Annotated[Optional[CosmosConfig], Config[CosmosConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(
            config or CosmosConfigVideo2World5B(), optim=optim, tokenizer=tokenizer, model_transform=model_transform
        )
        self.config = config

    def get_inference_wrapper(self, params_dtype, inference_batch_times_seqlen_threshold) -> torch.Tensor:
        # This is to get the MCore model required in GPTInferenceWrapper.
        mcore_model = self.module

        vocab_size = self.config.vocab_size

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=mcore_model.config.hidden_size,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
            padded_vocab_size=vocab_size,
        )

        model_inference_wrapper = CosmosInferenceWrapper(mcore_model, inference_wrapper_config, self.config)
        return model_inference_wrapper

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        inference_params=None,
        packed_seq_params=None,
        extra_block_kwargs=None,
    ) -> torch.Tensor:
        extra_kwargs = {"packed_seq_params": packed_seq_params} if packed_seq_params is not None else {}
        output_tensor = self.module(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_params=inference_params,
            extra_block_kwargs=extra_block_kwargs,
            **extra_kwargs,
        )

        return output_tensor


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
def _import_qkv_cosmos_self_attention(ctx: io.TransformCTX, q, k, v):
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


@io.state_transform(
    source_key=(
        "model.layers.*.cross_attention.wk.weight",
        "model.layers.*.cross_attention.wv.weight",
    ),
    target_key="decoder.layers.*.cross_attention.linear_kv.weight",
)
def _import_kv_cosmos_cross_attention(ctx: io.TransformCTX, k, v):
    megatron_config = ctx.target.config

    num_query_groups = megatron_config.num_query_groups
    hidden_size = megatron_config.crossattn_emb_size
    head_size = megatron_config.kv_channels
    new_kv_tensor_shape = (num_query_groups, head_size) + (hidden_size,)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    kv_weights_l = []
    for i in range(num_query_groups):
        kv_weights_l.append(k[i : i + 1, :, :])
        kv_weights_l.append(v[i : i + 1, :, :])
    kv_weights = torch.cat(kv_weights_l)

    kv_weights = kv_weights.reshape([2 * hidden_size, hidden_size])

    return kv_weights


@io.model_importer(CosmosVideo2WorldModel, "pt")
class PTCosmosVideo2WorldImporter(io.ModelConnector["PTCosmosVideo2WorldModel", CosmosVideo2WorldModel]):
    def init(self) -> CosmosVideo2WorldModel:
        return CosmosVideo2WorldModel(self.config, tokenizer=self.tokenizer)

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
            "model.layers.*.cross_attention.wo.weight": "decoder.layers.*.cross_attention.linear_proj.weight",
            "model.layers.*.cross_attention.wq.weight": "decoder.layers.*.cross_attention.linear_q.weight",
            "model.layers.*.cross_attention.q_norm.weight": "decoder.layers.*.cross_attention.q_layernorm.weight",
            "model.layers.*.cross_attention.k_norm.weight": "decoder.layers.*.cross_attention.k_layernorm.weight",
            "model.layers.*.cross_attention.weight": "decoder.layers.*.cross_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.cross_attention_norm.weight": "decoder.layers.*.pre_cross_attn_layernorm.weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "model.output.weight": "output_layer.weight",
        }

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[_import_qkv_cosmos_self_attention, _mlp_glu, _import_kv_cosmos_cross_attention],
        )

    @property
    def tokenizer(self):
        return None

    @property
    def config(self):
        if "5B" in str(self) or "5b" in str(self):
            return CosmosConfigVideo2World5B()
        elif "13B" in str(self) or "13b" in str(self):
            return CosmosConfigVideo2World13B()
        else:
            raise ValueError("Unable to infer model size from checkpoint")
