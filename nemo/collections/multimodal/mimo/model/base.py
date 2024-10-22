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
from nemo.lightning import io
from megatron.core.transformer.transformer_config import TransformerConfig
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional, Union
import torch
from dataclasses import dataclass, field
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from torch import nn
from megatron.core.transformer.spec_utils import ModuleSpec
from nemo.lightning import get_vocab_size, io
from dataclasses import dataclass
from nemo.collections.llm import fn
from megatron.core.inference_params import InferenceParams
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
from megatron.core.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel
from nemo.collections.llm import Llama2Config7B, Llama2Config13B, LlamaConfig
import torch.nn.functional as F
import pytorch_lightning as L
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from nemo.collections.llm.gpt.model import local_layer_spec, transformer_engine_layer_spec

if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel

    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm.gpt.model.base import get_batch_on_this_context_parallel_rank, get_packed_seq_params


def mimo_forward_step(model, batch) -> torch.Tensor:
    forward_args = {
        "images": batch["images"],
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
        "labels": batch.get("labels", None),
    }
    loss_mask = batch.get("loss_mask", None)
    return model(**forward_args), loss_mask


def mimo_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
    from megatron.core import parallel_state

    batch = next(dataloader_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_keys = set()
    required_keys.add("attention_mask")
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(("images", "tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask"))

    _batch = {
        key: val.cuda(non_blocking=True) if key in required_keys and val is not None else None
        for key, val in _batch.items()
    }
    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch)

    return output


@dataclass
class BaseInputProjectorConfig(TransformerConfig, io.IOMixin):
    projector_type: str = "mlp2x_gelu"
    input_size: Optional[int] = 1024
    hidden_size: int = 4096
    ffn_hidden_size: int = 4096
    activation_func: Callable = F.gelu
    bias: bool = True
    bias_activation_fusion: bool = True
    add_bias_linear: bool = True
    layer_spec: ModuleSpec = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    ).submodules
    num_layers: int = 1  # placeholder, NOT used!
    num_attention_heads: int = 8  # placeholder, NOT used!


@dataclass
class BaseVisionTransformerConfig(TransformerConfig, io.IOMixin):
    num_layers: int = 24
    num_attention_heads: int = 16 # was 32?
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    hidden_size: int = 1024
    hidden_dropout: int = 0.0
    attention_dropout: float = 0.0
    ffn_hidden_size: int = 4096
    gated_linear_unit: bool = False
    # activation_func = quick_gelu
    # kv_channels: int = 64
    # num_query_groups: int = 16
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False  # TODO: Yash Check this
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization = 'LayerNorm'
    layer_spec: ModuleSpec = transformer_engine_layer_spec
    img_h: int = 336
    img_w: int = 336
    patch_dim: int = 14
    vision_model_type = 'clip'


@dataclass
class Llama2Config1B(LlamaConfig):
    num_layers: int = 1
    hidden_size: int = 1024
    num_attention_heads: int = 1
    num_query_groups: int = 1
    ffn_hidden_size: int = 1024


@dataclass
class BaseMimoConfig(TransformerConfig, io.IOMixin):
    language_transformer_config: Optional[TransformerConfig] = field(default_factory=lambda: Llama2Config7B())
    vision_transformer_config: Optional[TransformerConfig] = field(
        default_factory=lambda: BaseVisionTransformerConfig()
    )
    vision_projection_config: Optional[TransformerConfig] = field(default_factory=lambda: BaseInputProjectorConfig())

    freeze_language_model: bool = True
    freeze_vision_model: bool = True
    freeze_vision_projection: bool = False

    forward_step_fn: Callable = mimo_forward_step
    data_step_fn: Callable = mimo_data_step

    vocab_size: int = 1
    num_layers: int = 1  # placeholder, NOT used!
    num_attention_heads: int = 8  # placeholder, NOT used!

    def configure_model(self, tokenizer) -> "MCoreLLaVAModel":

        model = MCoreLLaVAModel(
            language_transformer_config=self.language_transformer_config,
            language_transformer_layer_spec=transformer_engine_layer_spec(self.language_transformer_config),
            language_vocab_size=self.vocab_size,
            language_max_sequence_length=self.language_transformer_config.seq_length,
            vision_transformer_config=self.vision_transformer_config,
            vision_transformer_layer_spec=transformer_engine_layer_spec(self.vision_transformer_config),
            drop_vision_class_token=True,
            vision_projection_config=self.vision_projection_config,
            vision_projection_layer_spec=self.vision_projection_config.layer_spec,
            vision_projection_type="mlp",
            allow_missing_vision_projection_checkpoint=True,
            parallel_output=True,
            pre_process=True,
            post_process=True,
            add_encoder=True,
            add_decoder=True,
            img_h=self.vision_transformer_config.img_h,
            img_w=self.vision_transformer_config.img_w,
            patch_dim=self.vision_transformer_config.patch_dim,
        )
        return model


class BaseMimoModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    def __init__(
        self,
        config: BaseMimoConfig,
        # TODO: Add transformer_layer_spec when we update mcore
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True))
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self.model_transform = model_transform
        self._training_loss_reduction = None
        self._validation_loss_reduction = None

    def configure_model(self) -> "MCoreLLaVAModel":
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: InferenceParams = None,
    ) -> torch.Tensor:

        output_tensor = self.module(
            images=images,
            input_ids=input_ids,
            position_ids=position_ids,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            labels=labels,
            inference_params=inference_params,
        )

        return output_tensor

    #   TODO: Yash: May be we can inherit GPTModel and not have someo of the common function implementations here.
    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

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
    
    
from nemo.lightning import OptimizerModule, io, teardown
from nemo.collections.multimodal.mimo.model.base import BaseMimoConfig, BaseMimoModel
from transformers import LlavaForConditionalGeneration
from pathlib import Path
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

@io.model_importer(BaseMimoModel, "hf")
class HFLlavaMimoImporter(io.ModelConnector["LlavaForConditionalGeneration", BaseMimoModel]):
    def init(self) -> BaseMimoModel:
        return BaseMimoModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        
        source = LlavaForConditionalGeneration.from_pretrained(str(self))
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        
        print(f"Converted Llava model to Nemo, saving to {output_path}")
        self.nemo_save(output_path, trainer)
        
        print(f"Converted Llava model saved to {output_path}")
        
        teardown(trainer, target)
        del trainer, target

        return output_path
    
    def convert_state(self, source, target):
        mapping = {}
        # vision module
        print(f"****** Yash Debug == target  *******")
        print(target)
        mapping.update(
            {
                "vision_tower.vision_model.embeddings.patch_embedding.weight": "vision_model.conv1.weight",
                "vision_tower.vision_model.embeddings.position_embedding.weight": "vision_model.position_embeddings.weight",
            }
        )
        # Update with pre-layer normalization
        mapping.update(
            {
                "vision_tower.vision_model.pre_layrnorm.weight": "vision_model.ln_pre.weight",
                "vision_tower.vision_model.pre_layrnorm.bias": "vision_model.ln_pre.bias",
            }
        )
        # Update with layer normalization layers
        mapping.update(
            {
                "vision_tower.vision_model.encoder.layers.*.layer_norm1.weight": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                "vision_tower.vision_model.encoder.layers.*.layer_norm1.bias": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
                "vision_tower.vision_model.encoder.layers.*.layer_norm2.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                "vision_tower.vision_model.encoder.layers.*.layer_norm2.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            }
        )

        # Update with MLP layers (Feedforward block)
        mapping.update(
            {
                "vision_tower.vision_model.encoder.layers.*.mlp.fc1.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.weight",
                "vision_tower.vision_model.encoder.layers.*.mlp.fc1.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.bias",
                "vision_tower.vision_model.encoder.layers.*.mlp.fc2.weight": "vision_model.decoder.layers.*.mlp.linear_fc2.weight",
                "vision_tower.vision_model.encoder.layers.*.mlp.fc2.bias": "vision_model.decoder.layers.*.mlp.linear_fc2.bias",
            }
        )

        # Update with self-attention linear projection
        mapping.update(
            {
                "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj.weight": "vision_model.decoder.layers.*.self_attention.linear_proj.weight",
                "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj.bias": "vision_model.decoder.layers.*.self_attention.linear_proj.bias",
            }
        )

        # projection module

        mapping.update(
            {
                "multi_modal_projector.linear_1.weight": "vision_projection.encoder.linear_fc1.weight",
                "multi_modal_projector.linear_1.bias": "vision_projection.encoder.linear_fc1.bias",
                "multi_modal_projector.linear_2.weight": "vision_projection.encoder.linear_fc2.weight",
                "multi_modal_projector.linear_2.bias": "vision_projection.encoder.linear_fc2.bias",
            }
        )

        # Language model

        mapping.update(
            {
                # "language_model.model.embed_tokens.weight": "language_model.embedding.word_embeddings.weight",
                "language_model.model.layers.*.self_attn.o_proj.weight": "language_model.decoder.layers.*.self_attention.linear_proj.weight",
                "language_model.model.layers.*.mlp.down_proj.weight": "language_model.decoder.layers.*.mlp.linear_fc2.weight",
                "language_model.model.layers.*.input_layernorm.weight": "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                "language_model.model.layers.*.post_attention_layernorm.weight": "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                "language_model.model.norm.weight": "language_model.decoder.final_layernorm.weight",
                # "language_model.lm_head.weight": "language_model.output_layer.weight",
            }
        )
        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            # transforms=[_import_class_token, _import_vison_qkv, _import_linear_fc1, _import_language_qkv],
            transforms=[_import_class_token, _import_linear_fc1, _import_language_qkv, _import_embed_tokens,_import_lm_head_weight, _import_vison_qkv, _transform_vision_qkv_bias],
            
        )
        
    
    @property
    def tokenizer(self) -> "AutoTokenizer":
        from transformers import AutoTokenizer
        temp = str(self)
        return AutoTokenizer.from_pretrained(str(self))
    
    @property
    def config(self) -> BaseMimoConfig:
        return BaseMimoConfig(vocab_size=self.tokenizer.vocab_size)
    
    

@io.state_transform(
    source_key=(
        "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.weight",
        "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj.weight",
        "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj.weight",
    ),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_vison_qkv(ctx: io.TransformCTX, q, k, v):
    megatron_config = ctx.target.vision_model.config
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
        "language_model.model.layers.*.mlp.gate_proj.weight",
        "language_model.model.layers.*.mlp.up_proj.weight",
    ),
    target_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
)
def _import_linear_fc1(down, gate):
    return torch.cat((down, gate), axis=0)


@io.state_transform(
    source_key="vision_tower.vision_model.embeddings.class_embedding",
    target_key="vision_model.class_token",
)
def _import_class_token(ctx: io.TransformCTX, class_embedding):
    # Source shape: (1024,)
    # Target shape: (1, 1, 1024)

    # Reshape the class embedding to match the target shape
    class_token = class_embedding.view(1, 1, -1)

    # Ensure the transformation is correct
    assert class_token.shape == (1, 1, 1024), f"Expected shape (1, 1, 1024), but got {class_token.shape}"

    return class_token


@io.state_transform(
    source_key=(
        "language_model.model.layers.*.self_attn.q_proj.weight",
        "language_model.model.layers.*.self_attn.k_proj.weight",
        "language_model.model.layers.*.self_attn.v_proj.weight",
    ),
    target_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_language_qkv(ctx: io.TransformCTX, q, k, v):
    megatron_config = ctx.target.language_model.config
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
    source_key="language_model.model.embed_tokens.weight",
    target_key="language_model.embedding.word_embeddings.weight",
)
def _import_embed_tokens(ctx: io.TransformCTX, source_embed):
    # Extract the size of the target embeddings from the target model
    target_shape = ctx.target.state_dict()["language_model.embedding.word_embeddings.weight"].shape

    # Extract the number of embeddings in the target (e.g., 32000)
    target_vocab_size = target_shape[0]
    embedding_dim = target_shape[1]

    # Ensure the source has enough embeddings to copy from
    assert source_embed.shape[1] == embedding_dim, (
        f"Embedding dimension mismatch: source={source_embed.shape[1]}, target={embedding_dim}"
    )

    # Truncate or copy only the first 'target_vocab_size' embeddings from the source
    truncated_embed = source_embed[:target_vocab_size, :]

    # Ensure the shape matches the target shape
    assert truncated_embed.shape == target_shape, (
        f"Expected shape {target_shape}, but got {truncated_embed.shape}"
    )

    return truncated_embed


@io.state_transform(
    source_key="language_model.lm_head.weight",
    target_key="language_model.output_layer.weight",
)
def _import_lm_head_weight(ctx: io.TransformCTX, source_weight):
    # Extract the shape of the target weight
    target_shape = ctx.target.state_dict()["language_model.output_layer.weight"].shape
    target_vocab_size, target_embedding_dim = target_shape
    source_vocab_size, source_embedding_dim = source_weight.shape
    assert target_embedding_dim == source_embedding_dim, (
        f"Embedding dimension mismatch: "
        f"source={source_embedding_dim}, target={target_embedding_dim}"
    )
    truncated_weight = source_weight[:target_vocab_size, :]

    assert truncated_weight.shape == target_shape, (
        f"Expected shape {target_shape}, but got {truncated_weight.shape}"
    )

    return truncated_weight

@io.state_transform(
    source_key=(
        "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj.bias",
        "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj.bias",
        "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj.bias",
    ),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
)
def _transform_vision_qkv_bias(ctx: io.TransformCTX, q_bias, k_bias, v_bias):
    """
    Transforms and concatenates Q, K, V biases from the source model to the target model.
    """

    # Concatenate the Q, K, V biases into a single bias tensor
    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)  # (3072,)

    # Ensure the concatenated bias has the correct shape
    expected_shape = (3072,)
    assert qkv_bias.shape == expected_shape, f"Expected shape {expected_shape}, but got {qkv_bias.shape}"

    return qkv_bias