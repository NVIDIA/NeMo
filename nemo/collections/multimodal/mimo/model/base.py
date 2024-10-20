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
    hidden_size: int = 1024
    ffn_hidden_size: int = 1024
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
    num_attention_heads: int = 32
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    hidden_size: int = 1024
    hidden_dropout: int = 0.0
    attention_dropout: float = 0.0
    ffn_hidden_size: int = 4096
    gated_linear_unit: bool = False
    # activation_func = quick_gelu
    kv_channels: int = 64
    num_query_groups: int = 16
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
    language_transformer_config: Optional[TransformerConfig] = field(default_factory=lambda: Llama2Config1B())
    vision_transformer_config: Optional[TransformerConfig] = field(
        default_factory=lambda: BaseVisionTransformerConfig()
    )
    vision_projection_config: Optional[TransformerConfig] = field(default_factory=lambda: BaseInputProjectorConfig())

    freeze_language_model: bool = True
    freeze_vision_model: bool = True
    freeze_vision_projection: bool = False

    forward_step_fn: Callable = mimo_forward_step
    data_step_fn: Callable = mimo_data_step

    vocab_size: int = 32000
    num_layers: int = 1  # placeholder, NOT used!
    num_attention_heads: int = 8  # placeholder, NOT used!

    def configure_model(self, tokenizer) -> "MCoreLLaVAModel":

        model = MCoreLLaVAModel(
            language_transformer_config=self.language_transformer_config,
            language_transformer_layer_spec=transformer_engine_layer_spec(self.language_transformer_config),
            language_vocab_size=self.vocab_size,
            language_max_sequence_length=2048,
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
