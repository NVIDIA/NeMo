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
from pathlib import Path
from typing import Callable, Literal, Optional

import torch

from nemo.utils import logging

try:
    from megatron.core import parallel_state
    from megatron.core.models.hyena import HyenaModel as MCoreHyenaModel
    from megatron.core.models.hyena.hyena_layer_specs import hyena_stack_spec, hyena_stack_spec_no_te
    from megatron.core.ssm.hyena_utils import hyena_no_weight_decay_cond

    HAVE_MEGATRON_CORE_OR_TE = True

except (ImportError, ModuleNotFoundError):
    logging.warning(
        "The package `megatron.core` was not imported in this environment which is needed for Hyena models."
    )

    HAVE_MEGATRON_CORE_OR_TE = False

from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.collections.llm.gpt.model.base import GPTModel, gpt_data_step
from nemo.lightning import get_vocab_size, io, teardown


def hyena_forward_step(model, batch) -> torch.Tensor:
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "labels": batch["labels"],
        "loss_mask": batch["loss_mask"],
        "attention_mask": None
    }
    return model(**forward_args)


@dataclass
class HyenaConfig(TransformerConfig, io.IOMixin):
    # From megatron.core.models.hyena.hyena_model.HyenaModel
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    fp16: bool = False
    bf16: bool = True
    num_layers: int = 2
    num_attention_heads: int = 8
    num_groups_hyena: int = None
    num_groups_hyena_medium: int = None
    num_groups_hyena_short: int = None
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    hybrid_override_pattern: str = None
    post_process: bool = True
    pre_process: bool = True
    seq_length: int = 2048
    position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'rope'
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    seq_len_interpolation_factor: Optional[float] = None
    apply_rope_fusion: bool = True
    make_vocab_size_divisible_by: int = 128
    gated_linear_unit: bool = False
    fp32_residual_connections: bool = True
    normalization: str = 'RMSNorm'
    add_bias_linear: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-6
    # TODO: Move this to better places?
    get_attention_mask_from_fusion: bool = False
    recompute_granularity: str = 'full'
    recompute_method: str = 'uniform'
    recompute_num_layers: int = 4
    fp8: str = 'hybrid'
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"
    forward_step_fn: Callable = hyena_forward_step
    data_step_fn: Callable = gpt_data_step
    tokenizer_model_path: str = None
    hyena_init_method: str = None
    hyena_output_layer_init_method: str = None
    hyena_filter_no_wd: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.hyena_no_weight_decay_cond_fn = hyena_no_weight_decay_cond if self.hyena_filter_no_wd else None

    def configure_model(self, tokenizer) -> "MCoreHyenaModel":
        model = MCoreHyenaModel(
            self,
            hyena_stack_spec=hyena_stack_spec,
            vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
            max_sequence_length=self.seq_length,
            num_groups_hyena=self.num_groups_hyena,
            num_groups_hyena_medium=self.num_groups_hyena_medium,
            num_groups_hyena_short=self.num_groups_hyena_short,
            hybrid_override_pattern=self.hybrid_override_pattern,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            share_embeddings_and_output_weights=True,
            hyena_init_method=self.hyena_init_method,
            hyena_output_layer_init_method=self.hyena_output_layer_init_method,
        )
        return model

@io.model_importer(GPTModel, "pytorch")
class PyTorchHyenaImporter(io.ModelConnector["GPTModel", GPTModel]):

    def __new__(cls, path: str, model_config=None):
        instance = super().__new__(cls, path)
        instance.model_config = model_config
        return instance

    def init(self) -> GPTModel:

        return GPTModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path, te_enabled=True) -> Path:

        source = torch.load(str(self), map_location='cpu')
        if 'model' in source:
            source = source['model']

        class ModelState:
            def __init__(self, state_dict, num_layers):
                self.num_layers = num_layers
                state_dict = self.transform_source_dict(state_dict)
                self._state_dict = state_dict

            def state_dict(self):
                return self._state_dict

            def to(self, dtype):
                for k, v in self._state_dict.items():
                    if "_extra" not in k:
                        if v.dtype != dtype:
                            logging.warning(f"Converting {k} from {v.dtype} (source model) to {dtype} (target model)")
                        self._state_dict[k] = v.to(dtype)

            def transform_source_dict(self, source):
                import re

                layer_map = {i + 2: i for i in range(self.num_layers)}
                layer_map[self.num_layers + 3] = self.num_layers
                updated_data = {}

                for key in list(source['module'].keys()):
                    if "_extra" in key:
                        source['module'].pop(key)
                    else:
                        match = re.search(r'sequential\.(\d+)', key)
                        if match:
                            original_layer_num = int(match.group(1))
                            if original_layer_num in layer_map:
                                # Create the updated key by replacing the layer number
                                new_key = re.sub(rf'\b{original_layer_num}\b', str(layer_map[original_layer_num]), key)
                                updated_data[new_key] = source['module'][key]
                            else:
                                # Keep the key unchanged if no mapping exists
                                updated_data[key] = source['module'][key]
                        else:
                            updated_data[key] = source['module'][key]
                return updated_data

        source = ModelState(source, self.config.num_layers)
        target = self.init()
        trainer = self.nemo_setup(target, ckpt_async_save=False, save_ckpt_format='zarr')
        source.to(self.config.params_dtype)
        target.to(self.config.params_dtype)
        self.convert_state(source, target, te_enabled)
        self.nemo_save(output_path, trainer)

        logging.info(f"Converted Hyena model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target, te_enabled=True):

        mapping = {}
        mapping['sequential.0.word_embeddings.weight'] = 'embedding.word_embeddings.weight'
        mapping[f'sequential.{len(self.config.hybrid_override_pattern)}.norm.weight'] = (
            'decoder.final_norm.weight'
        )
        for i, symbol in enumerate(self.config.hybrid_override_pattern):
            if te_enabled:
                mapping[f'sequential.{i}.pre_mlp_layernorm.weight'] = (
                    f'decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight'
                )
            else:
                mapping[f'sequential.{i}.pre_mlp_layernorm.weight'] = (
                    f'decoder.layers.{i}.pre_mlp_layernorm.weight'
                )
            mapping[f'sequential.{i}.mlp.w3.weight'] = f'decoder.layers.{i}.mlp.linear_fc2.weight'

            if symbol != '*':
                if te_enabled:
                    mapping[f'sequential.{i}.input_layernorm.weight'] = (
                        f'decoder.layers.{i}.mixer.dense_projection.layer_norm_weight'
                    )
                else:
                    mapping[f'sequential.{i}.input_layernorm.weight'] = f'decoder.layers.{i}.norm.weight'

                mapping[f'sequential.{i}.mixer.dense_projection.weight'] = (
                    f'decoder.layers.{i}.mixer.dense_projection.weight'
                )
                mapping[f'sequential.{i}.mixer.hyena_proj_conv.short_conv_weight'] = (
                    f'decoder.layers.{i}.mixer.hyena_proj_conv.short_conv_weight'
                )
                mapping[f'sequential.{i}.mixer.dense.weight'] = f'decoder.layers.{i}.mixer.dense.weight'
                mapping[f'sequential.{i}.mixer.dense.bias'] = f'decoder.layers.{i}.mixer.dense.bias'

                if symbol == 'S':
                    mapping[f'sequential.{i}.mixer.mixer.short_conv.short_conv_weight'] = (
                        f'decoder.layers.{i}.mixer.mixer.short_conv.short_conv_weight'
                    )

                elif symbol == 'D':
                    mapping[f'sequential.{i}.mixer.mixer.conv_bias'] = f'decoder.layers.{i}.mixer.mixer.conv_bias'
                    mapping[f'sequential.{i}.mixer.mixer.filter.h'] = f'decoder.layers.{i}.mixer.mixer.filter.h'
                    mapping[f'sequential.{i}.mixer.mixer.filter.decay'] = (
                        f'decoder.layers.{i}.mixer.mixer.filter.decay'
                    )

                elif symbol == 'H':
                    mapping[f'sequential.{i}.mixer.mixer.conv_bias'] = f'decoder.layers.{i}.mixer.mixer.conv_bias'
                    mapping[f'sequential.{i}.mixer.mixer.filter.gamma'] = (
                        f'decoder.layers.{i}.mixer.mixer.filter.gamma'
                    )
                    mapping[f'sequential.{i}.mixer.mixer.filter.R'] = f'decoder.layers.{i}.mixer.mixer.filter.R'
                    mapping[f'sequential.{i}.mixer.mixer.filter.p'] = f'decoder.layers.{i}.mixer.mixer.filter.p'

            elif symbol == '*':
                if te_enabled:
                    mapping[f'sequential.{i}.input_layernorm.weight'] = (
                        f'decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight'
                    )
                else:
                    mapping[f'sequential.{i}.input_layernorm.weight'] = (
                        f'decoder.layers.{i}.input_layernorm.weight'
                    )

                mapping[f'sequential.{i}.mixer.dense_projection.weight'] = (
                    f'decoder.layers.{i}.self_attention.linear_qkv.weight'
                )
                mapping[f'sequential.{i}.mixer.dense.weight'] = f'decoder.layers.{i}.self_attention.linear_proj.weight'
                mapping[f'sequential.{i}.mixer.dense.bias'] = f'decoder.layers.{i}.self_attention.linear_proj.bias'
            else:
                raise ValueError(f'Unknown symbol: {symbol}')

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_import_linear_fc1])

    @property
    def tokenizer(self):
        from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

        tokenizer = get_nmt_tokenizer(
            library=self.model_config.tokenizer_library,
        )

        return tokenizer

    @property
    def config(self) -> HyenaConfig:
        return self.model_config


@io.state_transform(
    source_key=("sequential.*.mlp.w1.weight", "sequential.*.mlp.w2.weight"),
    target_key="decoder.layers.*.mlp.linear_fc1.weight",
)
def _import_linear_fc1(w1, w2):
    return torch.cat((w1, w2), axis=0)


@dataclass
class HyenaTestConfig(HyenaConfig):
    hybrid_override_pattern: str = "SDH*"
    num_layers: int = 4
    seq_length: int = 8192
    hidden_size: int = 4096
    num_groups_hyena: int = 4096
    num_groups_hyena_medium: int = 256
    num_groups_hyena_short: int = 256
    make_vocab_size_divisible_by: int = 8
    tokenizer_library: str = 'byte-level'
    mapping_type: str = "base"
    ffn_hidden_size: int = 11008
    gated_linear_unit: bool = True
    num_attention_heads: int = 32
    use_cpu_initialization: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    params_dtype: torch.dtype = torch.bfloat16
    normalization: str = "RMSNorm"
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-6
    # fp8: str = 'hybrid'
    # fp8_amax_history_len: int = 16
    # fp8_amax_compute_algo: str = "max"
    recompute_granularity: str = 'full'
    recompute_method: str = 'uniform'
    recompute_num_layers: int = 2
    hyena_init_method: str = 'small_init'
    hyena_output_layer_init_method: str = 'wang_init'
    hyena_filter_no_wd: bool = True


@dataclass
class Hyena7bConfig(HyenaConfig):
    hybrid_override_pattern: str = "SDH*SDHSDH*SDHSDH*SDHSDH*SDHSDH*"
    num_layers: int = 32
    seq_length: int = 8192
    hidden_size: int = 4096
    num_groups_hyena: int = 4096
    num_groups_hyena_medium: int = 256
    num_groups_hyena_short: int = 256
    make_vocab_size_divisible_by: int = 8
    tokenizer_library: str = 'byte-level'
    mapping_type: str = "base"
    ffn_hidden_size: int = 11008
    gated_linear_unit: bool = True
    num_attention_heads: int = 32
    use_cpu_initialization: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    params_dtype: torch.dtype = torch.bfloat16
    normalization: str = "RMSNorm"
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-6
    fp8: str = 'hybrid'
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"
    recompute_granularity: str = 'full'
    recompute_method: str = 'uniform'
    recompute_num_layers: int = 4
    hyena_init_method: str = 'small_init'
    hyena_output_layer_init_method: str = 'wang_init'
    hyena_filter_no_wd: bool = True

@dataclass
class Hyena40bConfig(HyenaConfig):
    hybrid_override_pattern: str = "SDH*SDHSDH*SDHSDH*SDHSDH*SDHSDH*SDH*SDHSDH*SDHSDH*"
    num_layers: int = 50
    seq_length: int = 8192
    hidden_size: int = 8192
    num_groups_hyena: int = 8192
    num_groups_hyena_medium: int = 512
    num_groups_hyena_short: int = 512
    make_vocab_size_divisible_by: int = 8
    tokenizer_library: str = 'byte-level'
    mapping_type: str = "base"
    ffn_hidden_size: int = 21888
    gated_linear_unit: bool = True
    num_attention_heads: int = 64
    use_cpu_initialization: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    params_dtype: torch.dtype = torch.bfloat16
    normalization: str = "RMSNorm"
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-6
    fp8: str = 'hybrid'
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"
    recompute_granularity: str = 'full'
    recompute_method: str = 'uniform'
    recompute_num_layers: int = 2
    hyena_init_method: str = 'small_init'
    hyena_output_layer_init_method: str = 'wang_init'
    hyena_filter_no_wd: bool = True

__all__ = [
    "HyenaConfig",
    "Hyena7bConfig",
    "Hyena40bConfig",
    "HyenaTestConfig",
]
