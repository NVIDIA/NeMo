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
    from megatron.core.models.hyena.hyena_layer_specs import hyena_stack_spec

    HAVE_MEGATRON_CORE_OR_TE = True

except (ImportError, ModuleNotFoundError):
    logging.warning("The package `megatron.core` was not imported in this environment which is needed for SSMs.")
    HAVE_MEGATRON_CORE_OR_TE = False

from megatron.core.transformer.transformer_config import TransformerConfig
from nemo.collections.llm.gpt.model.base import GPTModel, gpt_data_step
from nemo.lightning import get_vocab_size, io, teardown


########## Temporary experimental code ##########
import yaml

class DotDict(dict):
    """A dictionary that supports dot notation for accessing keys."""
    def __getattr__(self, attr):
        return self.get(attr)
    
    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        del self[item]

def load_yaml_as_dotdict(filepath):
    with open(filepath, 'r') as file:
        yaml_content = yaml.safe_load(file)
    
    # Recursively convert dictionary to DotDict and replace "-" with "_"
    return dict_to_dotdict(yaml_content)

def dict_to_dotdict(d):
    """Convert a dictionary into a DotDict recursively and replace '-' with '_' in keys."""
    if not isinstance(d, dict):
        return d
    
    transformed_dict = {}
    for k, v in d.items():
        # Replace "-" with "_" in the key
        new_key = k.replace('-', '_')
        transformed_dict[new_key] = dict_to_dotdict(v)
    
    return DotDict(transformed_dict)

GLOBAL_CONFIG = load_yaml_as_dotdict('/home/ataghibakhsh/savanna/dummy_config.yml')
#################################################

def hyena_forward_step(model, batch) -> torch.Tensor:

    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "labels": batch["labels"],
    }
    forward_args["attention_mask"] = None
    return model(**forward_args)


@dataclass
class HyenaConfig(TransformerConfig, io.IOMixin):
    # From megatron.core.models.mamba.mamba_model.MambaModel
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    fp16: bool = False
    bf16: bool = True
    num_layers: int = 2
    mamba_ssm_ngroups: int = 8
    num_attention_heads: int = 8
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    hybrid_override_pattern: str = None
    post_process: bool = True
    pre_process: bool = True
    seq_length: int = 2048
    # Mamba with no attention has no need for position embeddings, so none is default
    position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'none'
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

    forward_step_fn: Callable = hyena_forward_step
    data_step_fn: Callable = gpt_data_step
    tokenizer_model_path: str = None

    def configure_model(self, tokenizer) -> "MCoreHyenaModel":
        self.hyena = GLOBAL_CONFIG
        model =  MCoreHyenaModel(
            self,
            hyena_stack_spec=hyena_stack_spec,
            vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
            max_sequence_length=self.seq_length,
            hybrid_override_pattern=self.hybrid_override_pattern,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        )
        # if torch.distributed.get_rank() == 0:
        #     import pdb; pdb.set_trace()
        # torch.distributed.barrier()
        return model    


@dataclass
class HyenaTestConfig(HyenaConfig):
    hybrid_override_pattern: str = "SDH*"
    num_layers: int = 4
    seq_length: int = 8192
    hidden_size: int = 4096
    mamba_ssm_ngroups: int = 1
    make_vocab_size_divisible_by: int = 8
    tokenizer_library: str = 'byte-level'
    mapping_type: str = "base"
    ffn_hidden_size: int = 11008
    gated_linear_unit:bool = True
    num_attention_heads: int = 32
    use_cpu_initialization: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    params_dtype: torch.dtype = torch.bfloat16
    normalization: str = "RMSNorm"
    add_qkv_bias:bool = False
    add_bias_linear:bool = False
    layernorm_epsilon: float = 1e-6
    fp8: str = 'hybrid'
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"

@dataclass
class Hyena7bConfig(HyenaConfig):
    hybrid_override_pattern: str = "SDH*SHDSDH*SDHSDH*SDHSDH*SDHSDH*"
    num_layers: int = 32
    seq_length: int = 8192
    hidden_size: int = 4096
    mamba_ssm_ngroups: int = 1
    make_vocab_size_divisible_by: int = 8
    tokenizer_library: str = 'byte-level'
    mapping_type: str = "base"
    ffn_hidden_size: int = 11008
    gated_linear_unit:bool = True
    num_attention_heads: int = 32
    use_cpu_initialization: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    params_dtype: torch.dtype = torch.bfloat16
    normalization: str = "RMSNorm"
    add_qkv_bias:bool = False
    add_bias_linear:bool = False
    layernorm_epsilon: float = 1e-6
    fp8: str = 'hybrid'
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"

__all__ = [
    "HyenaConfig",
    "HyenaTestConfig",
]
