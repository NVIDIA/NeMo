# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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

from nemo.collections.llm.gpt.model.megatron.hyena.hyena_layer_specs import hyena_stack_spec, hyena_stack_spec_no_te
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_model import HyenaModel as MCoreHyenaModel
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils import hyena_no_weight_decay_cond
from nemo.utils import logging

try:
    from megatron.core import parallel_state
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig

    HAVE_MEGATRON_CORE_OR_TE = True

except (ImportError, ModuleNotFoundError):
    logging.warning(
        "The package `megatron.core` was not imported in this environment which is needed for Hyena models."
    )

    HAVE_MEGATRON_CORE_OR_TE = False
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig

from nemo.collections.llm.gpt.model.base import GPTModel, gpt_data_step
from nemo.lightning import get_vocab_size, io, teardown


class HyenaModel(GPTModel):
    """
    This is a wrapper around the MCoreHyenaModel to allow for inference. Our model follows the same API as the GPTModel,
        but the megatron model class is different so we need to handle the inference wrapper slightly differently.
    """

    def get_inference_wrapper(self, params_dtype, inference_batch_times_seqlen_threshold) -> torch.Tensor:
        # This is to get the MCore model required in GPTInferenceWrapper.
        mcore_model = self.module
        while mcore_model:
            if type(mcore_model) is MCoreHyenaModel:
                break
            mcore_model = getattr(mcore_model, "module", None)
        if mcore_model is None or type(mcore_model) is not MCoreHyenaModel:
            raise ValueError("Exact MCoreHyenaModel instance not found in the model structure.")

        vocab_size = None
        if self.tokenizer is not None:
            vocab_size = self.tokenizer.vocab_size
        elif hasattr(self.config, 'vocab_size'):
            vocab_size = self.config.vocab_size
        else:
            raise ValueError(
                'Unable to find vocab size.'
                ' Either pass in a tokenizer with vocab size, or set vocab size in the model config'
            )

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=mcore_model.config.hidden_size,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
            padded_vocab_size=vocab_size,
        )

        model_inference_wrapper = GPTInferenceWrapper(mcore_model, inference_wrapper_config)
        return model_inference_wrapper

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        inference_params=None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        extra_kwargs = {'packed_seq_params': packed_seq_params} if packed_seq_params is not None else {}
        output_tensor = self.module(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_params=inference_params,
            loss_mask=loss_mask,
            **extra_kwargs,
        )
        return output_tensor


def hyena_forward_step(model, batch) -> torch.Tensor:

    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "labels": batch["labels"],
        "loss_mask": batch["loss_mask"],
    }
    forward_args["attention_mask"] = None
    return model(**forward_args)


@dataclass
class HyenaConfig(TransformerConfig, io.IOMixin):
    """
    Configuration dataclass for Hyena.

    For adjusting ROPE when doing context extension, set seq_len_interpolation_factor relative to 8192.
    For example, if your context length is 512k, then set the factor to 512k / 8k = 64.
    """

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
    fp32_residual_connection: bool = True
    normalization: str = 'RMSNorm'
    add_bias_linear: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-6
    attention_backend: AttnBackend = AttnBackend.flash
    # TODO: Move this to better places?
    get_attention_mask_from_fusion: bool = False
    recompute_granularity: str = 'full'
    recompute_method: str = 'uniform'
    recompute_num_layers: int = 4
    forward_step_fn: Callable = hyena_forward_step
    data_step_fn: Callable = gpt_data_step
    tokenizer_model_path: str = None
    hyena_init_method: str = None
    hyena_output_layer_init_method: str = None
    hyena_filter_no_wd: bool = True
    remove_activation_post_first_layer: bool = True
    add_attn_proj_bias: bool = True
    cross_entropy_loss_fusion: bool = False  # Faster but lets default to False for more precision
    tp_comm_overlap: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    add_bias_output: bool = False
    use_te: bool = True
    to_upper: str = "normalized_weighted"  # choose between "weighted" and "normalized_weighted"

    def __post_init__(self):
        super().__post_init__()
        self.hyena_no_weight_decay_cond_fn = hyena_no_weight_decay_cond if self.hyena_filter_no_wd else None

    def configure_model(self, tokenizer) -> "MCoreHyenaModel":

        self.bias_activation_fusion = False if self.remove_activation_post_first_layer else self.bias_activation_fusion

        model = MCoreHyenaModel(
            self,
            hyena_stack_spec=hyena_stack_spec if self.use_te else hyena_stack_spec_no_te,
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
            remove_activation_post_first_layer=self.remove_activation_post_first_layer,
            add_attn_proj_bias=self.add_attn_proj_bias,
        )
        return model


@io.model_importer(HyenaModel, "pytorch")
class PyTorchHyenaImporter(io.ModelConnector["HyenaModel", HyenaModel]):

    def __new__(cls, path: str, model_config=None):
        instance = super().__new__(cls, path)
        instance.model_config = model_config
        return instance

    def init(self) -> HyenaModel:

        return HyenaModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path, checkpoint_format: str = 'torch_dist', weights_only: bool = True) -> Path:

        source = torch.load(str(self), map_location='cpu', weights_only=weights_only)
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

            def adjust_medium_filter(self, updated_data):
                from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig

                for k, v in updated_data.items():
                    if "filter.h" in k or "filter.decay" in k:
                        updated_data[k] = v[:, : HyenaConfig().hyena_medium_conv_len]
                return updated_data

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
                updated_data = self.adjust_medium_filter(updated_data)
                return updated_data

        source = ModelState(source, self.config.num_layers)
        target = self.init()
        trainer = self.nemo_setup(target, ckpt_async_save=False, save_ckpt_format=checkpoint_format)
        source.to(self.config.params_dtype)
        target.to(self.config.params_dtype)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        logging.info(f"Converted Hyena model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):

        mapping = {}
        mapping['sequential.0.word_embeddings.weight'] = 'embedding.word_embeddings.weight'
        mapping[f'sequential.{len(self.config.hybrid_override_pattern)}.norm.weight'] = 'decoder.final_norm.weight'
        te_enabled = self.config.use_te
        for i, symbol in enumerate(self.config.hybrid_override_pattern):
            if te_enabled:
                mapping[f'sequential.{i}.pre_mlp_layernorm.weight'] = (
                    f'decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight'
                )
            else:
                mapping[f'sequential.{i}.pre_mlp_layernorm.weight'] = f'decoder.layers.{i}.pre_mlp_layernorm.weight'
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
                    mapping[f'sequential.{i}.input_layernorm.weight'] = f'decoder.layers.{i}.input_layernorm.weight'

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

@io.model_importer(HyenaModel, "pytorch-vortex")
class PytorchVortexHyenaImporter(io.ModelConnector["HyenaModel", HyenaModel]):

    def __new__(cls, path: str, model_config=None):
        instance = super().__new__(cls, path)
        instance.model_config = model_config
        return instance
    
    def init(self) -> HyenaModel:

        return HyenaModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path, checkpoint_format: str = 'torch_dist', weights_only: bool = False) -> Path:

        source = torch.load(str(self), map_location='cpu', weights_only=weights_only)
        if 'model' in source:
            source = source['model']
        config = self.config
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

            def adjust_medium_filter(self, updated_data):
                # from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig

                # for k, v in updated_data.items():
                #     if "filter.h" in k or "filter.decay" in k:
                #         updated_data[k] = v[:, : HyenaConfig().hyena_medium_conv_len]
                return updated_data
            def squeeze_filters(self, updated_data):
                for i, symbol in enumerate(config.hybrid_override_pattern):
                    if symbol == 'D':
                        # Weak filter decay
                        log_r_min = -2
                        log_r_max = 2
                        d_model = updated_data[f'blocks.{i}.filter.h'].shape[0]
                        t = torch.linspace(0, 1, 128)[None]
                        decay = torch.logspace(log_r_min, log_r_max, d_model)[:, None]
                        decay = torch.exp((-decay * t).cuda())
                        updated_data[f'blocks.{i}.filter.h'] = updated_data[f'blocks.{i}.filter.h'].squeeze(1) / decay
                for k, v in updated_data.items():
                    if "filter.short_filter_weight" in k:
                        updated_data[k] = v.squeeze(1)
                return updated_data

            def transform_source_dict(self, source):
                import re

                layer_map = {i + 2: i for i in range(self.num_layers)}
                layer_map[self.num_layers + 3] = self.num_layers
                updated_data = {}
                if "module" in source:
                    source_module = source['module']
                else:
                    source_module = source
                for key in list(source_module.keys()):
                    if "_extra" in key:
                        source_module.pop(key)
                    else:
                        # match = re.search(r'blocks\.(\d+)', key)
                        # if match:
                        #     original_layer_num = int(match.group(1))
                        #     if original_layer_num in layer_map:
                        #         # Create the updated key by replacing the layer number
                        #         new_key = re.sub(rf'\b{original_layer_num}\b', str(layer_map[original_layer_num]), key)
                        #         updated_data[new_key] = source_module[key]
                        #     else:
                        #         # Keep the key unchanged if no mapping exists
                        #         updated_data[key] = source_module[key]
                        # else:
                        updated_data[key] = source_module[key]
                updated_data = self.adjust_medium_filter(updated_data)
                updated_data = self.squeeze_filters(updated_data)
                return updated_data

        source = ModelState(source, self.config.num_layers)
        target = self.init()
        trainer = self.nemo_setup(target, ckpt_async_save=False, save_ckpt_format=checkpoint_format)
        source.to(self.config.params_dtype)
        target.to(self.config.params_dtype)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        logging.info(f"Converted Hyena model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {}
        # Word embeddings and output layer
        mapping['embedding_layer.weight'] = 'embedding.word_embeddings.weight'
        #mapping['unembed.weight'] = 'output_layer.weight'
        
        # Final norm
        mapping['norm.scale'] = 'decoder.final_norm.weight'
        #mapping['norm.bias'] = 'decoder.final_norm.bias'
        
        # Layer mappings
        for i, symbol in enumerate(self.config.hybrid_override_pattern):
            
            # MLP layers and norms for all blocks
            mapping[f'blocks.{i}.post_norm.scale'] = f'decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight'
            #mapping[f'blocks.{i}.post_norm.bias'] = f'decoder.layers.{i}.mlp.linear_fc1.layer_norm_bias'
            # mapping[f'blocks.{i}.mlp.l1.weight'] = f'decoder.layers.{i}.mlp.linear_fc1.weight'  # Partial
            # mapping[f'blocks.{i}.mlp.l2.weight'] = f'decoder.layers.{i}.mlp.linear_fc1.weight'  # Partial
            mapping[f'blocks.{i}.mlp.l3.weight'] = f'decoder.layers.{i}.mlp.linear_fc2.weight'

            if symbol != '*':
                # Common mixer mappings for S/D/H blocks
                mapping[f'blocks.{i}.filter.short_filter_weight'] = f'decoder.layers.{i}.mixer.hyena_proj_conv.short_conv_weight'
                mapping[f'blocks.{i}.out_filter_dense.weight'] = f'decoder.layers.{i}.mixer.dense.weight'
                mapping[f'blocks.{i}.out_filter_dense.bias'] = f'decoder.layers.{i}.mixer.dense.bias'
                # Pre-norm and projections for all blocks
                mapping[f'blocks.{i}.pre_norm.scale'] = f'decoder.layers.{i}.mixer.dense_projection.layer_norm_weight'
                #mapping[f'blocks.{i}.pre_norm.bias'] = f'decoder.layers.{i}.mixer.dense_projection.layer_norm_bias'
                mapping[f'blocks.{i}.projections.weight'] = f'decoder.layers.{i}.mixer.dense_projection.weight'
                if symbol == 'S':
                    mapping[f'blocks.{i}.filter.h'] = (  # This one should not be squeezed.
                        f'decoder.layers.{i}.mixer.mixer.short_conv.short_conv_weight'
                    )

                elif symbol == 'D':
                    mapping[f'blocks.{i}.filter.h'] = f'decoder.layers.{i}.mixer.mixer.filter.h'  # this one should be squeezed
                    mapping[f'blocks.{i}.filter.D'] = f'decoder.layers.{i}.mixer.mixer.conv_bias'

                elif symbol == 'H':
                    #mapping[f'blocks.{i}.filter.log_poles'] = f'decoder.layers.{i}.mixer.mixer.filter.gamma'
                    mapping[f'blocks.{i}.filter.D'] = f'decoder.layers.{i}.mixer.mixer.conv_bias'
                    mapping[f'blocks.{i}.filter.residues'] = f'decoder.layers.{i}.mixer.mixer.filter.R'
                    #mapping[f'blocks.{i}.filter.p'] = f'decoder.layers.{i}.mixer.mixer.filter.p'

            elif symbol == '*':
                # Attention block mappings
                mapping[f'blocks.{i}.inner_mha_cls.Wqkv.weight'] = f'decoder.layers.{i}.self_attention.linear_qkv.weight'
                mapping[f'blocks.{i}.inner_mha_cls.out_proj.weight'] = f'decoder.layers.{i}.self_attention.linear_proj.weight'
                mapping[f'blocks.{i}.inner_mha_cls.out_proj.bias'] = f'decoder.layers.{i}.self_attention.linear_proj.bias'
            else:
                raise ValueError(f'Unknown symbol: {symbol}')

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_import_vortex_linear_fc1, _log_poles_and_p_to_gamma])

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
    source_key=("blocks.*.mlp.l1.weight", "blocks.*.mlp.l2.weight"),
    target_key="decoder.layers.*.mlp.linear_fc1.weight",
)
def _import_vortex_linear_fc1(w1, w2):
    return torch.cat((w1, w2), axis=0)

@io.state_transform(
    source_key="blocks.*.filter.log_poles",
    target_key=("decoder.layers.*.mixer.mixer.filter.gamma", "decoder.layers.*.mixer.mixer.filter.p"),
)
def _log_poles_and_p_to_gamma(log_poles):
    # There are an infinite number of possible p,gamma values that will produce the same log_poles
    # Here we just choose the simplest one, p = -1, gamma = log(-log_poles) + 1
    p = -torch.ones_like(log_poles[..., 0])  # shape [d_model, order]
    gamma = torch.log(-log_poles[..., 0]) + 1  # shape [d_model, order]
    return gamma, p

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
    recompute_granularity: str = 'full'
    recompute_method: str = 'uniform'
    recompute_num_layers: int = 2
    hyena_init_method: str = 'small_init'
    hyena_output_layer_init_method: str = 'wang_init'
    hyena_filter_no_wd: bool = True


@dataclass
class HyenaNVTestConfig(HyenaTestConfig):
    """Several unintentional design choices were made to the original Arc implementation that are required to use the
    original Arc checkpoints, but may result in less stable model training. If you are training from scratch,
    these are the recommended configs.
    """

    remove_activation_post_first_layer: bool = False
    add_attn_proj_bias: bool = False


@dataclass
class Hyena7bConfig(HyenaConfig):
    """Config matching the 7b 8k context Evo2 model"""

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
    recompute_granularity: str = 'full'
    recompute_method: str = 'uniform'
    recompute_num_layers: int = 4
    hyena_init_method: str = 'small_init'
    hyena_output_layer_init_method: str = 'wang_init'
    hyena_filter_no_wd: bool = True


@dataclass
class HyenaNV7bConfig(Hyena7bConfig):
    """Several unintentional design choices were made to the original Arc implementation that are required to use the
    original Arc checkpoints, but may result in less stable model training. If you are training from scratch,
    these are the recommended configs.
    """

    remove_activation_post_first_layer: bool = False
    add_attn_proj_bias: bool = False


@dataclass
class Hyena40bConfig(HyenaConfig):
    """Config matching the 40b 8k context Evo2 model"""

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
    recompute_granularity: str = 'full'
    recompute_method: str = 'uniform'
    recompute_num_layers: int = 2
    hyena_init_method: str = 'small_init'
    hyena_output_layer_init_method: str = 'wang_init'
    hyena_filter_no_wd: bool = True


@dataclass
class HyenaNV40bConfig(Hyena40bConfig):
    """Several unintentional design choices were made to the original Arc implementation that are required to use the
    original Arc checkpoints, but may result in less stable model training. If you are training from scratch,
    these are the recommended configs.
    """

    remove_activation_post_first_layer: bool = False
    add_attn_proj_bias: bool = False


@dataclass
class Hyena7bARCLongContextConfig(Hyena7bConfig):
    """The checkpoint from ARC requires padding to the FFN dim
    due to constraintes from large TP size for training."""

    ffn_hidden_size: int = 11264


@dataclass
class Hyena40bARCLongContextConfig(Hyena40bConfig):
    """The checkpoint from ARC requires padding to the FFN dim
    due to constraintes from large TP size for training."""

    ffn_hidden_size: int = 22528


__all__ = [
    "HyenaConfig",
    "Hyena7bConfig",
    "HyenaNV7bConfig",
    "Hyena40bConfig",
    "HyenaNV40bConfig",
    "Hyena7bARCLongContextConfig",
    "Hyena40bARCLongContextConfig",
    "HyenaTestConfig",
    "HyenaNVTestConfig",
]
