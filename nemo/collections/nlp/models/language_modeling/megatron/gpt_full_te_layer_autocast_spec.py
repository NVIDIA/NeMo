# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from importlib.metadata import version
from typing import Any, Callable, Optional

import packaging
import torch

from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults
from nemo.collections.nlp.parts import utils_funcs
from nemo.utils.import_utils import safe_import_from

TransformerLayer, HAVE_TE = safe_import_from("transformer_engine.pytorch", "TransformerLayer")
if not HAVE_TE:

    TransformerLayer = ApexGuardDefaults

try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
    from megatron.core.transformer.cuda_graphs import CudaGraphManager
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import BaseTransformerLayer, get_transformer_layer_offset
    from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError) as e:

    MegatronModule = ModuleSpec = ApexGuardDefaults
    BaseTransformerLayer = object  # try to avoid inconsistent-mro for TETransformerLayerAutocast

    HAVE_MEGATRON_CORE = False
    IMPORT_ERROR = e


# Copied from nemo/collections/nlp/modules/common/megatron/transformer.py
# as the source file is slated to be removed
class AutocastTransformerLayer(TransformerLayer):
    """
    Wrapper of te.pytorch.TransformerLayer: a single transformerlayer
    that takes input with size [s, b, h] and returns an output of
    the same size.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        layernorm_epsilon: float,
        num_attention_heads: int,
        init_method: Callable,
        output_layer_init_method: Callable,
        hidden_dropout: float,
        attention_dropout: float,
        layer_number: Optional[int] = None,
        kv_channels: Optional[int] = None,
        self_attn_mask_type: str = "causal",
        tp_group: Optional[Any] = None,
        tp_size: int = 1,
        params_dtype: torch.dtype = torch.float32,
        get_rng_state_tracker: Optional[Callable] = None,
        fuse_wgrad_accumulation: bool = False,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        sequence_parallel: bool = False,
        apply_residual_connection_post_layernorm: bool = False,
        output_layernorm: bool = False,
        layer_type: str = "encoder",
        drop_path_rate: float = 0,
        use_emha: bool = False,
        ub_tp_comm_overlap: bool = False,
        ub_bulk_wgrad: bool = True,
        ub_bulk_dgrad: bool = True,
        autocast_dtype: Any = 16,
        zero_centered_gamma: bool = False,
        device: str = 'cuda',
        **kwargs,
    ) -> None:
        assert (
            HAVE_MEGATRON_CORE and HAVE_TE
        ), "AutocastTransformerLayer requires Megatron Core and Transformer Engine to be installed."

        transformer_layer_args = {
            "hidden_size": hidden_size,
            "ffn_hidden_size": ffn_hidden_size,
            "layernorm_epsilon": layernorm_epsilon,
            "num_attention_heads": num_attention_heads,
            "init_method": init_method,
            "output_layer_init_method": output_layer_init_method,
            "hidden_dropout": hidden_dropout,
            "attention_dropout": attention_dropout,
            "layer_number": layer_number,
            "kv_channels": kv_channels,
            "self_attn_mask_type": self_attn_mask_type,
            "tp_group": tp_group,
            "tp_size": tp_size,
            "params_dtype": params_dtype,
            "get_rng_state_tracker": get_rng_state_tracker,
            "fuse_wgrad_accumulation": fuse_wgrad_accumulation,
            "seq_length": seq_length,
            "micro_batch_size": micro_batch_size,
            "sequence_parallel": sequence_parallel,
            "apply_residual_connection_post_layernorm": apply_residual_connection_post_layernorm,
            "output_layernorm": output_layernorm,
            "layer_type": layer_type,
            "drop_path_rate": drop_path_rate,
            "set_parallel_mode": tp_size > 1,
            "fuse_qkv_params": True,
            "zero_centered_gamma": zero_centered_gamma,
            "ub_tp_comm_overlap": ub_tp_comm_overlap,
            "ub_bulk_wgrad": ub_bulk_wgrad,
            "ub_bulk_dgrad": ub_bulk_dgrad,
            "device": device,
        }
        te_version = packaging.version.Version(version("transformer-engine"))
        if te_version > packaging.version.Version("1.5.0"):
            for comm in ["ag", "rs"]:
                ub_overlap_flag = "ub_overlap_" + comm
                split_gemm_flag = "ub_split_" + comm
                atomic_gemm_flag = "ub_atomic_gemm_" + comm
                # Use old overlap flags if they were supplied instead
                if ub_overlap_flag in kwargs:
                    transformer_layer_args[ub_overlap_flag] = kwargs[ub_overlap_flag]
                else:
                    transformer_layer_args[ub_overlap_flag] = kwargs.get(split_gemm_flag, True) or kwargs.get(
                        atomic_gemm_flag, False
                    )
            if te_version > packaging.version.Version("1.6.0.dev0"):
                transformer_layer_args["ub_overlap_rs_dgrad"] = kwargs.get("ub_overlap_rs_dgrad", False)
        else:
            transformer_layer_args["ub_split_ag"] = kwargs.get("ub_split_ag", True)
            transformer_layer_args["ub_split_rs"] = kwargs.get("ub_split_rs", True)
            transformer_layer_args["ub_atomic_gemm_ag"] = kwargs.get("ub_atomic_gemm_ag", False)
            transformer_layer_args["ub_atomic_gemm_rs"] = kwargs.get("ub_atomic_gemm_rs", False)
        super().__init__(**transformer_layer_args)

        # Dtype for forward pass - ignore amp O2
        self.dtype = utils_funcs.torch_dtype_from_precision(autocast_dtype, megatron_amp_O2=None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        encoder_output: Optional[torch.Tensor] = None,
        enc_dec_attn_mask: Optional[torch.Tensor] = None,
        inference_params: Optional[Any] = None,
        is_first_microbatch: Optional[bool] = None,
        checkpoint_core_attention: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the transformer layer.
        """
        if self.dtype == torch.float32:
            return super().forward(
                hidden_states,
                attention_mask,
                encoder_output=encoder_output,
                enc_dec_attn_mask=enc_dec_attn_mask,
                inference_params=inference_params,
                is_first_microbatch=is_first_microbatch,
                checkpoint_core_attention=checkpoint_core_attention,
            )
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                encoder_output=encoder_output,
                enc_dec_attn_mask=enc_dec_attn_mask,
                inference_params=inference_params,
                is_first_microbatch=is_first_microbatch,
                checkpoint_core_attention=checkpoint_core_attention,
            )


class TETransformerLayerAutocast(MegatronModule, BaseTransformerLayer):  # type: ignore
    """
    A MegatronModule that wraps the AutocastTransformerLayer.
    """

    def __init__(self, config, layer_number=1, hidden_dropout=None, **kwargs):
        assert (
            HAVE_MEGATRON_CORE and HAVE_TE
        ), "TETransformerLayerAutocast requires Megatron Core and Transformer Engine to be installed."

        # to make type check happy
        if HAVE_MEGATRON_CORE:
            init_kwargs = {'config': config}
        else:
            init_kwargs = {}
        super().__init__(**init_kwargs)

        # since mcore 0.13.0, vp_stage must be provided in layer initialization when PP and VPP are used
        vp_stage = kwargs.get('vp_stage', None)
        if (
            parallel_state.get_pipeline_model_parallel_world_size is not None
            and parallel_state.get_pipeline_model_parallel_world_size() > 1
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() > 1
            and vp_stage is None
        ):
            raise ValueError("vp_stage must be provided if virtual pipeline model parallel is enabled")
        self.layer_number = layer_number + get_transformer_layer_offset(config, vp_stage=vp_stage)

        self.config = config
        self.is_first_microbatch = True
        precision = 'bf16' if config.bf16 else 16

        transformer_layer_args = {
            "hidden_size": config.hidden_size,
            "ffn_hidden_size": config.ffn_hidden_size,
            "layernorm_epsilon": config.layernorm_epsilon,
            "num_attention_heads": config.num_attention_heads,
            "init_method": config.init_method,
            "output_layer_init_method": config.output_layer_init_method,
            "hidden_dropout": config.hidden_dropout,
            "attention_dropout": config.attention_dropout,
            "layer_number": self.layer_number,
            "kv_channels": config.kv_channels,
            "tp_size": parallel_state.get_tensor_model_parallel_world_size(),
            "params_dtype": config.params_dtype,
            "get_rng_state_tracker": tensor_parallel.random.get_cuda_rng_tracker,
            "fuse_wgrad_accumulation": config.gradient_accumulation_fusion,
            "seq_length": None,  # used for jit warmup
            "micro_batch_size": None,  # used for jit warmup
            "sequence_parallel": config.sequence_parallel,
            "apply_residual_connection_post_layernorm": config.apply_residual_connection_post_layernorm,
            "autocast_dtype": precision,
            "ub_tp_comm_overlap": config.tp_comm_overlap,
            "ub_bulk_wgrad": config.tp_comm_bulk_wgrad,
            "ub_bulk_dgrad": config.tp_comm_bulk_dgrad,
            "zero_centered_gamma": config.layernorm_zero_centered_gamma,
            "device": 'cpu' if config.use_cpu_initialization else 'cuda',
        }
        te_version = packaging.version.Version(version("transformer-engine"))
        if te_version > packaging.version.Version("1.5.0"):
            # Use old overlap flags if they were supplied instead
            transformer_layer_args["ub_overlap_ag"] = (
                config.tp_comm_overlap_ag
                if hasattr(config, "tp_comm_overlap_ag")
                else config.tp_comm_split_ag or config.tp_comm_atomic_ag
            )
            transformer_layer_args["ub_overlap_rs"] = (
                config.tp_comm_overlap_rs
                if hasattr(config, "tp_comm_overlap_rs")
                else config.tp_comm_split_rs or config.tp_comm_atomic_rs
            )
            if te_version > packaging.version.Version("1.6.0.dev0"):
                transformer_layer_args["ub_overlap_rs_dgrad"] = (
                    config.tp_comm_overlap_rs_dgrad if hasattr(config, "tp_comm_overlap_rs_dgrad") else False
                )
        else:
            transformer_layer_args["ub_split_ag"] = config.tp_comm_split_ag
            transformer_layer_args["ub_split_rs"] = config.tp_comm_split_rs
            transformer_layer_args["ub_atomic_gemm_ag"] = config.tp_comm_atomic_ag
            transformer_layer_args["ub_atomic_gemm_rs"] = config.tp_comm_atomic_rs
        self.transformer_layer = AutocastTransformerLayer(**transformer_layer_args)

        if self.config.enable_cuda_graph and self.training:
            assert not config.cpu_offloading and config.recompute_granularity is None, "Cudagraphs not supported"
            self.add_module('cudagraph_manager', CudaGraphManager(config))

    # Called by MCore's TransformerBlock.forward
    # megatron/core/transformer/transformer_block.py
    def forward(
        self,
        hidden_states,
        is_first_microbatch=None,
        attention_mask=None,
        context=None,
        context_mask=None,
        inference_params=None,
        **kwargs,
    ):
        """Forward function of TETransformerLayerAutocast. Called by MCore's TransformerBlock.forward."""
        # Use is_first_microbatch argument during CUDA graph capture. Use self.is_first_microbatch otherwise.
        hidden_states = self.transformer_layer.forward(
            hidden_states,
            attention_mask=attention_mask,
            encoder_output=context,
            enc_dec_attn_mask=context_mask,
            inference_params=inference_params,
            is_first_microbatch=is_first_microbatch if is_first_microbatch is not None else self.is_first_microbatch,
            # checkpoint_core_attention,
        )
        self.is_first_microbatch = False
        context = None

        # External CUDA graph requires returned values to be Tensors
        if hasattr(self.config, 'external_cuda_graph') and self.config.external_cuda_graph and self.training:
            return hidden_states
        return hidden_states, context

    def _get_layer_offset(self):

        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()

        num_layers_per_pipeline_rank = (
            self.config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()
        )

        assert (
            self.config.virtual_pipeline_model_parallel_size is None
        ), "Virtual pipeline model parallel size is no longer supported for nemo 1.0"
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

            total_num_layers = self.config.num_layers
            num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
            total_virtual_chunks = total_num_layers // vp_size
            offset = vp_rank * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

        else:
            # Each stage gets a contiguous set of layers.
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                offset = pipeline_rank * num_layers_per_pipeline_rank
            else:
                offset = 0

        return offset

    def sharded_state_dict(self, prefix: str = '', sharded_offsets: tuple = (), metadata=None):
        """Get the sharded state dict for the transformer layer."""
        TENSOR_PARALLEL_LAYERS_AXIS_MAP = {
            'self_attention.layernorm_qkv.weight': 0,
            'self_attention.layernorm_qkv.bias': 0,
            "self_attention.proj.weight": 1,
            "layernorm_mlp.fc1_weight": 0,
            "layernorm_mlp.fc1_bias": 0,
            "layernorm_mlp.fc2_weight": 1,
        }

        state_dict = self.state_dict(prefix='', keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            state_dict, prefix, TENSOR_PARALLEL_LAYERS_AXIS_MAP, sharded_offsets
        )

        # TODO: we need to add sharded_state_dict_keys_map to the config. Like in TransformerLayer submodules config
        # prefixed_map = {
        #    f'{prefix}{k}': f'{prefix}{v}'
        #    for k, v in self.config.sharded_state_dict_keys_map.items()
        # }

        # if prefixed_map:
        #    apply_prefix_mapping(sharded_state_dict, prefixed_map)

        return sharded_state_dict

    def __call__(self, *args, **kwargs):
        if hasattr(self, 'cudagraph_manager'):
            return self.cudagraph_manager(self, args, kwargs)
        return super().__call__(*args, **kwargs)


# Use this spec to use the full Transformer layer from Transformer Engine
def get_gpt_full_te_layer_autocast_spec(transformer_config, vp_stage: Optional[int] = None) -> ModuleSpec:
    """Get the ModuleSpec for full Transformer layer from Transformer Engine."""
    assert HAVE_MEGATRON_CORE and HAVE_TE, "Please ensure Megatron Core and Transformer Engine are installed."
    num_layers = get_num_layers_to_build(transformer_config, vp_stage=vp_stage)
    return TransformerBlockSubmodules(
        layer_specs=[ModuleSpec(module=TETransformerLayerAutocast)] * num_layers, layer_norm=FusedLayerNorm
    )
