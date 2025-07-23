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
from typing import Any, Optional, Union

import torch
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    deprecate_inference_params,
    is_te_min_version,
    log_single_rank,
    make_viewless_tensor,
    nvtx_range_pop,
    nvtx_range_push,
)
from torch import Tensor

from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig


@dataclass
class HyenaLayerSubmodules:
    """Submodules for the HyenaLayer."""

    norm: Union[ModuleSpec, type] = IdentityOp
    mixer: Union[ModuleSpec, type] = IdentityOp
    hyena_bda: Union[ModuleSpec, type] = IdentityOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityOp


class HyenaLayer(MegatronModule):
    """Top level Hyena Layer."""

    def __init__(
        self,
        transformer_config: TransformerConfig,
        hyena_config: HyenaConfig,
        operator_type,
        max_sequence_length,
        submodules: HyenaLayerSubmodules,
        layer_number: int = 1,
        residual_in_fp32=False,
        model_comm_pgs=None,
    ):
        """
        Top level Hyena Layer
        """
        super().__init__(config=transformer_config)
        self.transformer_config = transformer_config
        self.hyena_config = hyena_config
        self.layer_number = layer_number
        self.hidden_dropout = transformer_config.hidden_dropout
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = build_module(
            submodules.mixer,
            self.transformer_config,
            self.hyena_config,
            max_sequence_length,
            layer_number=layer_number,
            operator_type=operator_type,
            model_comm_pgs=model_comm_pgs,
        )
        self.norm = build_module(
            submodules.norm,
            self.transformer_config,
            self.transformer_config.hidden_size,
            eps=self.transformer_config.layernorm_epsilon,
        )

        self.hyena_bda = build_module(submodules.hyena_bda)
        self.bias_dropout_add_exec_handler = torch.enable_grad

        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.transformer_config,
            hidden_size=self.transformer_config.hidden_size,
            eps=self.transformer_config.layernorm_epsilon,
        )

        self.mlp = build_module(submodules.mlp, config=self.transformer_config, tp_group=model_comm_pgs.tp)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        self.mlp_bda = build_module(submodules.mlp_bda)
        if hasattr(self.mlp_bda, 'set_layer_number'):
            self.mlp_bda.set_layer_number(self.layer_number)
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,  # Not used in HyenaLayer
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """Forward pass for the HyenaLayer."""
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        residual = hidden_states
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = hidden_states.to(dtype=self.transformer_config.params_dtype)
        hidden_states = self.norm(hidden_states)

        mixer_out_with_bias = self.mixer(hidden_states, inference_context=inference_context)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.hyena_bda(self.training, self.transformer_config.bias_dropout_fusion)(
                mixer_out_with_bias, residual, self.hidden_dropout
            )

        residual = hidden_states

        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.transformer_config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        return hidden_states, context

    # def get_layer_static_inputs(self, seq_length, micro_batch_size):
    #     """
    #     Get the static inputs for the transformer layer.

    #     Returns:
    #         Dict[str, torch.Tensor]: A dictionary containing the static inputs for the layer.
    #     """
    #     # Calculate data shape related values.
    #     context_parallel_size = self.config.context_parallel_size
    #     slen_per_cp = seq_length // context_parallel_size
    #     sequence_parallel = self.config.sequence_parallel
    #     tensor_model_parallel_size = self.config.tensor_model_parallel_size
    #     slen_per_cptp = (
    #         slen_per_cp // tensor_model_parallel_size if sequence_parallel else slen_per_cp
    #     )

    #     static_inputs = {}
    #     static_inputs["hidden_states"] = torch.ones(
    #         (slen_per_cptp, micro_batch_size, self.config.hidden_size),
    #         dtype=torch.bfloat16,
    #         requires_grad=True,
    #         device=torch.cuda.current_device(),
    #     )
    #     return static_inputs

    # def setup_manual_hooks(self, make_hook_func):
    #     """
    #     Set CUDA Graph manual hooks for the modules that contain direct parameters and are
    #     covered by cudagraphs.
    #     """
    #     self.cuda_graph_manual_hooks = []

    #     # Select the modules who contain direct parameters and are covered by cudagraphs.
    #     # Add these modules to the `cuda_graph_manual_hooks` because their hooks will not
    #     # be automatically triggered when they go through the CUDA Graph path.
    #     high_level_modules = [self]

    #     param_modules = []
    #     for module in high_level_modules:
    #         for submodule in module.modules():
    #             if next(submodule.parameters(recurse=False), None) is not None:
    #                 # Module contains direct parameters.
    #                 param_modules.append(submodule)
    #                 continue
    #     if len(param_modules) > 0:
    #         for module in param_modules:
    #             self.cuda_graph_manual_hooks.append((make_hook_func(), (module,)))

    # def _cuda_graph_capture(self, *args, **kwargs):
    #     """
    #     CUDA Graph capture for this layer. There are some differences from the normal pass:
    #     1. In some conditions CUDA graph cannot cover the entire layer. The `cuda_graph_scope`
    #        attribute can be set to control the scope of the CUDA graph.
    #     2. If context is None, it cannot be returned as output.
    #     """
    #     hidden_states, context = self.forward(*args, **kwargs)

    #     cuda_graph_outputs = [hidden_states]

    #     if context is not None:
    #         cuda_graph_outputs.append(context)
    #     return tuple(cuda_graph_outputs)

    # def _cuda_graph_replay(self, *args, **kwargs):
    #     """
    #     CUDA graph replay for this layer and microbatch
    #     `self.current_microbatch`. TransformerEngine versions>=1.10
    #     allow keyword arguments with CUDA graph. However, CUDA graph
    #     acccepts only Tensor inputs and Tensor outputs. Hence,
    #     `inference_context` and `packed_seq_params` are excluded from
    #     input list while output is limited to `hidden_states`.
    #     """

    #     def _check_cuda_graph_replay_args(*args, **kwargs):
    #         """Helper function to get optional tensor arguments for CUDA graph."""

    #         assert len(args) <= 1, "At most one positional argument `hidden_states` is expected."
    #         if len(args) == 1:
    #             hidden_states = args[0]
    #         else:
    #             hidden_states = kwargs.pop("hidden_states")
    #         cudagraph_args = [hidden_states]

    #         optional_inputs = kwargs.copy()
    #         optional_inputs['is_first_microbatch'] = self.current_microbatch == 0
    #         try:
    #             import transformer_engine.pytorch as te  # pylint: disable=unused-import

    #         except ImportError:
    #             raise RuntimeError("CUDAGraph requires TransformerEngine, but not installed")
    #         return tuple(cudagraph_args), optional_inputs

    #     cg_index = self.current_microbatch % len(self.cuda_graphs)
    #     assert ('inference_context' not in kwargs or kwargs['inference_context'] is None) and (
    #         'packed_seq_params' not in kwargs or kwargs['packed_seq_params'] is None
    #     ), "CUDA graph accepts only Tensor inputs."
    #     cudagraph_args, cudagraph_kwargs = _check_cuda_graph_replay_args(*args, **kwargs)

    #     for hook, hook_args in self.cuda_graph_manual_hooks:
    #         hook(*hook_args)
    #     cuda_graph_output = self.cuda_graphs[cg_index](*cudagraph_args, **cudagraph_kwargs)

    #     if cudagraph_kwargs['context'] is not None:
    #         context = cuda_graph_output[-1]
    #         cuda_graph_output = cuda_graph_output[:-1]
    #     else:
    #         context = None
    #     output = cuda_graph_output[0]
    #     return output, context

    # def __call__(self, *args, **kwargs):
    #     # Training and validation mode CUDA graphs
    #     if hasattr(self, 'cudagraph_manager') and kwargs.get('inference_context') is None:
    #         return self.cudagraph_manager(self, args, kwargs)
    #     # Inference mode. CUDA graphs are used in the decode phase only, when attn mask is None
    #     elif not self.training and (
    #         hasattr(self, 'cudagraph_manager')
    #         and kwargs['attention_mask'] is None
    #         and (
    #             (
    #                 kwargs.get('inference_context') is not None
    #                 and kwargs['inference_context'].is_decode_only()
    #             )
    #             or (
    #                 kwargs.get('inference_params') is not None
    #                 and kwargs['inference_params'].is_decode_only()
    #             )
    #         )
    #     ):
    #         assert (
    #             kwargs.get('attention_mask') is None
    #         ), f"Attention mask must not be set when using CUDA graphs for decode"
    #         return self.cudagraph_manager(self, args, kwargs)
    #     elif self.config.external_cuda_graph and self.training:
    #         if not self.cuda_graphs:
    #             # Do CUDA Graphs capture.
    #             cuda_graph_func = self._cuda_graph_capture
    #         else:
    #             # Do CUDA Graphs replay.
    #             cuda_graph_func = self._cuda_graph_replay
    #         return cuda_graph_func(*args, **kwargs)
    #     return super(MegatronModule, self).__call__(*args, **kwargs)
