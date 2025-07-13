from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor

from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import make_viewless_tensor
from megatron.core import tensor_parallel
from contextlib import nullcontext
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.attention import CrossAttention, CrossAttentionSubmodules
from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.wrapped_tensor import WrappedTensor

from nemo.collections.vlm.layer_specs import get_norm_mlp_module_spec_te
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm


def get_layer_spec_thinker() -> ModuleSpec:
    """
    Cross-attention Transformer Layer Spec w/ TE Modules

    this forms the base transformer layer spec that is copied as many layers as needed by the thinker module
    """
    attn_mask_type = AttnMaskType.no_mask

    mlp = get_norm_mlp_module_spec_te()
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            cross_attention=ModuleSpec(
                module=CrossAttention,
                params={"attn_mask_type": attn_mask_type},
                submodules=CrossAttentionSubmodules(
                    linear_q=TELayerNormColumnParallelLinear,
                    linear_kv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            cross_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )

class ThinkingAttnRefineModule(TransformerBlock):
    """
    A specialized transformer block that alternates between processing hidden states 
    and context through cross attention layers.

    For reference: 
    Qwen2VL uses a simple TransformerBlock as a decoder
    Qwen25VL uses a modified TransformerBlock as a decoder where the sequence packing is done cleverly for alternating between chunked attention and full attention

    in our case, we want to alternate cross attention between hidden states and context
    
    This module implements a "thinking" mechanism where each layer alternates between:
    1. Using hidden states as query and context as key/value
    2. Using context as query and hidden states as key/value
    
    This allows bidirectional refinement between the two sequences.
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        model_comm_pgs = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(
            config=config,
            spec=spec,
            post_layer_norm=post_layer_norm,
            pre_process=pre_process,
            post_process=post_process,
            model_comm_pgs=model_comm_pgs,
            vp_stage=vp_stage,
        )

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
    ):
        """
        Forward pass that alternates between processing hidden states and context.
        
        On even layers: hidden_states = layer(hidden_states, context=context)
        On odd layers: context = layer(context, context=hidden_states)
        
        Args:
            hidden_states: Primary sequence tensor [seq_len, batch, hidden_size]
            attention_mask: Attention mask for hidden states
            context: Secondary sequence tensor [context_len, batch, hidden_size] 
            context_mask: Attention mask for context
            rotary_pos_emb: Optional rotary positional embeddings
            attention_bias: Optional attention bias tensor
            inference_context: Optional inference optimization context
            packed_seq_params: Optional packed sequence parameters
            
        Returns:
            tuple: (hidden_states, context) - The refined sequences after alternating processing
        """
        # Handle wrapped tensor case
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            hidden_states = self.input_tensor

        # Make tensors viewless for gradient computation
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        if context is not None:
            context = make_viewless_tensor(inp=context, requires_grad=True, keep_graph=True)

        # Setup RNG context for sequence parallelism
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # Determine if we need outer or inner FP8 contexts
        use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == "delayed"
        use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != "delayed"
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

        with rng_context, outer_fp8_context:
            # Forward pass through layers with alternating processing
            for i, layer in enumerate(self.layers):
                inner_fp8_context = (
                    get_fp8_context(self.config, layer.layer_number - 1)
                    if use_inner_fp8_context
                    else nullcontext()
                )
                
                with self.offload_context, inner_fp8_context:
                    if i % 2 == 0:
                        # Even layers: Process hidden states with context attention
                        hidden_states, _ = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )
                    else:
                        # Odd layers: Process context with hidden states attention
                        context, _ = layer(
                            hidden_states=context,
                            attention_mask=context_mask,
                            context=hidden_states,
                            context_mask=attention_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )

                # Handle CPU offloading if enabled
                if (
                    torch.is_grad_enabled()
                    and self.config.cpu_offloading
                    and self.group_prefetch_offload_commit_async is not None
                ):
                    hidden_states = self.group_prefetch_offload_commit_async(hidden_states)
                    if context is not None:
                        context = self.group_prefetch_offload_commit_async(context)

        # Apply final layer norm if present
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            if context is not None:
                context = self.final_layernorm(context)
            
            # Make output tensors viewless
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
            if context is not None:
                context = make_viewless_tensor(inp=context, requires_grad=True, keep_graph=True)

        # Handle empty transformer block case
        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            hidden_states = hidden_states.clone()
            if context is not None:
                context = context.clone()

        return hidden_states, context
