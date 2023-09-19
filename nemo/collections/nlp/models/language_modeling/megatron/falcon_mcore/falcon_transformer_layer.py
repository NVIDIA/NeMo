# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import re

import torch
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor

# from megatron.core.transformer.attention import SelfAttention
# change attention due to extra layernorm before mlp, ln_mlp.
from nemo.collections.nlp.models.language_modeling.megatron.falcon_mcore.falcon_attention import SelfAttention

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""


class FalconTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    
    Args: 
        new_decoder_architecture (bool):
    Whether to use Falcon's new decoder architecture that were used in 7B/40B/180B variants.
        
        parallel_attention (bool):
    Whether to use parallel attention, which computes attention in parallel with feed forward layer.
        
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int = 1,
        self_attn_mask_type=AttnMaskType.padding,
        parallel_attention=False,
        new_decoder_architecture=False,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config

        self.layer_number = layer_number + self._get_layer_offset()

        self.self_attn_mask_type = self_attn_mask_type

        self.new_decoder_architecture = new_decoder_architecture
        self.parallel_attention = parallel_attention

        # Layernorm on the input data.
        # TODO: add pytorch only layernorm
        self.input_layernorm = self._create_identity_op()

        self.mlp_layernorm = self._create_identity_op() if self.new_decoder_architecture else None

        if self.new_decoder_architecture or self.parallel_attention:
            self.post_self_attn_layernorm = None
        else:
            # Layernorm on the attention output
            self.post_self_attn_layernorm = self._create_identity_op()

        # Self attention.
        self.self_attention = SelfAttention(
            config=self.config, layer_number=layer_number, attn_mask_type=self_attn_mask_type,
        )

        # MLP
        self.mlp = MLP(config=self.config)

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def _create_identity_op(self):
        """Helper function to create an IdentityOp with common parameters."""
        return IdentityOp(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            normalization=self.config.normalization,
        )

    def _get_layer_offset(self):

        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()

        num_layers_per_pipeline_rank = (
            self.config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()
        )

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

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_output=None,
        enc_dec_attn_mask=None,
        inference_params=None,
        rotary_pos_emb=None,
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        input_mlp_ln = layernorm_output

        # Self attention.
        attention_output_with_bias = self.self_attention(
            layernorm_output, attention_mask, inference_params=inference_params, rotary_pos_emb=rotary_pos_emb,
        )

        # Residual connection.
        if self.config.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # falcon specific
        if self.new_decoder_architecture:
            mlp_ln_output = self.mlp_layernorm(hidden_states)

        bias_dropout_add_func = get_bias_dropout_add(self.training, self.config.bias_dropout_fusion)

        # bias_dropout_add fusion returning fp32 instead of bf16
        with self.bias_dropout_add_exec_handler():
            layernorm_input = bias_dropout_add_func(attention_output_with_bias, residual, self.config.hidden_dropout)

        # falcon specific
        if not self.new_decoder_architecture:
            if self.parallel_attention:
                layernorm_output = input_mlp_ln
            else:
                layernorm_output = self.post_self_attn_layernorm(layernorm_input)
                residual = (
                    layernorm_input if not self.config.apply_residual_connection_post_layernorm else layernorm_output
                )
        else:
            layernorm_output = mlp_ln_output

        # MLP.
        mlp_output_with_bias = self.mlp(layernorm_output)

        # falcon specific:
        if self.new_decoder_architecture or self.parallel_attention:
            mlp_output_with_bias = mlp_output_with_bias + attention_output_with_bias

        with self.bias_dropout_add_exec_handler():
            output = bias_dropout_add_func(mlp_output_with_bias, residual, self.config.hidden_dropout)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)

        return output

    def sharded_state_dict(self, prefix=''):

        # state_dict = self.state_dict(prefix=prefix, keep_vars=True)
        state_dict = self.state_dict(keep_vars=True)

        tensor_parallel_layers_axis_map = {
            'self_attention.linear_qkv.weight': 0,
            'self_attention.linear_qkv.bias': 0,
            'self_attention.linear_proj.weight': 1,
            'mlp.linear_fc1.weight': 0,
            'mlp.linear_fc1.bias': 0,
            'mlp.linear_fc2.weight': 1,
        }

        offset = self._get_layer_offset()
        num_layers = self.config.num_layers

        sharded_state_dict = {}

        for layer_name in state_dict.keys():
            tensor = state_dict[layer_name]
            global_layer_offset = self.layer_number - 1  # self.layer_number starts at 1
            layer_key = f'{prefix}{global_layer_offset - offset}.{layer_name}'  # module list index in TransformerBlock
            sharded_offsets = [(0, global_layer_offset, num_layers)]  # PP sharding

            if layer_name in tensor_parallel_layers_axis_map:
                tp_axis = tensor_parallel_layers_axis_map[layer_name]
                # TP sharding
                sharded_offsets.append(
                    [
                        tp_axis + 1,  # +1 for PP dimension
                        parallel_state.get_tensor_model_parallel_rank(),
                        parallel_state.get_tensor_model_parallel_world_size(),
                    ]
                )
                replica_id = parallel_state.get_data_parallel_rank()
            else:
                replica_id = (
                    parallel_state.get_data_parallel_rank() * parallel_state.get_data_parallel_world_size()
                    + parallel_state.get_tensor_model_parallel_rank()
                )

            if layer_name.endswith('._extra_state'):
                sharded_state_dict[layer_key] = ShardedObject(
                    f'{prefix}{layer_name}', tensor, (num_layers,), (global_layer_offset,), replica_id,
                )

            else:
                sharded_state_dict[layer_key] = ShardedTensor.from_rank_offsets(
                    f'{prefix}{layer_name}',
                    tensor,
                    *sharded_offsets,
                    replica_id=replica_id,
                    prepend_axis_num=1,  # for PP sharding
                )

        return sharded_state_dict
