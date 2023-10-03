# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Union

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor

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

@dataclass
class FalconTransformerLayerSubmodules:
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    post_self_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    
    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp


class FalconTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
        
    """

    def __init__(
        self,
        config: TransformerConfig, # should come from FalconTransformerConfig class
        submodules: FalconTransformerLayerSubmodules,
        layer_number: int = 1,
        self_attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        

        self.layer_number = layer_number + self._get_layer_offset()

        self.self_attn_mask_type = self_attn_mask_type

        if hasattr(self.config, 'new_decoder_architecture'):
            self.new_decoder_architecture = self.config.new_decoder_architecture
        else:
            self.new_decoder_architecture = None  
        
        if hasattr(self.config, 'parallel_attention'):
            self.parallel_attention = self.config.parallel_attention
        else:
            self.parallel_attention = None 


        ## [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            normalization=self.config.normalization,
        )

        ## [Module 2: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention, config=self.config, layer_number=layer_number,
        )

        ## [Module 3: BiasDropoutFusion] Optional
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        ## [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        if self.new_decoder_architecture or self.parallel_attention:
            self.post_self_attn_layernorm = None
        else:
            self.post_self_attn_layernorm = build_module(
                submodules.post_self_attn_layernorm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
                persist_layer_norm=self.config.persist_layer_norm,
                sequence_parallel=self.config.sequence_parallel,
                zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
                normalization=self.config.normalization,
            )

        ## [Module 5: pre mlp layernorm] Optional Layernorm before MLP, used in Falcon's new decoder architecture
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            persist_layer_norm=self.config.persist_layer_norm,
            sequence_parallel=self.config.sequence_parallel,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            normalization=self.config.normalization,
        ) if self.new_decoder_architecture else None

        ## [Module 6: MLP block]
        self.mlp = build_module(submodules.mlp, config=self.config)

        ## [Module 7: BiasDropoutFusion] Optional
        self.mlp_bda = build_module(submodules.mlp_bda)
        
        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

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
        
        # Residual connection.
        residual = hidden_states
        
        if self.new_decoder_architecture:
            mlp_ln_output = self.pre_mlp_layernorm(hidden_states)
        
        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        input_mlp_ln = input_layernorm_output

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
        )
        
        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.config.hidden_dropout
            )

        if not self.new_decoder_architecture:
            if self.parallel_attention:
                layernorm_output = input_mlp_ln
            else:
                residual = hidden_states
                layernorm_output = self.post_self_attn_layernorm(hidden_states)

        else:
            layernorm_output = mlp_ln_output

        mlp_output_with_bias = self.mlp(layernorm_output)
        
        # falcon specific:
        if self.new_decoder_architecture or self.parallel_attention:
            mlp_output= mlp_output_with_bias[0]
            attn_output= attention_output_with_bias[0]
            mlp_output_without_bias = mlp_output + attn_output
            mlp_output_with_bias = (mlp_output_without_bias, None)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.config.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

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
                    parallel_state.get_data_parallel_rank()
                    * parallel_state.get_data_parallel_world_size()
                    + parallel_state.get_tensor_model_parallel_rank()
                )

            if layer_name.endswith('._extra_state'):
                sharded_state_dict[layer_key] = ShardedObject(
                    f'{prefix}{layer_name}',
                    tensor,
                    (num_layers,),
                    (global_layer_offset,),
                    replica_id,
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
