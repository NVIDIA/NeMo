# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer based language model."""
import torch
import copy
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.transformer import ParallelTransformer
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    attn_mask_postprocess,
    build_attention_mask_3d,
)
from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType

try:
    from apex.transformer import parallel_state
    from apex.transformer.enums import AttnMaskType, ModelType
    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False
    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()
    ModelType = ApexGuardDefaults()

__all__ = ["MegatronPerceiverEncoderModule"]


class MegatronPerceiverEncoderModule(MegatronModule):
    """Transformer encoder model.
    """

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        hidden_size,
        ffn_hidden_size,
        num_layers,
        num_attention_heads,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        pre_process=True,
        post_process=True,
        use_cpu_initialization=False,
        encoder_attn_mask_type=AttnMaskType.padding,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        position_embedding_type='learned_absolute',
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        layernorm_epsilon=1e-5,
        bias_gelu_fusion=True,
        bias_dropout_add_fusion=True,
        masked_softmax_fusion=True,
        persist_layer_norm=False,
        openai_gelu=False,
        onnx_safe=False,
        activation='gelu',
        bias=True,
        normalization='layernorm',
        transformer_block_type='pre_ln',
        headscale=False,
        parent_model_type=ModelType.encoder_or_decoder,
        hidden_steps=32,
        num_self_attention_per_cross_attention=1,
        num_init_cross_attn_layers=1
    ):
        super(MegatronPerceiverEncoderModule, self).__init__()

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_method = init_method
        self.model_attn_mask_type = encoder_attn_mask_type
        self.hidden_dropout = hidden_dropout
        self.output_layer_init_method = output_layer_init_method
        self.parent_model_type = parent_model_type
        self.normalization = normalization
        self.transformer_block_type = transformer_block_type
        self.hidden_steps = hidden_steps
        self.num_self_attention_per_cross_attention = num_self_attention_per_cross_attention
        self.num_init_cross_attn_layers = num_init_cross_attn_layers

        assert self.num_self_attention_per_cross_attention >= 1
        assert self.num_init_cross_attn_layers >= 1
        assert self.hidden_steps >= 1

        if kv_channels is None:

            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads
        
        latent_attention_mask = torch.ones(hidden_steps, hidden_steps)
        self.register_buffer('latent_attention_mask', latent_attention_mask)

        if parallel_state.is_pipeline_first_stage():
            self.init_hidden = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(hidden_steps, hidden_size)))
            self.init_cross_att = ParallelTransformer(
                layer_type=LayerType.decoder,
                init_method=self.init_method,
                output_layer_init_method=self.output_layer_init_method,
                num_layers=self.num_init_cross_attn_layers,
                hidden_size=self.hidden_size,
                num_attention_heads=num_attention_heads,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                ffn_hidden_size=ffn_hidden_size,
                self_attn_mask_type=self.model_attn_mask_type,
                pre_process=self.pre_process,
                post_process=False, # This is to avoid the final layernorm and transpose.
                precision=precision,
                fp32_residual_connection=fp32_residual_connection,
                activations_checkpoint_method=activations_checkpoint_method,
                activations_checkpoint_num_layers=activations_checkpoint_num_layers,
                layernorm_epsilon=layernorm_epsilon,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                position_embedding_type=position_embedding_type,
                relative_attention_num_buckets=relative_attention_num_buckets,
                relative_attention_max_distance=relative_attention_max_distance,
                use_cpu_initialization=use_cpu_initialization,
                bias_gelu_fusion=bias_gelu_fusion,
                bias_dropout_fusion=bias_dropout_add_fusion,
                masked_softmax_fusion=masked_softmax_fusion,
                persist_layer_norm=persist_layer_norm,
                openai_gelu=openai_gelu,
                onnx_safe=onnx_safe,
                activation=activation,
                bias=bias,
                normalization=normalization,
                model_type=parent_model_type,
                transformer_block_type=transformer_block_type,
                headscale=headscale,
            )

        cross_attn_layer = ParallelTransformer(
            layer_type=LayerType.decoder,
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            ffn_hidden_size=ffn_hidden_size,
            self_attn_mask_type=self.model_attn_mask_type,
            pre_process=self.pre_process,
            post_process=False, # This is to avoid the final layernorm and transpose.
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            layernorm_epsilon=layernorm_epsilon,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            position_embedding_type=position_embedding_type,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            use_cpu_initialization=use_cpu_initialization,
            bias_gelu_fusion=bias_gelu_fusion,
            bias_dropout_fusion=bias_dropout_add_fusion,
            masked_softmax_fusion=masked_softmax_fusion,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            activation=activation,
            bias=bias,
            normalization=normalization,
            model_type=parent_model_type,
            transformer_block_type=transformer_block_type,
            headscale=headscale,
        )
        self.cross_attn_layers = torch.nn.ModuleList([copy.deepcopy(cross_attn_layer) for _ in range(self.num_layers)])

        self_attn_layer = ParallelTransformer(
            layer_type=LayerType.encoder,
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            ffn_hidden_size=ffn_hidden_size,
            self_attn_mask_type=self.model_attn_mask_type,
            pre_process=self.pre_process,
            post_process=False, # This is to avoid the final layernorm and transpose.
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            layernorm_epsilon=layernorm_epsilon,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            position_embedding_type=position_embedding_type,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            use_cpu_initialization=use_cpu_initialization,
            bias_gelu_fusion=bias_gelu_fusion,
            bias_dropout_fusion=bias_dropout_add_fusion,
            masked_softmax_fusion=masked_softmax_fusion,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            activation=activation,
            bias=bias,
            normalization=normalization,
            model_type=parent_model_type,
            transformer_block_type=transformer_block_type,
            headscale=headscale,
        )
        self.self_attn_layers = torch.nn.ModuleList([copy.deepcopy(self_attn_layer) for _ in range(self.num_layers * self.num_self_attention_per_cross_attention)])

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""
        # TODO: Fix this.
        pass

    def forward(
        self, enc_input, enc_attn_mask, layer_past=None, get_key_value=False,
    ):
        # convert to Megatron mask
        latent_attention_mask_3d = build_attention_mask_3d(
            source_mask=self.latent_attention_mask, target_mask=self.latent_attention_mask, attn_mask_type=AttnMaskType.padding
        )
        enc_dec_attn_mask_3d = build_attention_mask_3d(
            source_mask=self.latent_attention_mask, target_mask=enc_attn_mask, attn_mask_type=AttnMaskType.padding,
        )

        if parallel_state.is_pipeline_first_stage():
            hidden_states = self.init_hidden.unsqueeze(0).expand(enc_input.size(0), -1, -1)
            hidden_states = self.init_cross_att(
                hidden_states=hidden_states,
                attention_mask=latent_attention_mask_3d,
                enc_dec_attn_mask=enc_dec_attn_mask_3d,
                encoder_output=enc_input,
            )
        else:
            hidden_states = enc_input

        for i in range(self.num_layers):
            residual = hidden_states

            hidden_states = self.cross_attn_layers[i](
                hidden_states=hidden_states,
                attention_mask=latent_attention_mask_3d,
                enc_dec_attn_mask=enc_dec_attn_mask_3d,
                encoder_output=enc_input,
            )
            for j in range(self.num_self_attention_per_cross_attention):
                hidden_states = self.self_attn_layers[i * self.num_self_attention_per_cross_attention + j](
                    hidden_states=hidden_states,
                    attention_mask=latent_attention_mask_3d,
                )
            
            hidden_states += residual

        return hidden_states
