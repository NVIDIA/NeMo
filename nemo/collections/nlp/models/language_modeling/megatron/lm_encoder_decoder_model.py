# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""T5 model."""

import torch
from apex.transformer import tensor_parallel
from apex.transformer.enums import AttnMaskType

from nemo.collections.nlp.modules.common.megatron.language_model import get_language_model,
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import (
    parallel_lm_logits,
    init_method_normal,
    scaled_init_method_normal,
    build_position_ids,
    enc_dec_extended_attention_mask,
)
from nemo.collections.nlp.modules.common.megatron.megatron_encoders import get_encoder_model
from nemo.collections.nlp.modules.common.megatron.megatron_decoders import get_decoder_model
from nemo.collections.nlp.modules.common.megatron.megatron_encoder_decoder import MegatronTransformerEncoderDecoderModel


class MegatronLMHead(MegatronModule):
    """Masked LM head for token-based encoder-decoder models (e.g., T5)

    Arguments:
        mpu_vocab_size: model parallel size of vocabulary.
        parallel_output: wether output logits being distributed or not.
    """

    def __init__(self, mpu_vocab_size, parallel_output):
        super(MegatronLMHead, self).__init__()

        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        self.bias.model_parallel = True
        self.bias.partition_dim = 0
        self.bias.stride = 1
        self.parallel_output = parallel_output

    def forward(self, hidden_states, word_embeddings_weight):
        output = parallel_lm_logits(hidden_states, word_embeddings_weight, self.parallel_output, bias=self.bias)
        return output


class LMEncoderDecoderModel(MegatronModule):
    """Token-based (input/output is tokens) encoder-decoder model (e.g. T5 Language model.)"""

    def __init__(
        self,
        encoder_arch,
        decoder_arch,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        num_layers,
        num_attention_heads,
        ffn_hidden_size,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
        init_method_std=0.02,
        fp16_lm_cross_entropy=False,
        use_cpu_initialization=False,
        hidden_dropout=0.1,
        fp32_residual_connection=False,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        layernorm_epsilon=1e-5,
        bias_gelu_fusion=True,
        openai_gelu=False,
        onnx_safe=False,
        hidden_steps=-1,
        hidden_blocks=1,
    ):
        super(LMEncoderDecoderModel, self).__init__()

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        encoder_input_embedder = Embedding(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            max_sequence_length=max_position_embeddings,
            init_method=init_method_normal(init_method_std),
            num_tokentypes=num_tokentypes,
            use_cpu_initialization=use_cpu_initialization,
            embedding_dropout_prob=hidden_dropout,
        )

        encoder = get_encoder_model(
            arch=encoder_arch,
            hidden_size=hidden_size,
            ffn_hidden_size=hidden_size,
            num_layers=num_layers,
            max_position_embeddings=max_position_embeddings,
            num_tokentypes=num_tokentypes,
            vocab_size=vocab_size,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            init_method=init_method_normal(init_method_std),
            scaled_init_method=scaled_init_method_normal(init_method_std, num_layers),
            encoder_attn_mask_type=AttnMaskType.padding,
            pre_process=pre_process,
            post_process=post_process,
            init_method_std=init_method_std,
            use_cpu_initialization=init_method_std,
            hidden_dropout=hidden_dropout,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            layernorm_epsilon=layernorm_epsilon,
            bias_gelu_fusion=bias_gelu_fusion,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            use_soft_prompts=use_soft_prompts,
            num_prompt_tokens=num_prompt_tokens,
            prompt_tags=prompt_tags,
            hidden_steps=hidden_steps,
            hidden_blocks=hidden_steps,
        )

        decoder_input_embedder = encoder_input_embedder
        decoder = get_decoder_model(
            arch=decoder_arch,
            hidden_size=hidden_size,
            ffn_hidden_size=hidden_size,
            num_layers=num_layers,
            max_position_embeddings=max_position_embeddings,
            num_tokentypes=num_tokentypes,
            vocab_size=vocab_size,
            num_attention_heads=num_attention_heads,
            decoder_attn_mask_type=decoder_attn_mask_type,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            init_method=init_method_normal(init_method_std),
            scaled_init_method=scaled_init_method_normal(init_method_std, num_layers),
            encoder_attn_mask_type=AttnMaskType.padding,
            pre_process=pre_process,
            post_process=post_process,
            init_method_std=init_method_std,
            use_cpu_initialization=init_method_std,
            hidden_dropout=hidden_dropout,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            layernorm_epsilon=layernorm_epsilon,
            bias_gelu_fusion=bias_gelu_fusion,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            use_soft_prompts=use_soft_prompts,
            num_prompt_tokens=num_prompt_tokens,
            prompt_tags=prompt_tags,
            hidden_steps=hidden_steps,
            hidden_blocks=hidden_steps,
        )

        self.enc_dec_model = MegatronTransformerEncoderDecoderModel(
            encoder_input_embedder=encoder_input_embedder,
            encoder=encoder,
            decoder_input_embedder=decoder_input_embedder,
            decoder=decoder,
        )
        self._enc_dec_model_key = "enc_dec_model"
        self._encoder_input_embedder_key = "encoder_input_embedder"
        self._decoder_input_embedder_key = "decoder_input_embedder"

        self.lm_head = MegatronLMHead(self.language_model.embedding.word_embeddings.weight.size(0), parallel_output)
        self._lm_head_key = 'lm_head'

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
        tokentype_ids=None,
        lm_labels=None,
        enc_hidden_states=None,
        output_enc_hidden_only=False,
    ):

        # Converting the attention masks to proper parameter settings
        encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask = enc_dec_extended_attention_mask(
            [encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask]
        )

        encoder_position_ids = build_position_ids(encoder_input_ids)

        # Handle case when decoder_input_ids is None to get just the encoder hidden states.
        decoder_position_ids = build_position_ids(decoder_input_ids) if decoder_input_ids is not None else None

        # FIXME: add learnable positional encoding into embeddings

        lm_output = self.enc_dec_model(
            enc_input_ids=encoder_input_ids,
            enc_position_ids=encoder_position_ids,
            enc_attn_mask=encoder_attn_mask,
            dec_input_ids=decoder_input_ids,
            dec_position_ids=decoder_position_ids,
            dec_attn_mask=decoder_attn_mask,
            enc_dec_attn_mask=encoder_decoder_attn_mask,
            tokentype_ids=tokentype_ids,
            enc_hidden_states=enc_hidden_states,
            output_enc_hidden_only=output_enc_hidden_only,
        )

        if output_enc_hidden_only:
            return lm_output

        encoder_output, encoder_output_mask, decoder_output = lm_output

        # TODO: do we want to return dict instead of tuple? Will allow more flexibility in extending model

        # Output.
        lm_logits = self.lm_head(decoder_output, self.enc_dec_model.decoder_input_embedder.word_embeddings.weight)

        if lm_labels is None:
            return lm_logits, encoder_output, encoder_output_mask
        else:
            if self.fp16_lm_cross_entropy:
                assert lm_logits.dtype == torch.half
                lm_loss = tensor_parallel.vocab_parallel_cross_entropy(lm_logits, lm_labels)
            else:
                lm_loss = tensor_parallel.vocab_parallel_cross_entropy(lm_logits.float(), lm_labels)
            return lm_loss, encoder_output, encoder_output_mask

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}

        state_dict_[self._encoder_input_embedder_key] = self.encoder_input_embedder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        state_dict_[self._decoder_input_embedder_key] = self.decoder_input_embedder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        state_dict_[self._enc_dec_model_key] = self.enc_dec_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        state_dict_[self._lm_head_key] = self.lm_head.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.encoder_input_embedder.encoder_input_embedderload_state_dict(state_dict[self._encoder_input_embedder_key], strict=strict)
        self.decoder_input_embedder.load_state_dict(state_dict[self._decoder_input_embedder_key], strict=strict)
        self.enc_dec_model.load_state_dict(state_dict[self._enc_dec_model_key], strict=strict)
        self.lm_head.load_state_dict(state_dict[self._lm_head_key], strict=strict)
