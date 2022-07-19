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

from omegaconf import OmegaConf, DictConfig
import torch

from nemo.collections.nlp.modules.common.megatron.language_model import Embedding
from nemo.collections.nlp.modules.common.megatron.megatron_decoders import get_decoder_model
from nemo.collections.nlp.modules.common.megatron.megatron_encoder_decoder import (
    MegatronTransformerEncoderDecoderModule,
)
from nemo.collections.nlp.modules.common.megatron.megatron_encoders import get_encoder_model
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    build_position_ids,
    init_method_normal,
    parallel_lm_logits,
    scaled_init_method_normal,
)

try:
    from apex.transformer import tensor_parallel
    from apex.transformer.enums import AttnMaskType, ModelType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False
    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()
    ModelType = ApexGuardDefaults()

__all__ = ["MegatronTokenLevelHead", "MegatronTokenLevelEncoderDecoderModule"]


class MegatronTokenLevelHead(MegatronModule):
    """Masked LM head for token-based encoder-decoder models (e.g., T5)

    Arguments:
        mpu_vocab_size: model parallel size of vocabulary.
        parallel_output: wether output logits being distributed or not.
    """

    def __init__(self, mpu_vocab_size, parallel_output):
        super(MegatronTokenLevelHead, self).__init__()

        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        self.bias.model_parallel = True
        self.bias.partition_dim = 0
        self.bias.stride = 1
        self.parallel_output = parallel_output

    def forward(self, hidden_states, word_embeddings_weight):
        output = parallel_lm_logits(hidden_states, word_embeddings_weight, self.parallel_output, bias=self.bias)
        return output


# TODO: add soft prompts as an Embedding sub-class


class MegatronTokenLevelEncoderDecoderModule(MegatronModule):
    """Token-based (input/output is tokens) encoder-decoder model (e.g. T5 Language model.)"""

    def __init__(
        self,
        encoder_cfg: DictConfig,
        decoder_cfg: DictConfig,
        vocab_size: int, # TODO: This should eventually go inside encoder_cfg and decoder_cfg when separate enc/dec tokenizers are supported.
        max_position_embeddings,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
        fp16_cross_entropy=False,
        use_cpu_initialization=False,
        precision=16,
        embedding_init_method_std=0.02,
        embedding_dropout=0.1,
        add_encoder=True,
        add_decoder=True,
    ):
        super(MegatronTokenLevelEncoderDecoderModule, self).__init__()

        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_cross_entropy = fp16_cross_entropy
        self.precision = precision
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        encoder_kv_channels, decoder_kv_channels = self._validate_config()

        # TODO: This should eventually be encoder and decoder embedding specific.
        add_position_embedding = encoder_cfg.get('position_embedding_type', 'learned_absolute') == 'learned_absolute'

        encoder, decoder = None, None
        if add_encoder:
            if pre_process:
                self.encoder_embedding = Embedding(
                    hidden_size=encoder_cfg.hidden_size,
                    vocab_size=vocab_size,
                    max_sequence_length=max_position_embeddings,
                    init_method=init_method_normal(embedding_init_method_std),
                    num_tokentypes=num_tokentypes,
                    use_cpu_initialization=use_cpu_initialization,
                    embedding_dropout_prob=embedding_dropout,
                    add_position_embedding=add_position_embedding,
                )
                self._encoder_embedding_key = "encoder_embedding"

            encoder = get_encoder_model(
                arch=encoder_cfg.arch,
                hidden_size=encoder_cfg.hidden_size,
                ffn_hidden_size=encoder_cfg.ffn_hidden_size,
                num_layers=encoder_cfg.num_layers,
                num_attention_heads=encoder_cfg.num_attention_heads,
                apply_query_key_layer_scaling=encoder_cfg.apply_query_key_layer_scaling,
                kv_channels=encoder_kv_channels,
                init_method=init_method_normal(encoder_cfg.init_method_std),
                scaled_init_method=scaled_init_method_normal(encoder_cfg.init_method_std, encoder_cfg.num_layers),
                encoder_attn_mask_type=AttnMaskType.padding,
                pre_process=pre_process,
                post_process=post_process,
                init_method_std=encoder_cfg.init_method_std,
                use_cpu_initialization=use_cpu_initialization,
                hidden_dropout=encoder_cfg.hidden_dropout,
                attention_dropout=encoder_cfg.attention_dropout,
                position_embedding_type=encoder_cfg.get('position_embedding_type', 'learned_absolute'),
                relative_attention_num_buckets=encoder_cfg.get('relative_attention_num_buckets', 32),
                relative_attention_max_distance=encoder_cfg.get('relative_attention_max_distance', 128),
                precision=precision,
                fp32_residual_connection=encoder_cfg.fp32_residual_connection,
                activations_checkpoint_method=encoder_cfg.activations_checkpoint_method,
                activations_checkpoint_num_layers=encoder_cfg.activations_checkpoint_num_layers,
                layernorm_epsilon=encoder_cfg.layernorm_epsilon,
                bias_activation_fusion=encoder_cfg.get('bias_activation_fusion', True),
                bias_dropout_add_fusion=encoder_cfg.get('bias_dropout_add_fusion', True),
                masked_softmax_fusion=encoder_cfg.get('masked_softmax_fusion', True),
                persist_layer_norm=encoder_cfg.get('persist_layer_norm', True),
                openai_gelu=encoder_cfg.get('openai_gelu', False),
                onnx_safe=encoder_cfg.get('onnx_safe', False),
                hidden_steps=encoder_cfg.get('hidden_steps', -1), 
                activation=encoder_cfg.get('activation', 'gelu'),
                bias=encoder_cfg.get('bias', True),
                normalization=encoder_cfg.get('normalization', 'layernorm'),
                transformer_block_type=encoder_cfg.get('transformer_block_type', 'pre_ln'),
                headscale=encoder_cfg.get('headscale', False),
                parent_model_type=ModelType.encoder_and_decoder,
                num_self_attention_per_cross_attention=encoder_cfg.get('num_self_attention_per_cross_attention', 1),
            )

        if add_decoder:
            # If this is the decoder first stage
            if pre_process:
                # If the encoder also lies on this rank (PP = 1), then just assign embeddings directly.
                if hasattr(self, 'encoder_embedding'):
                    self.decoder_embedding = self.encoder_embedding
                else:
                    # This is the case where PP > 1 and first decoder first stage.
                    # We initialize decoder embeddings, but set them to zero since we they're tied with the encoder embeddings.
                    # A later initialize_embedding call will synchronize the embeddings.
                    self.decoder_embedding = Embedding(
                        hidden_size=decoder_cfg.hidden_size,
                        vocab_size=vocab_size,
                        max_sequence_length=max_position_embeddings,
                        init_method=init_method_normal(embedding_init_method_std),
                        num_tokentypes=num_tokentypes,
                        use_cpu_initialization=use_cpu_initialization,
                        embedding_dropout_prob=embedding_dropout,
                        add_position_embedding=add_position_embedding,
                    )
                    self.decoder_embedding.zero_parameters()

                self._decoder_embedding_key = "decoder_embedding"

            decoder = get_decoder_model(
                arch=decoder_cfg.arch,
                hidden_size=decoder_cfg.hidden_size,
                ffn_hidden_size=decoder_cfg.ffn_hidden_size,
                num_layers=decoder_cfg.num_layers,
                num_attention_heads=decoder_cfg.num_attention_heads,
                apply_query_key_layer_scaling=decoder_cfg.apply_query_key_layer_scaling,
                kv_channels=decoder_kv_channels,
                init_method=init_method_normal(decoder_cfg.init_method_std),
                scaled_init_method=scaled_init_method_normal(decoder_cfg.init_method_std, decoder_cfg.num_layers),
                decoder_attn_mask_type=AttnMaskType.causal,
                pre_process=pre_process,
                post_process=post_process,
                init_method_std=decoder_cfg.init_method_std,
                use_cpu_initialization=use_cpu_initialization,
                hidden_dropout=decoder_cfg.hidden_dropout,
                attention_dropout=decoder_cfg.attention_dropout,
                position_embedding_type=decoder_cfg.get('position_embedding_type', 'learned_absolute'),
                relative_attention_num_buckets=decoder_cfg.get('relative_attention_num_buckets', 32),
                relative_attention_max_distance=decoder_cfg.get('relative_attention_max_distance', 128),
                precision=precision,
                fp32_residual_connection=decoder_cfg.fp32_residual_connection,
                activations_checkpoint_method=decoder_cfg.activations_checkpoint_method,
                activations_checkpoint_num_layers=decoder_cfg.activations_checkpoint_num_layers,
                layernorm_epsilon=decoder_cfg.layernorm_epsilon,
                bias_activation_fusion=decoder_cfg.get('bias_activation_fusion', True),
                bias_dropout_add_fusion=decoder_cfg.get('bias_dropout_add_fusion', True),
                masked_softmax_fusion=decoder_cfg.get('masked_softmax_fusion', True),
                persist_layer_norm=decoder_cfg.get('persist_layer_norm', True),
                openai_gelu=decoder_cfg.get('openai_gelu', False),
                onnx_safe=decoder_cfg.get('onnx_safe', False),
                hidden_steps=decoder_cfg.get('hidden_steps', -1), 
                activation=decoder_cfg.get('activation', 'gelu'),
                bias=decoder_cfg.get('bias', True),
                normalization=decoder_cfg.get('normalization', 'layernorm'),
                transformer_block_type=decoder_cfg.get('transformer_block_type', 'pre_ln'),
                headscale=decoder_cfg.get('headscale', False),
                parent_model_type=ModelType.encoder_and_decoder,
            )

        self.enc_dec_model = MegatronTransformerEncoderDecoderModule(
            encoder=encoder, decoder=decoder, hidden_steps=encoder_cfg.get('hidden_steps', -1),
        )
        self._enc_dec_model_key = "enc_dec_model"

        self.initialize_word_embeddings(
            init_method=init_method_normal(embedding_init_method_std), vocab_size=vocab_size, hidden_size=encoder_cfg.hidden_size
        )

        if add_decoder and post_process:
            self.tokens_head = MegatronTokenLevelHead(self.word_embeddings_weight().size(0), parallel_output)
            self._tokens_head_key = 'tokens_head'

    def _validate_kv_channels(self, cfg):
        kv_channels = cfg.kv_channels
        if cfg.kv_channels is None:
            assert (
                cfg.hidden_size % cfg.num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = cfg.hidden_size // cfg.num_attention_heads

        return kv_channels

    def _validate_enc_dec_hidden_size(self, encoder_cfg, decoder_cfg):
        if encoder_cfg.hidden_size != decoder_cfg.hidden_size:
            raise ValueError(f"Encoder and decoder hidden_size must be equal, but got encoder: {encoder_cfg.hidden_size} and decoder: {decoder_cfg.hidden_size}")

    # TODO: (sandeepsub) - this can be removed one we rebase to sequence parallel that contains the RPE refactor.
    def _validate_position_embedding_type(self, encoder_cfg, decoder_cfg):
        if encoder_cfg.get('position_embedding_type', 'learned_absolute') != decoder_cfg.get('position_embedding_type', 'learned_absolute'):
            raise ValueError(f"Encoder and decoder position_embedding_type must be the same, but got encoder: {encoder_cfg.position_embedding_type} and decoder: {decoder_cfg.position_embedding_type}")
        
        if encoder_cfg.get('position_embedding_type', 'learned_absolute') == 'relative':
            if encoder_cfg.relative_attention_num_buckets != decoder_cfg.relative_attention_num_buckets:
                raise ValueError(f"Encoder and decoder relative_attention_num_buckets must be equal, but got encoder: {encoder_cfg.relative_attention_num_buckets} and decoder: {decoder_cfg.relative_attention_num_buckets}")
            if encoder_cfg.relative_attention_max_distance != decoder_cfg.relative_attention_max_distance:
                raise ValueError(f"Encoder and decoder relative_attention_max_distance must be equal, but got encoder: {encoder_cfg.relative_attention_max_distance} and decoder: {decoder_cfg.relative_attention_max_distance}")

    def _validate_config(self):
        encoder_kv_channels = self._validate_kv_channels(self.encoder_cfg)
        decoder_kv_channels = self._validate_kv_channels(self.decoder_cfg)
        self._validate_enc_dec_hidden_size(self.encoder_cfg, self.decoder_cfg)
        self._validate_position_embedding_type(self.encoder_cfg, self.decoder_cfg)
        return encoder_kv_channels, decoder_kv_channels
        
    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None

        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        if self.add_encoder and self.add_decoder:
            assert (
                len(input_tensor) == 1
            ), 'input_tensor should only be length 1 for stage with both encoder and decoder'
            self.enc_dec_model.encoder.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            assert len(input_tensor) == 1, 'input_tensor should only be length 1 for stage with only encoder'
            self.enc_dec_model.encoder.set_input_tensor(input_tensor[0])
        elif self.add_decoder:
            if len(input_tensor) == 2:
                self.enc_dec_model.decoder.set_input_tensor(input_tensor[0])
                self.enc_dec_model.encoder_hidden_state = input_tensor[1]
            elif len(input_tensor) == 1:
                self.enc_dec_model.decoder.set_input_tensor(None)
                self.enc_dec_model.encoder_hidden_state = input_tensor[0]
            else:
                raise Exception('input_tensor must have either length 1 or 2')
        else:
            raise Exception('Stage must have at least either encoder or decoder')

    def forward(
        self,
        enc_input_ids,
        enc_attn_mask,
        dec_input_ids,
        dec_attn_mask,
        token_type_ids=None,
        labels=None,
        enc_hidden_states=None,
        enc_output_mask=None,
        output_enc_hidden_only=False,
        enc_input=None,
    ):
        """
        Return value is per token / per dimension (i.e., non collapsed loss value)
        """
        if enc_input is None:
            if self.pre_process and self.add_encoder:
                # encoder embeddings
                enc_position_ids = build_position_ids(enc_input_ids)
                enc_input = self.encoder_embedding(enc_input_ids, enc_position_ids, token_type_ids=token_type_ids)
            else:
                enc_input = None

        if output_enc_hidden_only:
            enc_output = self.enc_dec_model.encode(
                enc_input=enc_input, enc_attn_mask=enc_attn_mask, enc_layer_past=None, enc_get_key_value=False,
            )
            return enc_output
        else:
            if self.pre_process and self.add_decoder:
                dec_position_ids = build_position_ids(dec_input_ids)
                dec_input = self.decoder_embedding(dec_input_ids, dec_position_ids, token_type_ids=token_type_ids)
            else:
                # Note: This is when the decoder itself is split across PP ranks.
                dec_input = None

            output = self.enc_dec_model(
                enc_input=enc_input,
                enc_attn_mask=enc_attn_mask,
                dec_input=dec_input,
                dec_attn_mask=dec_attn_mask,
                enc_layer_past=None,
                enc_get_key_value=False,
                enc_output=None,
                dec_layer_past=None,
                dec_get_key_value=False,
            )

            if self.post_process and self.add_decoder:
                dec_output, enc_output = output
                # project decoder output to vocabulary-size dimensions
                token_logits = self.tokens_head(dec_output, self.word_embeddings_weight())

                if labels is not None:
                    # tensor_parallel.vocab_parallel_cross_entropy performs log_softmax and return log p(x_i|z) per token i
                    if self.fp16_cross_entropy:
                        assert token_logits.dtype == torch.half
                        tokens_loss = tensor_parallel.vocab_parallel_cross_entropy(token_logits, labels)
                    else:
                        tokens_loss = tensor_parallel.vocab_parallel_cross_entropy(token_logits.float(), labels)
                    return tokens_loss
                else:
                    return token_logits

            elif self.add_decoder and not self.add_encoder:
                decoder_output, _ = output
                return decoder_output
            else:
                encoder_output = output
                return encoder_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}

        state_dict_[self._encoder_embedding_key] = self.encoder_embedding.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        state_dict_[self._decoder_embedding_key] = self.decoder_embedding.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        state_dict_[self._enc_dec_model_key] = self.enc_dec_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        state_dict_[self._tokens_head_key] = self.tokens_head.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.encoder_embedding.encoder_embeddingload_state_dict(state_dict[self._encoder_embedding_key], strict=strict)
        self.decoder_embedding.load_state_dict(state_dict[self._decoder_embedding_key], strict=strict)
        self.enc_dec_model.load_state_dict(state_dict[self._enc_dec_model_key], strict=strict)
        self.tokens_head.load_state_dict(state_dict[self._tokens_head_key], strict=strict)
