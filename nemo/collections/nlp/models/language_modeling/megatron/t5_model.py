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
from megatron import get_args

from nemo.collections.nlp.modules.common.megatron.language_model import get_language_model, parallel_lm_logits
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.transformer import LayerNorm
from nemo.collections.nlp.modules.common.megatron.utils import init_method_normal, scaled_init_method_normal


def t5_extended_attention_mask(attention_mask_list):
    def attn_mask_postprocess(attn_mask):
        # [b, 1, s, s]
        extended_attention_mask = attn_mask.unsqueeze(1)
        return extended_attention_mask

    return [attn_mask_postprocess(attn_mask) for attn_mask in attention_mask_list]


def t5_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids


class T5LMHead(MegatronModule):
    """Masked LM head for T5

    Arguments:
        mpu_vocab_size: model parallel size of vocabulary.
        hidden_size: hidden size
        init_method: init method for weight initialization
        layernorm_epsilon: tolerance for layer norm divisions
        parallel_output: wether output logits being distributed or not.
    """

    def __init__(self, mpu_vocab_size, parallel_output):
        super(T5LMHead, self).__init__()

        args = get_args()

        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        self.bias.model_parallel = True
        self.bias.partition_dim = 0
        self.bias.stride = 1
        self.parallel_output = parallel_output

    def forward(self, hidden_states, word_embeddings_weight):
        output = parallel_lm_logits(hidden_states, word_embeddings_weight, self.parallel_output, bias=self.bias)
        return output


class T5Model(MegatronModule):
    """T5 Language model."""

    def __init__(self, num_tokentypes=0, parallel_output=True):
        super(T5Model, self).__init__()
        args = get_args()

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            add_decoder=True,
            encoder_attn_mask_type=AttnMaskType.padding,
            init_method=init_method,
            scaled_init_method=scaled_init_method,
        )

        self.lm_head = T5LMHead(self.language_model.embedding.word_embeddings.weight.size(0), parallel_output)
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
    ):

        # Converting the attention masks to proper parameter settings
        encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask = t5_extended_attention_mask(
            [encoder_attn_mask, decoder_attn_mask, encoder_decoder_attn_mask]
        )

        encoder_position_ids = t5_position_ids(encoder_input_ids)
        decoder_position_ids = t5_position_ids(decoder_input_ids)

        lm_output = self.language_model(
            encoder_input_ids,
            encoder_position_ids,
            encoder_attn_mask,
            decoder_input_ids,
            decoder_position_ids,
            decoder_attn_mask,
            encoder_decoder_attn_mask,
            tokentype_ids=tokentype_ids,
            enc_hidden_states=enc_hidden_states,
        )

        decoder_output, encoder_output = lm_output

        # Output.
        lm_logits = self.lm_head(decoder_output, self.language_model.embedding.word_embeddings.weight)

        if lm_labels is None:
            return lm_logits, encoder_output
        else:
            if self.fp16_lm_cross_entropy:
                assert lm_logits.dtype == torch.half
                lm_loss = tensor_parallel.vocab_parallel_cross_entropy(lm_logits, lm_labels)
            else:
                lm_loss = tensor_parallel.vocab_parallel_cross_entropy(lm_logits.float(), lm_labels)
            return lm_loss, encoder_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] = self.language_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        state_dict_[self._lm_head_key] = self.lm_head.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(state_dict[self._language_model_key], strict=strict)
        self.lm_head.load_state_dict(state_dict[self._lm_head_key], strict=strict)
