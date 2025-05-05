# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from transformers import GPT2LMHeadModel

from nemo.collections.nlp.modules.common.gpt_module import GPTModule
from nemo.core.classes import typecheck

__all__ = ['GPT2Encoder']


class GPT2Encoder(GPT2LMHeadModel, GPTModule):
    """
    Wraps around the Huggingface transformers implementation repository for easy use within NeMo.
    """

    @typecheck()
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
        past_key_values=None,
        use_cache=False,
        position_ids=None,
        max_length=128,
    ):
        res = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=use_cache,
        )

        return res if not return_dict else res
