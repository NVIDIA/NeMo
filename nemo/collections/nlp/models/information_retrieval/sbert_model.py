# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import torch
import torch.nn.functional as F

from torch import Tensor, nn
from nemo.collections.nlp.models.language_modeling.megatron.bert_model import MCoreBertModel
from nemo.collections.nlp.models.language_modeling.megatron.bert_model import BertModel, bert_extended_attention_mask
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    build_position_ids,
)

try:
    from megatron.core import ModelParallelConfig, parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False


class BertEmbeddingHead(nn.Module):
    """Performs mean pooling on the token embeddings.
    """

    def __init__(
        self,
        word_embedding_dimension: int,
        pooling_mode_mean_tokens: bool = True,

    ):
        super(BertEmbeddingHead, self).__init__()

        self.config_keys = [
            "word_embedding_dimension",
            "pooling_mode_mean_tokens",
        ]
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens

    def forward(self, token_embeddings: Tensor, attention_mask: Tensor):

        token_embeddings = token_embeddings.permute(1, 0, 2)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)

        sum_mask = torch.clamp(sum_mask, min=1e-9)

        output_vector = sum_embeddings / sum_mask

        output_vector = F.normalize(output_vector, p=2, dim=1)

        return output_vector

    def __repr__(self):
        return "Pooling({}) and Normalize".format(self.get_config_dict())
    
    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}


class MCoreSBertModel(MCoreBertModel):
    def __init__(self, add_embedding_head=True, *args, **kwargs):

        super(MCoreSBertModel, self).__init__(*args, **kwargs)
        self.add_embedding_head = add_embedding_head
        
        if self.add_embedding_head:
            self.post_process = False
            self.binary_head = None
            self.lm_head = None
            self.output_layer = None
            self.encoder.final_layernorm = None
            self.encoder.post_process = False

        if self.add_embedding_head:
            self.embedding_head = BertEmbeddingHead(
                word_embedding_dimension=self.config.hidden_size,
                pooling_mode_mean_tokens=True,
            )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        tokentype_ids: Tensor = None,
        lm_labels: Tensor = None,
        inference_params=None,
    ):
        """Forward function of BERT model

        Forward function of the BERT Model This function passes the input tensors
        through the embedding layer, and then the encoder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        extended_attention_mask = self.bert_extended_attention_mask(attention_mask)

        if parallel_state.is_pipeline_first_stage():
            input_ids = input_ids
            position_ids = self.bert_position_ids(input_ids)
        else:
            position_ids = None
            input_ids = None

        # Encoder embedding.
        if self.pre_process:
            encoder_input = self.embedding(
                input_ids=input_ids, position_ids=position_ids, tokentype_ids=tokentype_ids
            )
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            encoder_input = None

        # Rotary positional embeddings (Why not move this into BERT/GPTEmberdding ?)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.encoder, encoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run decoder.
        hidden_states = self.encoder(
            hidden_states=encoder_input,
            attention_mask=extended_attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
        )
        if self.add_embedding_head:

            embeddings_out = self.embedding_head(hidden_states, attention_mask)
            return embeddings_out

        if self.add_pooler:
            pooled_output = self.pooler(hidden_states, 0)

        if self.return_embeddings:
            embeddings = torch.transpose(hidden_states, 0, 1)
            masks = torch.sum(attention_mask, dim=1)
            # Collect masked embeddings.
            output = torch.zeros(
                size=(embeddings.shape[0], embeddings.shape[2]),
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
            for i, (embedding, mask) in enumerate(zip(embeddings, masks)):
                output[i, :] = torch.mean(embedding[1 : mask - 1], dim=0)
            return output

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        logits = self.lm_head(hidden_states=hidden_states, word_embeddings_weight=output_weight)

        binary_logits = None
        if self.binary_head is not None:
            binary_logits = self.binary_head(pooled_output)

        if lm_labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous(), binary_logits

        loss = self.compute_language_model_loss (lm_labels, logits)

        return loss, binary_logits


class SBertModel(BertModel):
    """
    Bert Language model.
    Model returns [seq, batch, hidden] shape
    """

    def __init__(self, add_embedding_head=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_embedding_head = add_embedding_head
        if self.add_embedding_head:
            self.embedding_head = BertEmbeddingHead(
                word_embedding_dimension=self.config.hidden_size,
                pooling_mode_mean_tokens=True,
            )

    def forward(
        self,
        bert_model_input,
        attention_mask,
        token_type_ids=None,
        lm_labels=None,
        checkpoint_activations_all_layers=None,
    ):

        extended_attention_mask = bert_extended_attention_mask(attention_mask)

        if parallel_state.is_pipeline_first_stage():
            input_ids = bert_model_input
            position_ids = build_position_ids(input_ids)
        else:
            position_ids = None
            input_ids = None

        lm_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            token_type_ids=token_type_ids,
            checkpoint_activations_all_layers=checkpoint_activations_all_layers,
        )

        if self.add_lm_head and self.post_process and self.add_binary_head:

            lm_output, _ = lm_output

        if self.add_embedding_head:

            embeddings_out = self.embedding_head(lm_output[0], attention_mask)
            return embeddings_out
