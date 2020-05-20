# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
# =============================================================================

import math

import torch
from torch import nn

from nemo import logging
from nemo.collections.nlp.utils.functional_utils import gelu

__all__ = []


try:
    from apex.normalization import FusedLayerNorm

    # Try to use FusedLayerNorm from Apex - this will trigger an error.
    _ = FusedLayerNorm(8, eps=1e-5)

except Exception as e:
    logging.warning("Unable to import FusedLayerNorm  from APEX. Using regular LayerNorm instead.")
    from torch.nn import LayerNorm as FusedLayerNorm


class FixedPositionalEncoding(nn.Module):
    """
    Fixed positional encoding (embedding layer) from sine and cosine functions
    of different frequencies according to https://arxiv.org/abs/1706.03762

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        max_sequence_length: maximum allowed length of the input sequence
    """

    def __init__(self, hidden_size, max_sequence_length=512):
        super().__init__()

        pos_enc = torch.zeros(max_sequence_length, hidden_size)
        position = torch.arange(0.0, max_sequence_length).unsqueeze(1)
        coef = -math.log(10000.0) / hidden_size
        div_term = torch.exp(coef * torch.arange(0.0, hidden_size, 2))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc.div_(math.sqrt(hidden_size))
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, position_ids):
        return torch.embedding(self.pos_enc, position_ids)


class TransformerEmbedding(nn.Module):
    """
    Embedding from token and position embeddings.
    Optionally add token_type embedding (e.g. type of the sentence in BERT).

    Args:
        vocab_size: size of the vocabulary
        hidden_size: size of the embeddings in the model, also known as d_model
        max_sequence_length: maximum allowed length of the input sequence
        num_token_types: number of different token types
            (e.g. tokens of sentence A and tokens of sentence B in BERT)
        embedding_dropout: probability of dropout applied to embeddings
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_sequence_length=512,
        num_token_types=2,
        embedding_dropout=0.0,
        learn_positional_encodings=False,
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if learn_positional_encodings:
            self.position_embedding = nn.Embedding(max_sequence_length, hidden_size)
        else:
            self.position_embedding = FixedPositionalEncoding(hidden_size, max_sequence_length)
        self.token_type_embedding = nn.Embedding(num_token_types, hidden_size)
        self.layer_norm = FusedLayerNorm(hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(embedding_dropout)

    def forward(self, input_ids, token_type_ids=None, start_pos=0):
        seq_length = input_ids.size(1)
        if seq_length > self.max_sequence_length:
            raise ValueError("Input sequence is longer than maximum allowed sequence length for positional encoding")
        position_ids = torch.arange(
            start=start_pos, end=start_pos + seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings

        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embedding(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention layer.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            whole layer, but before layer normalization
    """

    def __init__(self, hidden_size, num_attention_heads, attn_score_dropout=0.0, attn_layer_dropout=0.0):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number "
                "of attention heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attn_head_size = int(hidden_size / num_attention_heads)
        self.attn_scale = math.sqrt(math.sqrt(self.attn_head_size))

        self.query_net = nn.Linear(hidden_size, hidden_size)
        self.key_net = nn.Linear(hidden_size, hidden_size)
        self.value_net = nn.Linear(hidden_size, hidden_size)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_score_dropout)
        self.layer_dropout = nn.Dropout(attn_layer_dropout)
        self.layer_norm = FusedLayerNorm(hidden_size, eps=1e-5)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attn_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, queries, keys, values, attention_mask):

        # attention_mask is needed to hide the tokens which correspond to [PAD]
        # in the case of BERT, or to hide the future tokens in the case of
        # vanilla language modeling and translation
        query = self.query_net(queries)
        key = self.key_net(keys)
        value = self.value_net(values)
        query = self.transpose_for_scores(query) / self.attn_scale
        key = self.transpose_for_scores(key) / self.attn_scale
        value = self.transpose_for_scores(value)

        # for numerical stability we pre-divide query and key by sqrt(sqrt(d))
        # and perform attention probs computation in float32
        attention_scores = torch.matmul(query, key.transpose(-1, -2)).float()
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.float()
        attention_probs = torch.softmax(attention_scores, dim=-1).to(key.dtype)
        attention_probs = self.attn_dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(*new_context_shape)

        # output projection
        output_states = self.out_projection(context)
        output_states = self.layer_dropout(output_states)
        output_states = self.layer_norm(queries + output_states)

        return output_states


class LightweightConv1d(nn.Module):
    """
    Lightweight convolution layer from https://arxiv.org/abs/1901.10430

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        num_heads: number of heads in lightweight convolution
        kernel_size: convolution kernel size
        conv_weight_dropout: probability of dropout applied to the convolution
            kernel (strictly speaking, DropConnect)
        conv_layer_dropout: probability of dropout applied to the output of the
            whole layer, but before layer normalization
    """

    def __init__(self, hidden_size, num_attention_heads, kernel_size, conv_weight_dropout=0.0, conv_layer_dropout=0.0):
        super().__init__()
        self.num_heads = num_attention_heads
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.Tensor(num_attention_heads, 1, kernel_size))
        self.in_projection = nn.Linear(hidden_size, hidden_size)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

        self.conv_weight_dropout = nn.Dropout(conv_weight_dropout)
        self.conv_layer_dropout = nn.Dropout(conv_layer_dropout)
        self.layer_norm = FusedLayerNorm(hidden_size, eps=1e-5)

    def forward(self, hidden_states, attention_mask):
        batch_size, seq_len, hidden_size = hidden_states.size()
        output_states = self.in_projection(hidden_states)
        output_states = output_states.permute(0, 2, 1)

        weight = torch.softmax(self.weight, dim=-1)
        weight = self.conv_weight_dropout(weight)

        if attention_mask:
            pivot = self.kernel_size // 2 + 1
            weight[:, :, pivot:] = 0

        output_states = output_states.contiguous().view(-1, self.num_heads, seq_len)
        output_states = torch.conv1d(output_states, weight, padding=self.kernel_size // 2, groups=self.num_heads)
        output_states = output_states.view(batch_size, hidden_size, seq_len)
        output_states = output_states.permute(0, 2, 1)

        # output projection
        output_states = self.out_projection(output_states)
        output_states = self.conv_layer_dropout(output_states)
        output_states = self.layer_norm(hidden_states + output_states)

        return output_states


class TwoStreamSelfAttention(nn.Module):
    """
    Two-Stream Self-Attention layer from https://arxiv.org/abs/1906.08237

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            whole layer, but before layer normalization
    """

    def __init__(self, hidden_size, num_attention_heads, attn_score_dropout=0.0, attn_layer_dropout=0.0):
        super().__init__()
        self.query_stream = MultiHeadAttention(
            hidden_size, num_attention_heads, attn_score_dropout, attn_layer_dropout
        )
        self.content_stream = MultiHeadAttention(
            hidden_size, num_attention_heads, attn_score_dropout, attn_layer_dropout
        )

    def forward(self, query_states, content_states, query_attention_mask, content_attention_mask):
        output_query_states = self.query_stream(query_states, content_states, content_states, query_attention_mask)
        output_content_states = self.content_stream(
            query_states, content_states, content_states, content_attention_mask
        )
        return output_query_states, output_content_states


class PositionWiseFF(nn.Module):
    """
    Position-wise feed-forward network of Transformer block.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
        ffn_dropout: probability of dropout applied to net output
        hidden_act: activation function used between two linear layers
    """

    def __init__(self, hidden_size, inner_size, ffn_dropout=0.0, hidden_act="relu"):
        super().__init__()
        self.dense_in = nn.Linear(hidden_size, inner_size)
        self.dense_out = nn.Linear(inner_size, hidden_size)
        self.layer_dropout = nn.Dropout(ffn_dropout)
        self.layer_norm = FusedLayerNorm(hidden_size, eps=1e-5)
        ACT2FN = {"gelu": gelu, "relu": torch.relu}
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states):
        output_states = self.dense_in(hidden_states)
        output_states = self.act_fn(output_states)
        output_states = self.dense_out(output_states)
        output_states = self.layer_dropout(output_states)
        output_states = self.layer_norm(hidden_states + output_states)
        return output_states
