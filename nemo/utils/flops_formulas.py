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

from nemo.collections.common.parts.perf_metrics_utils import LLM_VOCAB_SIZE_MAP


def gpt3(mdl):
    """Model FLOPs for GPT3 family"""

    vocab_size = LLM_VOCAB_SIZE_MAP["gpt3"]

    return (
        24 * mdl.gbs * mdl.enc_seq_len * mdl.hs * mdl.hs + 4 * mdl.gbs * mdl.enc_seq_len * mdl.enc_seq_len * mdl.hs
    ) * (3 * mdl.layers) + (6 * mdl.gbs * mdl.enc_seq_len * mdl.hs * vocab_size)


def llama2(mdl):
    """Model FLOPs for llama2 family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["llama2"]

    return (
        mdl.gbs
        * mdl.enc_seq_len
        * mdl.layers
        * mdl.hs
        * mdl.hs
        * (
            12
            + (12 * mdl.query_groups / mdl.attention_heads)
            + (18 * mdl.ffn_hs / mdl.hs)
            + (12 * mdl.enc_seq_len / mdl.hs)
            + (6 * vocab_size / (mdl.layers * mdl.hs))
        )
    )


def llama3(mdl):
    """Model FLOPs for llama3 family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["llama3"]

    return (
        mdl.gbs
        * mdl.enc_seq_len
        * mdl.layers
        * mdl.hs
        * mdl.hs
        * (
            12
            + (12 * mdl.query_groups / mdl.attention_heads)
            + (18 * mdl.ffn_hs / mdl.hs)
            + (12 * mdl.enc_seq_len / mdl.hs)
            + (6 * vocab_size / (mdl.layers * mdl.hs))
        )
    )


def nemotron(mdl):
    """Model FLOPs for nemotron family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["nemotron"]

    return (
        mdl.gbs
        * mdl.enc_seq_len
        * mdl.layers
        * mdl.hs
        * mdl.hs
        * (
            12
            + (12 * mdl.query_groups / mdl.attention_heads)
            + (12 * mdl.ffn_hs / mdl.hs)
            + (12 * mdl.enc_seq_len / mdl.hs)
            + (6 * vocab_size / (mdl.layers * mdl.hs))
        )
    )


def mixtral(mdl):
    """Model FLOPs for mixtral family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["mixtral"]

    return (
        mdl.gbs
        * mdl.enc_seq_len
        * mdl.layers
        * mdl.hs
        * mdl.hs
        * (
            12
            + (12 * mdl.query_groups / mdl.attention_heads)
            + (18 * mdl.moe_router_topk * mdl.ffn_hs / mdl.hs)
            + (12 * mdl.enc_seq_len / mdl.hs)
            + (6 * vocab_size / (mdl.layers * mdl.hs))
        )
    )


def bert(mdl):
    """Model FLOPs for BERT family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["bert"]

    return (
        72
        * mdl.gbs
        * mdl.layers
        * mdl.enc_seq_len
        * mdl.hs
        * mdl.hs
        * (1 + (mdl.enc_seq_len / (6 * mdl.hs)) + (vocab_size / (12 * mdl.hs * mdl.layers)))
    )
