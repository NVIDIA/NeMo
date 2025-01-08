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
