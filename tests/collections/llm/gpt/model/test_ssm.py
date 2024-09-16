from nemo.collections.llm.gpt.model.ssm import (
    SSMConfig,
    gpt_data_step,
    ssm_forward_step,
)


def test_ssm_config():
    config = SSMConfig()
    assert config.fp16_lm_cross_entropy is False
    assert config.parallel_output is True
    assert config.share_embeddings_and_output_weights is False
    assert config.num_layers == 2
    assert config.mamba_ssm_ngroups == 8
    assert config.num_attention_heads == 1
    assert config.hybrid_attention_ratio == 0.0
    assert config.hybrid_mlp_ratio == 0.0
    assert config.hybrid_override_pattern is None
    assert config.post_process is True
    assert config.pre_process is True
    assert config.seq_length == 2048
    assert config.position_embedding_type == "none"
    assert config.rotary_percent == 1.0
    assert config.rotary_base == 10000
    assert config.seq_len_interpolation_factor is None
    assert config.apply_rope_fusion is True
    assert config.make_vocab_size_divisible_by == 128
    assert config.gated_linear_unit is False
    assert config.fp32_residual_connections is True
    assert config.normalization == "RMSNorm"
    assert config.add_bias_linear is False
    assert config.hidden_dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.layernorm_epsilon == 1e-5
    assert config.get_attention_mask_from_fusion is False
    assert config.forward_step_fn == ssm_forward_step
    assert config.data_step_fn == gpt_data_step