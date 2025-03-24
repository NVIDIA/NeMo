import torch
from nemo.collections.llm.recipes.precision.mixed_precision import (
    bf16_mixed,
    fp16_mixed,
    bf16_with_fp8_mixed,
    fp16_with_fp8_mixed,
)


def test_bf16_mixed_config():
    config = bf16_mixed()
    assert config.precision == "bf16-mixed"
    assert config.params_dtype == torch.bfloat16
    assert config.pipeline_dtype == torch.bfloat16
    assert config.autocast_enabled is False
    assert config.grad_reduce_in_fp32 is True


def test_fp16_mixed_config():
    config = fp16_mixed()
    assert config.precision == "16-mixed"
    assert config.params_dtype == torch.half
    assert config.pipeline_dtype == torch.half
    assert config.autocast_enabled is False
    assert config.grad_reduce_in_fp32 is False


def test_bf16_with_fp8_mixed_config():
    config = bf16_with_fp8_mixed()
    # Check base bf16 settings
    assert config.precision == "bf16-mixed"
    assert config.params_dtype == torch.bfloat16
    assert config.pipeline_dtype == torch.bfloat16

    # Check FP8 specific settings
    assert config.fp8 == "hybrid"
    assert config.fp8_margin == 0
    assert config.fp8_amax_history_len == 1024
    assert config.fp8_amax_compute_algo == "max"
    assert config.fp8_params is True


def test_fp16_with_fp8_mixed_config():
    config = fp16_with_fp8_mixed()
    # Check base fp16 settings
    assert config.precision == "16-mixed"
    assert config.params_dtype == torch.half
    assert config.pipeline_dtype == torch.half

    # Check FP8 specific settings
    assert config.fp8 == "hybrid"
    assert config.fp8_margin == 0
    assert config.fp8_amax_history_len == 1024
    assert config.fp8_amax_compute_algo == "max"
    assert config.fp8_params is True