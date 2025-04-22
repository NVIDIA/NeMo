import pytest
import numpy as np
import torch
from nemo.export.trt_llm.converter.utils import (
    any_word_in_key,
    get_trt_llm_keyname,
    is_scaling_factor,
    get_scaling_factor_keys,
    split,
    generate_int8
)

def test_any_word_in_key():
    # Test positive cases
    assert any_word_in_key("model.layer1.attention.dense.weight", ["attention", "mlp"]) == True
    assert any_word_in_key("model.layer1.mlp.weight", ["attention", "mlp"]) == True
    
    # Test negative cases
    assert any_word_in_key("model.layer1.other.weight", ["attention", "mlp"]) == False
    assert any_word_in_key("", ["attention", "mlp"]) == False

def test_get_trt_llm_keyname():
    # Test final layernorm case
    assert get_trt_llm_keyname("final_layernorm.weight") == "transformer.ln_f.weight"
    
    # Test layer cases
    assert get_trt_llm_keyname("layers.1.attention.dense.weight") == "transformer.layers.1.attention.dense.weight"
    assert get_trt_llm_keyname("layers.2.mlp.linear_fc2.weight") == "transformer.layers.2.mlp.proj.weight"

def test_is_scaling_factor():
    assert is_scaling_factor("model.layer1.scale_fwd.weight") == True
    assert is_scaling_factor("model.layer1.weight") == False
    assert is_scaling_factor("") == False

def test_get_scaling_factor_keys():
    key = "layers.1.mlp.dense_h_to_4h.scale_fwd"
    keys, gate_keys = get_scaling_factor_keys(key)
    
    # Check main keys
    assert keys[0].endswith(".weights_scaling_factor")
    assert keys[1].endswith(".activation_scaling_factor")
    
    # Check gate keys
    assert gate_keys[0].endswith(".activation_scaling_factor")
    assert gate_keys[1].endswith(".weights_scaling_factor")

def test_split():
    # Test numpy array splitting
    arr = np.array([1, 2, 3, 4])
    assert np.array_equal(split(arr, tp_size=2, idx=0), np.array([1, 2]))
    assert np.array_equal(split(arr, tp_size=2, idx=1), np.array([3, 4]))
    
    # Test torch tensor splitting
    tensor = torch.tensor([1, 2, 3, 4])
    assert torch.equal(split(tensor, tp_size=2, idx=0), torch.tensor([1, 2]))
    assert torch.equal(split(tensor, tp_size=2, idx=1), torch.tensor([3, 4]))
    
    # Test no split case
    assert np.array_equal(split(arr, tp_size=1, idx=0), arr)

def test_generate_int8():
    # Create test weights and activation ranges
    weights = np.random.randn(4, 4).astype(np.float32)
    act_range = {
        "w": torch.tensor(2.0),
        "x": torch.tensor(3.0),
        "y": torch.tensor(4.0)
    }
    
    result = generate_int8(weights, act_range)
    
    # Check that all expected keys are present
    expected_keys = [
        "weight.int8", "weight.int8.col",
        "scale_x_orig_quant", "scale_w_quant_orig",
        "scale_w_quant_orig.col", "scale_y_accum_quant",
        "scale_y_accum_quant.col", "scale_y_quant_orig"
    ]
    assert all(key in result for key in expected_keys)
    
    # Check that int8 weights are in correct range
    assert np.all(result["weight.int8"] >= -127)
    assert np.all(result["weight.int8"] <= 127)
    assert np.all(result["weight.int8.col"] >= -127)
    assert np.all(result["weight.int8.col"] <= 127)


if __name__ == "__main__":
    pytest.main([__file__])