import pytest
import torch
from transformers import MllamaConfig
from transformers.models.mllama.configuration_mllama import MllamaTextConfig, MllamaVisionConfig

from nemo.collections.vlm import MLlamaConfig11BInstruct
from nemo.export.multimodal.converter import (
    convert_mllama_config,
    split_gate_weight,
    split_kv_weight,
    split_qkv_weight,
)


class TestMultimodalConverter:
    @pytest.fixture
    def model_config(self):
        # Create a simple test config
        config = type(
            'TestConfig',
            (),
            {'hidden_size': 128, 'num_attention_heads': 4, 'num_query_groups': 2, 'kv_channels': None},
        )()
        return config

    def test_split_qkv_weight(self, model_config):
        # Create a test QKV weight tensor
        batch_size = model_config.num_attention_heads + 2 * model_config.num_query_groups
        qkv_weight = torch.randn(
            batch_size, model_config.hidden_size // model_config.num_attention_heads, model_config.hidden_size
        )

        result = split_qkv_weight(qkv_weight, model_config)

        assert len(result) == 3
        assert result[0][0] == 'q_proj'
        assert result[1][0] == 'k_proj'
        assert result[2][0] == 'v_proj'

        # Check shapes
        assert result[0][1].shape == (
            model_config.num_attention_heads,
            model_config.hidden_size // model_config.num_attention_heads,
            model_config.hidden_size,
        )
        assert result[1][1].shape == (
            model_config.num_query_groups,
            model_config.hidden_size // model_config.num_attention_heads,
            model_config.hidden_size,
        )
        assert result[2][1].shape == (
            model_config.num_query_groups,
            model_config.hidden_size // model_config.num_attention_heads,
            model_config.hidden_size,
        )

    def test_split_kv_weight(self, model_config):
        # Create a test KV weight tensor
        batch_size = 2 * model_config.num_query_groups
        kv_weight = torch.randn(
            batch_size, model_config.hidden_size // model_config.num_attention_heads, model_config.hidden_size
        )

        result = split_kv_weight(kv_weight, model_config)

        assert len(result) == 2
        assert result[0][0] == 'k_proj'
        assert result[1][0] == 'v_proj'

        # Check shapes
        assert result[0][1].shape == (
            model_config.num_query_groups,
            model_config.hidden_size // model_config.num_attention_heads,
            model_config.hidden_size,
        )
        assert result[1][1].shape == (
            model_config.num_query_groups,
            model_config.hidden_size // model_config.num_attention_heads,
            model_config.hidden_size,
        )

    def test_split_gate_weight(self):
        # Create a test gate weight tensor
        gate_weight = torch.randn(200, 100)  # Example dimensions

        result = split_gate_weight(gate_weight)

        assert len(result) == 2
        assert result[0][0] == 'gate_proj'
        assert result[1][0] == 'up_proj'

        # Check shapes
        assert result[0][1].shape == (100, 100)
        assert result[1][1].shape == (100, 100)
