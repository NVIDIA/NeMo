# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock, patch

import torch

from nemo.collections.llm.gpt.model.llama_embedding import get_nv_embedding_layer_spec
from nemo.collections.llm.gpt.model.reranker import (
    Llama32Reranker1BConfig,
    Llama32Reranker500MConfig,
    ReRankerBaseConfig,
    ReRankerLoss,
    ReRankerModel,
)


def test_reranker_base_config():
    config = ReRankerBaseConfig()
    assert config.truncation_method == 'right'
    assert config.num_hard_negatives == 4
    assert config.ce_loss_scale == 50
    assert config.label_smoothing == 0.0
    assert config.in_batch_negatives is False
    assert config.negative_sample_strategy == 'first'
    assert config.add_bos is True
    assert config.add_eos is False
    assert config.pool_type == 'avg'
    assert config.temperature == 1.0


def test_llama32_reranker_1b_config():
    config = Llama32Reranker1BConfig()
    # Test inherited Llama32Config1B properties
    assert config.num_layers == 16
    assert config.hidden_size == 2048
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 8
    assert config.ffn_hidden_size == 8192

    # Test ReRankerBaseConfig properties
    assert config.truncation_method == 'right'
    assert config.num_hard_negatives == 4
    assert config.ce_loss_scale == 50
    assert config.label_smoothing == 0.0
    assert config.in_batch_negatives is False
    assert config.negative_sample_strategy == 'first'
    assert config.add_bos is True
    assert config.add_eos is False
    assert config.pool_type == 'avg'
    assert config.temperature == 1.0

    # Test specific properties
    assert config.transformer_layer_spec == get_nv_embedding_layer_spec
    assert config.forward_step_fn.__name__ == 'reranker_forward_step'
    assert config.data_step_fn.__name__ == 'reranker_data_step'


def test_llama32_reranker_500m_config():
    config = Llama32Reranker500MConfig()
    # Test inherited properties
    assert config.hidden_size == 2048
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 8
    assert config.ffn_hidden_size == 8192

    # Test ReRankerBaseConfig properties
    assert config.truncation_method == 'right'
    assert config.num_hard_negatives == 4
    assert config.ce_loss_scale == 50
    assert config.label_smoothing == 0.0
    assert config.in_batch_negatives is False
    assert config.negative_sample_strategy == 'first'
    assert config.add_bos is True
    assert config.add_eos is False
    assert config.pool_type == 'avg'
    assert config.temperature == 1.0


@patch('torch.distributed.all_reduce')
@patch('torch.distributed.get_world_size')
@patch('megatron.core.parallel_state.get_data_parallel_group')
@patch('megatron.core.parallel_state.get_context_parallel_world_size')
def test_reranker_loss(mock_cp_size, mock_dp_group, mock_world_size, mock_all_reduce):
    # Mock distributed environment
    mock_cp_size.return_value = 1
    mock_dp_group.return_value = MagicMock()
    mock_world_size.return_value = 1
    mock_all_reduce.return_value = None

    # Test initialization
    loss_fn = ReRankerLoss(validation_step=False, val_drop_last=True, num_hard_negatives=2, label_smoothing=0.1)
    assert loss_fn.validation_step is False
    assert loss_fn.val_drop_last is True
    assert loss_fn.num_hard_negatives == 2
    assert loss_fn.cross_entropy_loss.label_smoothing == 0.1

    # Test forward pass
    batch_size = 4
    num_hard_negatives = 2
    num_tensors_per_example = 1 + num_hard_negatives

    # Create dummy input
    forward_out = torch.randn(batch_size * num_tensors_per_example, device='cpu')
    batch = {}

    # Test forward pass
    loss, metrics = loss_fn.forward(batch, forward_out)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(metrics, dict)
    assert 'avg' in metrics
    assert metrics['avg'].shape == (1,)

    # Verify distributed calls
    mock_cp_size.assert_called_once()
    mock_dp_group.assert_called()
    mock_world_size.assert_called_once()
    mock_all_reduce.assert_called_once()


def test_reranker_model_pooling():
    # Test different pooling strategies
    config = Llama32Reranker1BConfig()
    model = ReRankerModel(config)

    batch_size = 2
    seq_length = 10
    hidden_size = config.hidden_size

    # Create dummy input
    last_hidden_states = torch.randn(seq_length, batch_size, hidden_size, device='cpu')
    attention_mask = torch.ones(batch_size, seq_length, device='cpu')

    # Test average pooling
    config.pool_type = 'avg'
    pooled = model.pool(last_hidden_states, attention_mask)
    assert pooled.shape == (batch_size, hidden_size)

    # Test weighted average pooling
    config.pool_type = 'weighted_avg'
    pooled = model.pool(last_hidden_states, attention_mask)
    assert pooled.shape == (batch_size, hidden_size)

    # Test CLS pooling
    config.pool_type = 'cls'
    pooled = model.pool(last_hidden_states, attention_mask)
    assert pooled.shape == (batch_size, hidden_size)

    # Test last token pooling
    config.pool_type = 'last'
    pooled = model.pool(last_hidden_states, attention_mask)
    assert pooled.shape == (batch_size, hidden_size)
