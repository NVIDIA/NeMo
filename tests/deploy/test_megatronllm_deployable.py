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

import pytest
from megatron.core.inference.common_inference_params import CommonInferenceParams

from nemo.deploy.nlp.megatronllm_deployable import MegatronLLMDeployableNemo2


@pytest.fixture
def mock_engine_and_tokenizer():
    """Fixture to mock the engine and tokenizer needed for testing."""
    mock_engine = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenizer.tokenizer = MagicMock()
    mock_tokenizer.tokenizer.tokenizer.chat_template = "{{messages}}"
    mock_tokenizer.tokenizer.tokenizer.bos_token = "<bos>"
    mock_tokenizer.tokenizer.tokenizer.eos_token = "<eos>"
    
    return mock_engine, mock_model, mock_tokenizer


@pytest.fixture
def deployable(mock_engine_and_tokenizer):
    """Fixture to create a deployable instance with mocked dependencies."""
    mock_engine, mock_model, mock_tokenizer = mock_engine_and_tokenizer
    
    # Patch the __init__ method to avoid file loading
    with patch.object(MegatronLLMDeployableNemo2, '__init__', return_value=None):
        deployable = MegatronLLMDeployableNemo2()
        
        # Set required attributes manually
        deployable.mcore_engine = mock_engine
        deployable.inference_wrapped_model = mock_model
        deployable.mcore_tokenizer = mock_tokenizer
        deployable.nemo_checkpoint_filepath = "dummy.nemo"
        deployable.max_batch_size = 32
        deployable.enable_cuda_graphs = True
        
        yield deployable


@pytest.mark.run_only_on("GPU")
def test_initialization(deployable):
    """Test initialization of the deployable class."""
    assert deployable.nemo_checkpoint_filepath == "dummy.nemo"
    assert deployable.max_batch_size == 32
    assert deployable.enable_cuda_graphs is True


@pytest.mark.run_only_on("GPU")
def test_generate_without_cuda_graphs(deployable):
    """Test text generation without CUDA graphs."""
    # Temporarily disable CUDA graphs
    deployable.enable_cuda_graphs = False
    
    prompts = ["Hello", "World"]
    inference_params = CommonInferenceParams(
        temperature=1.0,
        top_k=1,
        top_p=0.0,
        num_tokens_to_generate=256,
        return_log_probs=False,
    )

    # Mock the generate method
    with patch.object(deployable.mcore_engine, 'generate') as mock_generate:
        mock_result = MagicMock()
        mock_result.generated_text = "Generated text"
        mock_generate.return_value = [mock_result, mock_result]

        results = deployable.generate(prompts, inference_params)
        assert len(results) == 2
        mock_generate.assert_called_once_with(prompts=prompts, add_BOS=False, common_inference_params=inference_params)


@pytest.mark.run_only_on("GPU")
def test_generate_with_cuda_graphs(deployable):
    """Test text generation with CUDA graphs enabled."""
    # Ensure CUDA graphs is enabled
    deployable.enable_cuda_graphs = True
    deployable.max_batch_size = 4
    
    prompts = ["Hello", "World"]
    inference_params = CommonInferenceParams(
        temperature=1.0,
        top_k=1,
        top_p=0.0,
        num_tokens_to_generate=256,
        return_log_probs=False,
    )

    # Mock the generate method
    with patch.object(deployable.mcore_engine, 'generate') as mock_generate:
        mock_result1 = MagicMock()
        mock_result1.generated_text = "Generated text 1"
        mock_result2 = MagicMock()
        mock_result2.generated_text = "Generated text 2"
        mock_result_pad = MagicMock()
        mock_result_pad.generated_text = "Padding text"
        mock_generate.return_value = [mock_result1, mock_result2, mock_result_pad, mock_result_pad]

        results = deployable.generate(prompts, inference_params)
        
        # Should only return the actual results, not the padding
        assert len(results) == 2
        
        # Check that the padding was applied in the call
        called_args = mock_generate.call_args[1]
        assert len(called_args['prompts']) == 4  # Should pad to max_batch_size
        assert called_args['prompts'][:2] == prompts  # Original prompts should be first
        assert called_args['add_BOS'] is False
        assert called_args['common_inference_params'] == inference_params


@pytest.mark.run_only_on("GPU")
def test_apply_chat_template(deployable):
    """Test chat template application."""
    messages = [{"role": "user", "content": "Hello"}]
    
    # Set up jinja2 mock
    from jinja2 import Template
    template_mock = MagicMock()
    template_mock.render.return_value = "Rendered template with Hello"
    
    with patch('nemo.deploy.nlp.megatronllm_deployable.Template', return_value=template_mock):
        template = deployable.apply_chat_template(messages)
        assert template == "Rendered template with Hello"
        template_mock.render.assert_called_once()


@pytest.mark.run_only_on("GPU")
def test_remove_eos_token(deployable):
    """Test EOS token removal."""
    texts = ["Hello<eos>", "World", "Test<eos>"]
    cleaned_texts = deployable.remove_eos_token(texts)
    assert cleaned_texts == ["Hello", "World", "Test"]


@pytest.mark.run_only_on("GPU")
def test_str_to_dict(deployable):
    """Test string to dictionary conversion."""
    json_str = '{"key": "value"}'
    result = deployable.str_to_dict(json_str)
    assert isinstance(result, dict)
    assert result["key"] == "value"


@pytest.mark.run_only_on("GPU")
def test_triton_input_output(deployable):
    """Test Triton input and output tensor definitions."""
    # Mock the Tensor class from pytriton.model_config
    with patch('nemo.deploy.nlp.megatronllm_deployable.Tensor') as mock_tensor:
        # Set up mock to return itself for testing
        mock_tensor.side_effect = lambda name, shape, dtype, optional=False: MagicMock(name=name, shape=shape, dtype=dtype, optional=optional)
        
        inputs = deployable.get_triton_input
        outputs = deployable.get_triton_output
        
        # Extract mock calls to see what was created
        input_calls = mock_tensor.call_args_list[:9]  # First 9 calls are for inputs
        output_calls = mock_tensor.call_args_list[9:]  # Rest are for outputs
        
        # Check inputs (simplified to just check count and first param names)
        assert len(input_calls) == 9
        input_names = [call[1]['name'] for call in input_calls]
        assert "prompts" in input_names
        assert "max_length" in input_names
        assert "max_batch_size" in input_names
        assert "top_k" in input_names
        assert "top_p" in input_names
        assert "temperature" in input_names
        assert "random_seed" in input_names
        assert "compute_logprob" in input_names
        assert "apply_chat_template" in input_names
        
        # Check outputs
        assert len(output_calls) == 2
        output_names = [call[1]['name'] for call in output_calls]
        assert "sentences" in output_names
        assert "log_probs" in output_names
