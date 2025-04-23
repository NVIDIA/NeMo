from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nemo.deploy.nlp.hf_deployable import HuggingFaceLLMDeploy


@pytest.fixture
def mock_model():
    model = MagicMock(spec=AutoModelForCausalLM)
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    return model


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=AutoTokenizer)
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.batch_decode.return_value = ["Generated text"]
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    return tokenizer


@pytest.fixture
def mock_peft_model():
    with patch("nemo.deploy.nlp.hf_deployable.PeftModel") as mock:
        mock.from_pretrained.return_value = MagicMock()
        yield mock


class TestHuggingFaceLLMDeploy:

    def test_initialization_invalid_task(self):
        with pytest.raises(AssertionError):
            HuggingFaceLLMDeploy(hf_model_id_path="test/model", task="invalid-task")

    def test_initialization_no_model(self):
        with pytest.raises(ValueError):
            HuggingFaceLLMDeploy(task="text-generation")

    def test_triton_input_output_config(self):
        deployer = HuggingFaceLLMDeploy(model=MagicMock(), tokenizer=MagicMock(), task="text-generation")

        inputs = deployer.get_triton_input
        outputs = deployer.get_triton_output

        assert len(inputs) == 10  # Verify number of input tensors
        assert len(outputs) == 3  # Verify number of output tensors

        # Verify required input tensor names
        assert any(tensor.name == "prompts" for tensor in inputs)
        assert any(tensor.name == "max_length" for tensor in inputs)

        # Verify output tensor names
        assert any(tensor.name == "sentences" for tensor in outputs)
        assert any(tensor.name == "logits" for tensor in outputs)
        assert any(tensor.name == "scores" for tensor in outputs)
