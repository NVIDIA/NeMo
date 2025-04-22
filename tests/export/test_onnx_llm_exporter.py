import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo.export.onnx_llm_exporter import OnnxLLMExporter


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, inputs):
        return self.linear(inputs['input_ids'])


class TestOnnxLLMExporter:
    @pytest.fixture
    def temp_dir(self, tmp_path):
        return str(tmp_path / "onnx_model")

    @pytest.fixture
    def dummy_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.save_pretrained = MagicMock()
        return tokenizer

    @pytest.fixture
    def dummy_model(self):
        return DummyModel()

    def test_init_with_model_and_tokenizer(self, temp_dir, dummy_model, dummy_tokenizer):
        exporter = OnnxLLMExporter(
            onnx_model_dir=temp_dir, model=dummy_model, tokenizer=dummy_tokenizer, load_runtime=False
        )
        assert exporter.model == dummy_model
        assert exporter.tokenizer == dummy_tokenizer
        assert exporter.onnx_model_dir == temp_dir

    def test_init_with_model_and_model_path_raises_error(self, temp_dir, dummy_model):
        with pytest.raises(ValueError, match="A model was also passed but it will be overridden"):
            OnnxLLMExporter(
                onnx_model_dir=temp_dir, model=dummy_model, model_name_or_path="some/path", load_runtime=False
            )
