import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo.export.utils.utils import (
    get_example_inputs,
    get_model_device_type,
    is_nemo2_checkpoint,
    is_nemo_tarfile,
    prepare_directory_for_export,
    torch_dtype_from_precision,
    validate_fp8_network,
)


class TestUtils:
    @pytest.fixture
    def temp_dir(self):
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir)

    def test_is_nemo2_checkpoint(self, temp_dir):
        # Test with non-existent path
        assert not is_nemo2_checkpoint("/non/existent/path")

        # Test with directory without context folder
        os.makedirs(os.path.join(temp_dir, "no_context"))
        assert not is_nemo2_checkpoint(os.path.join(temp_dir, "no_context"))

        # Test with valid NeMo 2.0 checkpoint
        os.makedirs(os.path.join(temp_dir, "valid_ckpt", "context"))
        assert is_nemo2_checkpoint(os.path.join(temp_dir, "valid_ckpt"))

    def test_prepare_directory_for_export(self, temp_dir):
        # Test creating new directory
        model_dir = os.path.join(temp_dir, "new_dir")
        prepare_directory_for_export(model_dir, delete_existing_files=False)
        assert os.path.exists(model_dir)
        assert os.path.isdir(model_dir)

        # Test with existing empty directory
        prepare_directory_for_export(model_dir, delete_existing_files=False)
        assert os.path.exists(model_dir)

        # Test with existing non-empty directory
        with open(os.path.join(model_dir, "test.txt"), "w") as f:
            f.write("test")

        with pytest.raises(RuntimeError):
            prepare_directory_for_export(model_dir, delete_existing_files=False)

        # Test with delete_existing_files=True
        prepare_directory_for_export(model_dir, delete_existing_files=True)
        assert os.path.exists(model_dir)
        assert not os.path.exists(os.path.join(model_dir, "test.txt"))

        # Test with subdir
        prepare_directory_for_export(model_dir, delete_existing_files=False, subdir="subdir")
        assert os.path.exists(os.path.join(model_dir, "subdir"))

    def test_is_nemo_tarfile(self, temp_dir):
        # Test with non-existent file
        assert not is_nemo_tarfile("/non/existent/file.nemo")

        # Test with non-nemo file
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        assert not is_nemo_tarfile(test_file)

        # Test with .nemo file
        nemo_file = os.path.join(temp_dir, "test.nemo")
        with open(nemo_file, "w") as f:
            f.write("test")
        assert is_nemo_tarfile(nemo_file)

    def test_torch_dtype_from_precision(self):
        # Test with megatron_amp_O2=False
        assert torch_dtype_from_precision("bf16", megatron_amp_O2=False) == torch.float32

        # Test with different precision values
        assert torch_dtype_from_precision("bf16") == torch.bfloat16
        assert torch_dtype_from_precision("bf16-mixed") == torch.bfloat16
        assert torch_dtype_from_precision(16) == torch.float16
        assert torch_dtype_from_precision("16") == torch.float16
        assert torch_dtype_from_precision("16-mixed") == torch.float16
        assert torch_dtype_from_precision(32) == torch.float32
        assert torch_dtype_from_precision("32") == torch.float32
        assert torch_dtype_from_precision("32-true") == torch.float32

        # Test with invalid precision
        with pytest.raises(ValueError):
            torch_dtype_from_precision("invalid")

    def test_get_example_inputs(self):
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

        result = get_example_inputs(mock_tokenizer)

        # Verify tokenizer was called with correct arguments
        mock_tokenizer.assert_called_once_with(
            ["example query one", "example query two"],
            ["example passage one", "example passage two"],
            return_tensors="pt",
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["attention_mask"], torch.Tensor)
