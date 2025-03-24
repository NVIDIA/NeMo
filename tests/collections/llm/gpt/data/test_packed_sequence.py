import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from nemo.collections.llm.gpt.data.packed_sequence import (
    tokenize_dataset,
    prepare_packed_sequence_data,
    PackedSequenceSpecs,
)


class MockTokenizer:
    def __init__(self):
        self.eos_id = 2

    def text_to_ids(self, text):
        # Simple mock implementation that converts each character to its ASCII value
        return [ord(c) % 10 for c in text]


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def sample_data_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write('{"input": "Hello"}\n')
        f.write('{"input": "World"}\n')
    yield Path(f.name)
    Path(f.name).unlink()


def test_tokenize_dataset(mock_tokenizer, sample_data_file):
    max_seq_length = 10
    seed = 42

    result = tokenize_dataset(
        path=sample_data_file,
        tokenizer=mock_tokenizer,
        max_seq_length=max_seq_length,
        seed=seed
    )

    assert isinstance(result, np.ndarray)
    assert len(result) > 0


def test_prepare_packed_sequence_data(mock_tokenizer, sample_data_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "packed_sequences.npy"
        metadata_path = Path(tmpdir) / "metadata.json"

        prepare_packed_sequence_data(
            input_path=sample_data_file,
            output_path=output_path,
            output_metadata_path=metadata_path,
            packed_sequence_size=16,
            tokenizer=mock_tokenizer,
            max_seq_length=10,
            seed=42,
            packing_algorithm="first_fit_shuffle"
        )

        # Check if output files were created
        assert output_path.exists()
        assert metadata_path.exists()


def test_packed_sequence_specs():
    # Test initialization with default values
    specs = PackedSequenceSpecs()
    assert specs.packed_sequence_size == -1
    assert specs.tokenizer_model_name is None
    assert specs.packed_train_data_path is None
    assert specs.packed_val_data_path is None
    assert specs.packed_metadata_path is None
    assert specs.pad_cu_seqlens is False

    # Test with valid packed data paths
    with tempfile.NamedTemporaryFile(suffix='.npy') as train_file, \
            tempfile.NamedTemporaryFile(suffix='.npy') as val_file:
        specs = PackedSequenceSpecs(
            packed_sequence_size=128,
            tokenizer_model_name="test-tokenizer",
            packed_train_data_path=train_file.name,
            packed_val_data_path=val_file.name
        )

        assert specs.packed_sequence_size == 128
        assert specs.tokenizer_model_name == "test-tokenizer"
        assert specs.packed_train_data_path == Path(train_file.name)
        assert specs.packed_val_data_path == Path(val_file.name)


def test_packed_sequence_specs_invalid_paths():
    # Test with non-existent file
    with pytest.raises(AssertionError):
        PackedSequenceSpecs(packed_train_data_path="nonexistent.npy")

    # Test with wrong file extension
    with tempfile.NamedTemporaryFile(suffix='.txt') as wrong_ext_file:
        with pytest.raises(AssertionError):
            PackedSequenceSpecs(packed_train_data_path=wrong_ext_file.name)