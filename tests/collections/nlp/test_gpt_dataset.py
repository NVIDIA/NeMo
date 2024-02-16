"""Tests for GPTDataset."""
import contextlib
import dataclasses
import io
import os
import subprocess
import tempfile

import jsonlines
import numpy as np
import pytest
import s3fs
import torch
import torch.distributed
import transformers
from megatron.core.parallel_state import initialize_model_parallel
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import (
    GPTDataset,
    _build_shuffle_idx,
    get_indexed_dataset_,
)
from nemo.utils import AppState
from tests.fixtures.s3 import local, local_parent_dir, s3, s3_client, s3_parent_dir, s3_server

# We preprocess these documents to create the .bin and .idx files.
_DOCS = [
    {"text": "a b c"},
    {"text": "d e f g h i j"},
    {"text": "k l"},
    {"text": "m"},
    {"text": "n o p"},
    {"text": "q r s"},
]


# The number of tokens in a sample.
_SEQ_LENGTH = 4


# This list consists of the tokens in the expected, decoded
# samples when:
# * we have 1 epoch worth of samples
# * we set `shuffle_documents` to False
# * the sequence length is `_SEQ_LENGTH`
# * we use the GPTDataset to look up each sample in order
#
# The way it works is that we construct samples from the
# documents in `_DOCS`. Each sample consists of `_SEQ_LENGTH`
# tokens. Note that <|endoftext|> is a special token that denotes
# the end of a document. Once the samples are constructed,
# we shuffle them.
_EXPECTED_TOKENS_1_EPOCH = [
    "<|endoftext|>q r s",
    " h i j<|endoftext|>",
    "d e f g",
    "k l<|endoftext|>m",
    "a b c<|endoftext|>",
    "<|endoftext|>n o p",
]


# Same as above except for labels rather than tokens.
_EXPECTED_LABELS_1_EPOCH = [
    "q r s<|endoftext|>",
    " i j<|endoftext|>k",
    " e f g h",
    " l<|endoftext|>m<|endoftext|>",
    " b c<|endoftext|>d",
    "n o p<|endoftext|>",
]


# This list consists of the tokens in the expected, decoded samples
# with the same settings as `_EXPECTED_1_EPOCH` except that we set
# `shuffle_documents` to True.
#
# The shuffled document order happens to be:
#
# [5, 2, 1, 3, 0, 4]
#
# So the shuffled docs are:
#
# {"text": "q r s"}
# {"text": "k l"}
# {"text": "d e f g h i j"}
# {"text": "m"}
# {"text": "a b c"}
# {"text": "n o p"}
#
# So the samples are:
#
# "q r s<|endoftext|>"
# "k l<|endoftext|>d"
# "e f g h"
# "i j<|endoftext|>m"
# "<|endoftext|>a b c"
# "<|endoftext|>n o p"
#
# Then, we shuffle the samples.
_EXPECTED_TOKENS_1_EPOCH_SHUFFLED_DOCS = [
    "k l<|endoftext|>d",
    " i j<|endoftext|>m",
    "<|endoftext|>a b c",
    "q r s<|endoftext|>",
    " e f g h",
    "<|endoftext|>n o p",
]


# Same as above except for labels rather than tokens.
_EXPECTED_LABELS_1_EPOCH_SHUFFLED_DOCS = [
    " l<|endoftext|>d e",
    " j<|endoftext|>m<|endoftext|>",
    "a b c<|endoftext|>",
    " r s<|endoftext|>k",
    " f g h i",
    "n o p<|endoftext|>",
]


# This list consists of the tokens in the expected, decoded samples
# with: the same settings as `_EXPECTED_1_EPOCH` except that we use
# 2 epochs worth of samples.
#
# The shuffled sample order happens to be:
#
# [6, 11, 4, 10, 2, 8, 1, 7, 9, 3, 0, 5]
_EXPECTED_TOKENS_2_EPOCHS = [
    "<|endoftext|>a b c",  # 6: 2nd epoch
    " p<|endoftext|>q r",  # 11: 2nd epoch
    "<|endoftext|>n o p",  # 4: 1st epoch
    "m<|endoftext|>n o",  # 10: 2nd epoch
    " h i j<|endoftext|>",  # 2: 1st epoch
    " g h i j",  # 8: 2nd epoch
    "d e f g",  # 1: 1st epoch
    "<|endoftext|>d e f",  # 7: 2nd epoch
    "<|endoftext|>k l<|endoftext|>",  # 9: 2nd epoch
    "k l<|endoftext|>m",  # 3: 1st epoch
    "a b c<|endoftext|>",  # 0: 1st epoch
    "<|endoftext|>q r s",  # 5: 1st epoch
]


# Same as above except for labels rather than tokens.
_EXPECTED_LABELS_2_EPOCHS = [
    "a b c<|endoftext|>",
    "<|endoftext|>q r s",
    "n o p<|endoftext|>",
    "<|endoftext|>n o p",
    " i j<|endoftext|>k",
    " h i j<|endoftext|>",
    " e f g h",
    "d e f g",
    "k l<|endoftext|>m",
    " l<|endoftext|>m<|endoftext|>",
    " b c<|endoftext|>d",
    "q r s<|endoftext|>",
]


@dataclasses.dataclass
class InputDataset:
    data_prefix: str
    num_docs: int
    num_tokens: int


@pytest.fixture(scope="session")
def setup_distributed_training():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    torch.distributed.init_process_group("gloo", rank=0, world_size=1)
    AppState().local_rank = 0
    initialize_model_parallel()


@pytest.fixture(scope="session")
def tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokenizer.eos_id = tokenizer.unk_token_id
    return tokenizer


@pytest.fixture(scope="session")
def input_dataset(tmpdir_factory, tokenizer):
    # Calculate the number of tokens in the documents.
    num_non_eos_tokens = sum([len(tokenizer.encode(doc["text"])) for doc in _DOCS])
    # The last doc does not get an end-of-sentence (eos) token.
    num_eos_tokens = len(_DOCS) - 1
    num_tokens = num_non_eos_tokens + num_eos_tokens

    # Write out the documents.
    name = "data"
    output_dir = str(tmpdir_factory.mktemp("tmp"))
    data_prefix = os.path.join(output_dir, name + "_text_document")
    path = os.path.join(output_dir, f"{name}.jsonl")
    with jsonlines.open(path, "w") as fout:
        fout.write_all(_DOCS)

    # Preprocess the documents to generate the .bin and .idx files.
    nemo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
    preprocess_script = os.path.join(nemo_dir, "scripts", "nlp_language_modeling", "preprocess_data_for_megatron.py")
    assert os.path.exists(preprocess_script)
    args = [
        "python",
        preprocess_script,
        "--input=" + path,
        "--json-keys=text",
        "--tokenizer-library=huggingface",
        "--tokenizer-type=gpt2",
        "--dataset-impl=mmap",
        "--output-prefix=" + os.path.join(output_dir, name),
        "--append-eod",
        "--workers=1",
    ]
    proc = subprocess.run(args)
    assert proc.returncode == 0

    # Return the input dataset.
    return InputDataset(data_prefix=data_prefix, num_docs=len(_DOCS), num_tokens=num_tokens)


class TestGPTDataset:
    @pytest.mark.parametrize(
        "fs,num_epochs,shuffle_documents,expected_tokens,expected_labels,expected_assertion_error",
        [
            ("local", 1, False, _EXPECTED_TOKENS_1_EPOCH, _EXPECTED_LABELS_1_EPOCH, False),
            ("local", 2, False, _EXPECTED_TOKENS_2_EPOCHS, _EXPECTED_LABELS_2_EPOCHS, False),
            ("local", 1, True, _EXPECTED_TOKENS_1_EPOCH_SHUFFLED_DOCS, _EXPECTED_LABELS_1_EPOCH_SHUFFLED_DOCS, False),
            ("s3", 1, False, _EXPECTED_TOKENS_1_EPOCH, _EXPECTED_LABELS_1_EPOCH, False),
            ("s3", 2, False, _EXPECTED_TOKENS_2_EPOCHS, _EXPECTED_LABELS_2_EPOCHS, False),
            # Expected AssertionError: S3 data loading is incompatible with enabled shuffle_documents.
            ("s3", 1, True, _EXPECTED_TOKENS_1_EPOCH_SHUFFLED_DOCS, _EXPECTED_LABELS_1_EPOCH_SHUFFLED_DOCS, True),
        ],
    )
    def test_basic(
        self,
        request,
        fs,
        monkeypatch,
        s3_client,
        local_parent_dir,
        setup_distributed_training,
        input_dataset,
        tokenizer,
        num_epochs,
        shuffle_documents,
        expected_tokens,
        expected_labels,
        expected_assertion_error,
    ):
        # Mock the call to get a boto3 client so that it returns our mock client.
        def mock_client(*args, **kwargs):
            return s3_client

        monkeypatch.setattr("boto3.client", mock_client)

        # Get the filesystem.
        parent_dir = request.getfixturevalue(fs + "_parent_dir")
        fs = request.getfixturevalue(fs)

        # Copy the .bin and .idx files for the input dataset into the
        # directory specific to the test item.
        data_prefix = parent_dir + input_dataset.data_prefix
        fs.makedirs(data_prefix)
        if isinstance(fs, s3fs.S3FileSystem):
            fs.upload(input_dataset.data_prefix + ".bin", data_prefix + ".bin")
            fs.upload(input_dataset.data_prefix + ".idx", data_prefix + ".idx")
        else:
            fs.copy(input_dataset.data_prefix + ".bin", data_prefix + ".bin")
            fs.copy(input_dataset.data_prefix + ".idx", data_prefix + ".idx")

        index_mapping_dir = local_parent_dir

        # Create the indexed dataset.
        indexed_dataset = get_indexed_dataset_(
            data_prefix,
            "mmap",
            skip_warmup=True,
            delay_data_mmap=False,
            index_cache_dir=index_mapping_dir,
            data_cache_nbytes=128 * 1024 * 1024,
        )

        # Create GPTDataset.
        assert input_dataset.num_tokens % _SEQ_LENGTH == 0
        cfg = OmegaConf.create(
            {"data": {"shuffle_documents": shuffle_documents, "index_mapping_dir": index_mapping_dir}}
        )

        context = pytest.raises(AssertionError) if expected_assertion_error else contextlib.nullcontext()
        gpt_dataset = None
        with context:
            gpt_dataset = GPTDataset(
                cfg=cfg,
                trainer=Trainer(),
                tokenizer=tokenizer,
                name="train",
                data_prefix=data_prefix,
                documents=np.arange(input_dataset.num_docs),
                indexed_dataset=indexed_dataset,
                num_samples=num_epochs * (input_dataset.num_tokens // _SEQ_LENGTH),
                seq_length=_SEQ_LENGTH,
                seed=0,
                drop_last=True,
            )
        if expected_assertion_error:
            return

        # Exhaust the GPTDataset and check the output.
        assert len(gpt_dataset) == len(expected_tokens)
        assert len(gpt_dataset) == len(expected_labels)
        for i in range(len(gpt_dataset)):
            sample = gpt_dataset[i]
            assert set(sample.keys()) == {"tokens", "labels", "position_ids", "loss_mask"}
            tokens = tokenizer.decode(sample["tokens"].tolist())
            assert tokens == expected_tokens[i]
            labels = tokenizer.decode(sample["labels"].tolist())
            assert labels == expected_labels[i]
            assert torch.equal(sample["position_ids"], torch.tensor([0, 1, 2, 3]))
            assert torch.equal(sample["loss_mask"], torch.tensor([1.0, 1.0, 1.0, 1.0]))

    @pytest.mark.parametrize(
        "num_samples,total_size,shuffle_block_size,expected",
        [
            # `shuffle_block_size` == 1
            (7, 10, 1, np.array([6, 2, 1, 3, 0, 5, 4, 9, 7, 8], dtype=np.uint32)),
            (7, 6, 1, np.array([6, 2, 1, 3, 0, 5, 4], dtype=np.uint32)),
            (7, 7, 1, np.array([6, 2, 1, 3, 0, 5, 4], dtype=np.uint32)),
            # `shuffle_block_size` > 1
            (12, 16, 4, np.array([10, 11, 8, 9, 4, 6, 5, 7, 3, 0, 2, 1, 13, 12, 14, 15], dtype=np.uint32)),
            (12, 11, 4, np.array([10, 11, 8, 9, 4, 6, 5, 7, 3, 0, 2, 1], dtype=np.uint32)),
            (12, 12, 4, np.array([10, 11, 8, 9, 4, 6, 5, 7, 3, 0, 2, 1], dtype=np.uint32)),
            # `shuffle_block_size` does not evenly divide `num_samples`
            # [10, 8, 9], [4, 6, 5, 7] and [3, 0, 2, 1] are the blocks.
            (11, 11, 4, np.array([10, 8, 9, 4, 6, 5, 7, 3, 0, 2, 1], dtype=np.uint32)),
            # `shuffle_block_size` is larger than `num_samples`
            (4, 4, 5, np.array([2, 3, 1, 0], dtype=np.uint32)),
            # edge cases
            (0, 0, 1, np.array([], dtype=np.uint32)),
            (0, 1, 1, np.array([0], dtype=np.uint32)),
            (1, 1, 1, np.array([0], dtype=np.uint32)),
            (1, 2, 1, np.array([0, 1], dtype=np.uint32)),
        ],
    )
    def test_build_shuffle_idx(self, num_samples, total_size, shuffle_block_size, expected):
        np_rng = np.random.RandomState(seed=0)
        actual = _build_shuffle_idx(num_samples, total_size, shuffle_block_size, np_rng)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
