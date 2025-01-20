# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import datetime
import json
import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import torch

from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Type

from nemo.collections.nlp.data.information_retrieval.bert_embedding_dataset import BertEmbeddingDataset
from nemo.lightning.base import NEMO_DATASETS_CACHE
from nemo.core import Dataset
from nemo.utils import AppState, logging

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec

__idx_version__ = "0.2"  # index file version
__idx_suffix__ = "idx"  # index file suffix


class BertEmbeddingDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = True,
        add_eos: bool = True,
        max_num_samples: int = None,
        seed: int = 1234,
        index_mapping_dir: str = None,
        virtual_tokens: int = 0,
        memmap_workers: Optional[int] = None,
        truncation_method: str = 'right',
        special_tokens: Optional[Mapping[str, str]] = None,  # special tokens, a dictory of {token_type: token}
        data_type: str = 'train',  # train, query or doc
        num_hard_negatives: int = 4,
    ):
        """
        file_path: Path to a JSONL dataset with (query,pos_doc,neg_doc) triplets in jsonl format.
        tokenizer: Tokenizer for the dataset. Instance of a class that inherits TokenizerSpec (ex: YTTM, SentencePiece).
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        index_mapping_dir: Directory to save the index mapping to. If None, will write to the same folder as the dataset.
        truncation_method: Truncation from which position. Options: ['left', 'right']
        special_tokens: special tokens for the chat prompts, a dictionary of {token_type: token}. Default: {'system_turn_start': '<extra_id_0>', 'turn_start': '<extra_id_1>', 'label_start': '<extra_id_2>', 'end_of_turn': '\n', "end_of_name": "\n"}
        """
        # TODO: lot of copy-paste from GPTSFDDataset, should refactor both to use a common base class (@adithyare)
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.max_num_samples = max_num_samples
        self.seed = seed
        self.index_mapping_dir = index_mapping_dir
        self.virtual_tokens = virtual_tokens
        self.truncation_method = truncation_method
        if special_tokens is None:
            self.special_tokens = {
                "system_turn_start": "<extra_id_0>",
                "turn_start": "<extra_id_1>",
                "label_start": "<extra_id_2>",
                "end_of_turn": "\n",
                "end_of_name": "\n",
            }
        else:
            self.special_tokens = special_tokens
        self.data_type = data_type
        self.num_hard_negatives = num_hard_negatives

        self.indexed_dataset = JSONLMemMapDataset(
            dataset_paths=[file_path],
            tokenizer=None,
            header_lines=0,
            index_mapping_dir=index_mapping_dir,
            workers=memmap_workers,
        )
        # Will be None after this call if `max_num_samples` is None
        self.samples_mapping = None
        self._build_samples_mapping()

    def _build_samples_mapping(self):
        if self.max_num_samples is not None:
            self.samples_mapping = get_samples_mapping(
                indexed_dataset=self.indexed_dataset,
                data_prefix=self.file_path,
                num_epochs=None,
                max_num_samples=self.max_num_samples,
                max_seq_length=self.max_seq_length - 2,
                short_seq_prob=0,
                seed=self.seed,
                name=self.file_path.split('/')[-1],
                binary_head=False,
                index_mapping_dir=self.index_mapping_dir,
            )
        else:
            self.samples_mapping = None

    def __len__(self):
        if self.max_num_samples is None:
            return len(self.indexed_dataset)
        else:
            assert self.samples_mapping is not None
            return len(self.samples_mapping)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = idx.item()

        if self.samples_mapping is not None:
            assert idx < len(self.samples_mapping)
            idx, _, _ = self.samples_mapping[idx]
            if isinstance(idx, np.uint32):
                idx = idx.item()

        if idx is not None:
            assert idx < len(self.indexed_dataset)
        else:
            idx = -1
        # idx may < 0 because we pad_samples_to_global_batch_size, e.g. id = -1
        if idx < 0:
            idx = len(self) + idx
            auto_gen_idx = True
        else:
            auto_gen_idx = False
        try:
            example = self.indexed_dataset[idx]
            if auto_gen_idx:
                example['__AUTOGENERATED__'] = True
        except Exception as e:
            logging.error(f"Error while loading example {idx} from dataset {self.file_path}")
            raise e
        return self._process_example(example)

    def _process_example(self, example):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """

        metadata = {k: v for k, v in example.items()}
        if self.data_type == 'train':
            q = self.tokenizer.text_to_ids("query: " + example['query'].strip())
            d = self.tokenizer.text_to_ids("passage: " + example['pos_doc'].strip())
            # handle cases where the required number of hard negatives are not present
            if len(example['neg_doc']) < self.num_hard_negatives:
                nd = example['neg_doc']
                # sample rest with replacement
                nd = nd + choices(example['neg_doc'], k=self.num_hard_negatives - len(example['neg_doc']))
            else:
                # sample without replacement
                nd = sample(example['neg_doc'], k=self.num_hard_negatives)
            assert len(nd) == self.num_hard_negatives, "Error in sampling required number of hard negatives"
            nd = [self.tokenizer.text_to_ids("passage: " + ex.strip()) for ex in nd]

        elif self.data_type == 'query':
            q = self.tokenizer.text_to_ids("query: " + example['query'].strip())
            d, nd = None, None
            assert "query_id" in example, "query_id is required for query dataset"
            assert "doc_id" in example, "doc_id is required for query dataset"
        elif self.data_type == 'doc':
            d = self.tokenizer.text_to_ids("passage: " + example['pos_doc'].strip())
            assert "doc_id" in example, "doc_id is required for doc dataset"
            q, nd = None, None
        else:
            raise ValueError(f"Invalid data type: {self.data_type}")

        q = q if q is not None else []
        d = d if d is not None else []
        nd = nd if nd is not None else []

        if self.virtual_tokens:
            # (@adithyare) we are going to insert "pad/eos" tokens in the beginning of the text and context
            # these pad/eos tokens are placeholders for virtual tokens for ptuning (if used)
            q = [self.tokenizer.eos_id] * self.virtual_tokens + q  # type: ignore
            d = [self.tokenizer.eos_id] * self.virtual_tokens + d  # type: ignore
            nd = [[self.tokenizer.eos_id] * self.virtual_tokens + n for n in nd]  # type: ignore

        if self.add_bos:
            q = [self.tokenizer.bos_id] + q  # type: ignore
            d = [self.tokenizer.bos_id] + d  # type: ignore
            nd = [[self.tokenizer.bos_id] + n for n in nd]  # type: ignore

        # TODO: (@adithyare) should probably add a warning before truncation
        q = q[: self.max_seq_length - 1]
        d = d[: self.max_seq_length - 1]
        nd = [n[: self.max_seq_length - 1] for n in nd]

        if self.add_eos:
            q = q + [self.tokenizer.eos_id]  # type: ignore
            d = d + [self.tokenizer.eos_id]  # type: ignore
            nd = [n + [self.tokenizer.eos_id] for n in nd]  # type: ignore

        processed_example = {
            'query': q,
            'pos_doc': d,
            'neg_doc': nd,
            'metadata': metadata,
        }
        return processed_example

    def _maybe_cast_to_list(self, x):
        if isinstance(x, np.ndarray):
            return [item.tolist() for item in x]
        return x

    def _ceil_to_nearest(self, n, m):
        return (n + m - 1) // m * m

    def _collate_item(self, item, max_length, pad_id):
        item = self._maybe_cast_to_list(item)
        # max_length = max([len(x) for x in item]) if item else 0
        # here [0] should be tokenizer.pad_id
        item = [x + [pad_id] * (max_length - len(x)) for x in item]
        return item

    @torch.no_grad()
    def _create_attention_mask(self, max_length):
        """Create `attention_mask`.
        Args:
            input_ids: A 1D tensor that holds the indices of tokens.
        """
        # seq_length = len(input_ids)
        # `attention_mask` has the shape of [1, seq_length, seq_length]
        attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
        attention_mask = attention_mask < 0.5
        return attention_mask

    @torch.no_grad()
    def _create_attention_mask2(self, max_length, item_lengh):
        """Create `attention_mask`.
        Args:
            input_ids: A 1D tensor that holds the indices of tokens.
        """
        # seq_length = len(input_ids)
        # `attention_mask` has the shape of [1, seq_length, seq_length]
        attention_mask = torch.zeros(max_length)
        attention_mask[:item_lengh] = 1
        return attention_mask

    def collate_fn(self, batch):
        input_ids = []
        metadata = []
        lengths = []
        max_length = -1
        for item in batch:
            metadata.append(item['metadata'])
            if self.data_type == 'train':
                input_ids.append(item['query'])
                lengths.append(len(item['query']))
                input_ids.append(item['pos_doc'])
                lengths.append(len(item['pos_doc']))
                for nd in item['neg_doc']:
                    input_ids.append(nd)
                    lengths.append(len(nd))
                max_length = max(
                    max_length, len(item['query']), len(item['pos_doc']), *(len(nd) for nd in item['neg_doc'])
                )
            elif self.data_type == 'query':
                input_ids.append(item['query'])
                lengths.append(len(item['query']))
                max_length = max(max_length, len(item['query']))
            elif self.data_type == 'doc':
                input_ids.append(item['pos_doc'])
                lengths.append(len(item['pos_doc']))
                max_length = max(max_length, len(item['pos_doc']))
            else:
                raise ValueError(f"Invalid data type: {self.data_type}")

        max_length = min(self.max_seq_length, self._ceil_to_nearest(max_length, 16))
        assert max_length <= self.max_seq_length

        attention_mask = [self._create_attention_mask2(max_length, len) for len in lengths]
        attention_mask = torch.stack(attention_mask)
        position_ids = [list(range(max_length)) for _ in batch]
        position_ids = torch.LongTensor(position_ids)
        input_ids = torch.LongTensor(self._collate_item(input_ids, max_length=max_length, pad_id=0))
        lengths = torch.LongTensor(lengths) - 1  # subtract 1 to account for the eos token

        processed_batch = {
            'input_ids': input_ids,
            'token_type_ids': torch.zeros_like(input_ids),
            'attention_mask': attention_mask,
            'metadata': metadata,
        }

        return processed_batch


def get_dataset_root(name: str) -> Path:
    """Retrieve the root path for the dataset. Create the folder if not exists."""
    output = Path(NEMO_DATASETS_CACHE) / name
    output.mkdir(parents=True, exist_ok=True)

    return output


def create_sft_dataset(
    path: Path,
    tokenizer: "TokenizerSpec",
    seq_length: int = 2048,
    add_bos: bool = False,
    add_eos: bool = True,
    seed: int = 1234,
    index_mapping_dir: Optional[str] = None,
    truncation_method: str = 'right',
    memmap_workers: int = 2,
    data_type: str = 'train',
    num_hard_negatives: int = 1,
    **kwargs,
) -> "BertEmbeddingDataset":
    """Create BertEmbeddingDataset for SFT training."""

    return BertEmbeddingDataset(
        file_path=str(path),
        tokenizer=tokenizer,
        max_seq_length=seq_length,
        add_bos=add_bos,
        add_eos=add_eos,
        memmap_workers=memmap_workers,
        seed=seed,
        index_mapping_dir=index_mapping_dir,
        truncation_method=truncation_method,
        data_type=data_type,
        num_hard_negatives=num_hard_negatives,
        **kwargs,
    )


class JSONLMemMapDataset(TextMemMapDataset):
    """
    Memory-mapped iteration over a JSONL file.
    """

    def __init__(
        self,
        dataset_paths: List[str],
        newline_int: Optional[int] = 10,
        header_lines: Optional[int] = 0,
        workers: Optional[int] = None,
        tokenizer: Optional[Type["TokenizerSpec"]] = None,
        sort_dataset_paths: Optional[bool] = True,
        index_mapping_dir: Optional[str] = None,
    ):
        """
        Args:
            dataset_paths: list of JSONL file paths.
            newline_int: ASCII code to use to interpret newlines in file.
            header_lines: number of header lines in JSON files.
            workers: number of workers to use for creating index files.
            tokenizer: tokenizer to use to convert text to tokens.
            sort_dataset_paths: whether to sort datasets by paths.
            index_mapping_dir: directory to save the index mapping to.
                If None, will write to the same folder as the dataset.
        """
        super().__init__(
            dataset_paths=dataset_paths,
            newline_int=newline_int,
            header_lines=header_lines,
            workers=workers,
            tokenizer=tokenizer,
            sort_dataset_paths=sort_dataset_paths,
            index_mapping_dir=index_mapping_dir,
        )

    def _build_data_from_text(self, text):
        """Return a dictionary of data based on a single JSON line."""
        try:
            record = json.loads(text)
        except Exception as e:
            logging.error(f"Exception: {e}")
            logging.error(f"datapoint: {text}")
            raise e
        return record


class TextMemMapDataset(Dataset):
    """
    Allow per-line lazy access to multiple text files using numpy memmap.
    """

    def __init__(
        self,
        dataset_paths: List[str],
        newline_int: Optional[int] = 10,
        header_lines: Optional[int] = 0,
        workers: Optional[int] = None,
        tokenizer: Optional[Type["TokenizerSpec"]] = None,
        build_index_fn: Optional[Callable[[str, Optional[int]], bool]] = _build_index_from_memdata,
        sort_dataset_paths: Optional[bool] = True,
        index_mapping_dir: Optional[str] = None,
    ):
        """
        Args:
            dataset_paths: list of JSONL file paths.
            newline_int: ASCII code to use to interpret newlines in file.
            header_lines: number of header lines in JSON files.
            workers: number of workers to use for creating index files.
            tokenizer: tokenizer to use to convert text to tokens.
            build_index_fn: a callable build_index_fn(fn, newline_int) -> midx [np.array]
                that returns the index of newlines in a file fn must be pickleable
                (to be used in multiprocessing.Pool.map).
            sort_dataset_paths: whether to sort datasets by paths.
            index_mapping_dir: directory to save the index mapping to.
                If None, will write to the same folder as the dataset.
        """
        super().__init__()
        self.mdata_midx_list = []

        # Make a single string into a list
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]

        if len(dataset_paths) < 1:
            raise ValueError("files_list must contain at leat one file name")

        self._newline_int = newline_int
        # skip first N lines
        self._header_lines = header_lines
        self._files_list = dataset_paths
        self._worker = workers
        self.tokenizer = tokenizer
        self._sort_dataset_paths = sort_dataset_paths

        if sort_dataset_paths:
            self._files_list = sorted(self._files_list)

        logging.info(f"Building data files")
        # load all files into memmap
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

        if not is_distributed or (is_distributed and torch.distributed.get_rank() == 0):
            # Create index files on global rank 0.
            build_index_files(
                dataset_paths,
                newline_int,
                workers=self._worker,
                build_index_fn=build_index_fn,
                index_mapping_dir=index_mapping_dir,
            )

        if is_distributed and not _lightning_prepare_data():
            torch.distributed.barrier()

        if is_distributed and AppState().local_rank == 0:
            # If we are in a distributed multi-node set-up and index files are not stored on
            # a shared filesystem, then the index files created on global rank 0 are only
            # accessible to the workers on that node.
            #
            # Two cases may occur here:
            #
            # 1. case of a shared filesystem, or global_rank==0: the index files are present in
            #    the locally available filesystem, calling build_index_files() again is a no-op.
            # 2. case of a non-shared filesystem, and global_rank>0: the index files are not
            #    present in the locally available filesystem, calling build_index_files() again
            #    will create them.
            #
            # Outcome in all cases: all nodes have access to the index files in their filesystem.
            build_index_files(
                dataset_paths,
                newline_int,
                workers=self._worker,
                build_index_fn=build_index_fn,
                index_mapping_dir=index_mapping_dir,
            )

        if is_distributed and not _lightning_prepare_data():
            torch.distributed.barrier()

        logging.info(f"Loading data files")
        start_time = time.time()
        mdata_midx_list = [self.load_file(fn, index_mapping_dir) for fn in self._files_list]
        logging.info(
            f"Time loading {len(mdata_midx_list)} mem-mapped files: {datetime.timedelta(seconds=time.time() - start_time)}"
        )

        logging.info("Computing global indices")
        midx_bins = np.cumsum([(len(midx) - header_lines) for _, midx in mdata_midx_list])

        self.midx_bins = midx_bins
        self.mdata_midx_list = mdata_midx_list

        # figure out size of the dataset
        self._size = self.midx_bins[-1]

    def __del__(self):
        if self.mdata_midx_list:
            for mdata, midx in self.mdata_midx_list:
                mdata._mmap.close()

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        """
        Return a string from binary memmap
        """
        if (idx >= len(self)) or (idx < 0):
            raise IndexError(f"Index {idx} if out of dataset range with {len(self)} samples")

        # Identify the file containing the record
        file_id = np.digitize(idx, self.midx_bins, right=False)
        base_idx = self.midx_bins[file_id - 1] if file_id > 0 else 0
        file_idx = idx - base_idx + self._header_lines
        mdata, midx = self.mdata_midx_list[file_id]
        # load sample
        if file_idx == 0:
            i = 0
            j = midx[0]
        else:
            i = midx[file_idx - 1] + 1  # ignore newline
            j = midx[file_idx]

        # fetch sample from memmap

        try:
            sample = self._fetch_sample_from_memmap(mdata, i, j)
        except Exception as e:
            logging.error(f"Error while fetching sample from memmap: {e}")
            logging.error(f"file_id: {file_id}, file_idx: {file_idx}, i: {i}, j: {j}")
            raise e

        # parse raw text (e.g., tokenize)
        try:
            data = self._build_data_from_text(sample)
        except Exception as e:
            logging.error(
                f"Error while building data from text, possible issue with sample expected format (see offending sample below): {e}"
            )
            logging.error(f"sample: {sample}, file_id: {file_id}, file_idx: {file_idx}, i: {i}, j: {j}")
            raise e

        return data

    def _fetch_sample_from_memmap(self, mdata, i, j):
        """Fetchs the text sample. Can be overriden by child-classes to support loading of partial samples and alternative decode methods"""
        # load text sample by slicing memmap data[i:j]
        text = mdata[i:j].tobytes().decode("utf-8")

        return text

    def _build_data_from_text(self, text):
        """Allows child-classes to modify the parsing of raw text, prior to tokenization"""
        # tokenize text if tokenizer is given
        if self.tokenizer is not None:
            data = self.tokenizer.text_to_ids(text)
        else:
            data = text

        return data

    def load_file(self, fn, index_mapping_dir: Optional[str] = None):
        """
        Loads a text file as np.int8.

        Returns:
            mdata - memorymap of np.int8
            midx - indices pointing to the end-of-line (or end of file) position
            size - number of lines in file
        """
        logging.info(f"Loading {fn}")
        idx_fn = _index_fn(fn, index_mapping_dir)

        # create data map
        mdata = np.memmap(fn, dtype=np.uint8, mode="r")

        if _index_file_exists(idx_fn):
            # load index file into memory map
            midx = np.load(idx_fn + ".npy", allow_pickle=True, mmap_mode="r")
            # test for header
            if len(midx) < self._header_lines:
                raise RuntimeError(f"Missing header, expected {self._header_lines} header lines")

            # load meta info
            with open(idx_fn + ".info", "rb") as fp:
                idx_info_dict = pickle.load(fp)
            # test for mismatch in expected newline_int
            if "newline_int" in idx_info_dict:
                newline_int = idx_info_dict["newline_int"]
                if self._newline_int != newline_int:
                    logging.warning(
                        f"Mismatch in newline_int, expected = {self._newline_int} but loaded {newline_int}"
                    )

            # test for version mismatch (useful to force recreation of index files)
            idx_version = idx_info_dict.get("version", "0.0")
            if __idx_version__ != idx_version:
                raise RuntimeError(
                    f"Version mismatch: Please delete existing '.{__idx_suffix__}' files. Expected version = {__idx_version__}, but file version = {idx_version}. File path = {idx_fn}"
                )
        else:
            raise ValueError(
                f"Memory Map for {fn} is not found, missing one or more of files: {idx_fn}.{{.npy,.info}}"
            )

        return (mdata, midx)