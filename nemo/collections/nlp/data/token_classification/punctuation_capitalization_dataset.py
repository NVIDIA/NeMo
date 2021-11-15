# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

__all__ = [
    'BertPunctuationCapitalizationDataset',
    'DEFAULT_CAPIT_LABEL_IDS_NAME',
    'DEFAULT_PUNCT_LABEL_IDS_NAME',
    'Progress',
    'PunctuationCapitalizationEvalDataConfig',
    'PunctuationCapitalizationTrainDataConfig',
    'create_label_ids',
    'create_masks_and_segment_ids',
    'is_legacy_data_config',
    'legacy_data_config_to_new_data_config',
    'load_label_ids',
]

import itertools
import multiprocessing as mp
import os
import pickle
import random
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from queue import Empty
from time import sleep
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from numpy.typing import ArrayLike
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.data_utils.data_preprocessing import get_label_stats, get_stats
from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, LabelsType, MaskType, NeuralType
from nemo.utils import logging


MAX_NUM_QUERIES_IN_SPLIT = 10 ** 4
TOKENIZATION_PROGRESS_REPORT_PERIOD = 10 ** 3
BATCH_MARK_UP_PROGRESS_REPORT_PERIOD = 10 ** 4
BATCH_BUILDING_PROGRESS_REPORT_PERIOD = 10 ** 4

DEFAULT_PUNCT_LABEL_IDS_NAME = 'punct_label_ids.csv'
DEFAULT_CAPIT_LABEL_IDS_NAME = 'capit_label_ids.csv'

DEFAULT_TOKENS_IN_BATCH = 5000
DEFAULT_MAX_SEQ_LENGTH = 512


@dataclass
class PunctuationCapitalizationDataConfigBase:
    #################################################
    # COMMON DATASET PARAMETERS
    #################################################
    # Whether to use tarred dataset. If True you should provide tar_metadata_file, otherwise text_file and labels_file
    use_tarred_dataset: bool = False

    #################################################
    # USUAL DATASET PARAMETERS
    #################################################
    text_file: Optional[str] = None  # Any -- Union[str, List[str]]  A name of dataset source file
    labels_file: Optional[str] = None  # Any = str or List[str]  A name of dataset target file
    tokens_in_batch: int = DEFAULT_TOKENS_IN_BATCH
    max_seq_length: Optional[int] = DEFAULT_MAX_SEQ_LENGTH
    num_samples: int = -1
    use_cache: Optional[bool] = True
    get_label_frequences: bool = False
    verbose: bool = True
    # If 0, then multiprocessing is not used; if null, then n_jobs is equal to the number of CPU cores.
    # There can be weird deadlocking with some tokenizers (e.g. SentencePiece) if `n_jobs` is greater than zero.
    n_jobs: Optional[int] = 0

    #################################################
    # TARRED DATASET PARAMETERS
    #################################################
    tar_metadata_file: Optional[str] = None  # Any = str or List[str]  A name of metadata file for tarred dataset
    tar_shuffle_n: int = 1

    #################################################
    # DATALOADER PARAMETERS
    #################################################
    # Shuffle batches every epoch. For not tarred training datasets parameter also activates batch
    # repacking every epoch. For tarred dataset it would be only batches permutation.
    shuffle: bool = True
    drop_last: Optional[bool] = False
    pin_memory: Optional[bool] = True
    num_workers: Optional[int] = 8
    persistent_workers: Optional[bool] = True


@dataclass
class PunctuationCapitalizationTrainDataConfig(PunctuationCapitalizationDataConfigBase):
    # Path to a directory where `tar_metadata_file` or `text_file` and `labels_file` lay
    ds_item: Optional[str] = None


@dataclass
class PunctuationCapitalizationEvalDataConfig(PunctuationCapitalizationDataConfigBase):
    # Path to a directory where `tar_metadata_file` or `text_file` and `labels_file` lay
    # Any = str or List[str]. If a List[str], then multiple dataset testing or evaluation is used
    ds_item: Optional[Any] = None


def is_legacy_data_config(ds_section: DictConfig) -> bool:
    return 'use_tarred_dataset' not in ds_section


def legacy_data_config_to_new_data_config(
    ds_section: DictConfig, legacy_dataset_section: DictConfig, train: bool
) -> DictConfig:
    if train:
        cls = PunctuationCapitalizationTrainDataConfig
        ds_item = legacy_dataset_section.get('data_dir')
    else:
        cls = PunctuationCapitalizationEvalDataConfig
        ds_item = ds_section.get('ds_item')
        ds_item = legacy_dataset_section.get('data_dir') if ds_item is None else ds_item
    if ds_item is None:
        raise ValueError(
            f"Data directory was not found in legacy config.\nspecific dataset configuration:\n"
            f"{OmegaConf.to_yaml(ds_section)}\nmodel.dataset:\n{OmegaConf.to_yaml(legacy_dataset_section)}"
        )
    new_config = OmegaConf.structured(
        cls(
            use_tarred_dataset=False,
            text_file=ds_section.text_file,
            labels_file=ds_section.labels_file,
            ds_item=ds_item,
            max_seq_length=legacy_dataset_section.get('max_seq_length', DEFAULT_MAX_SEQ_LENGTH),
        )
    )
    return new_config


def check_number_of_labels(words, query, qi, split_i, punctuation_labels, capitalization_labels):
    if len(words) != len(punctuation_labels):
        raise ValueError(
            f"Number of punctuation labels for query {qi} in split {split_i} is not equal to number of "
            f"words. Number of words: {len(words)}, number of punctuation labels: "
            f"{len(punctuation_labels)}. Query: '{query}', punctuation labels: '{punctuation_labels}'"
        )
    if len(words) != len(capitalization_labels):
        raise ValueError(
            f"Number of capitalization labels for query {qi} in split {split_i} is not equal to number of "
            f"words. Number of words: {len(words)}, number of capitalization labels: "
            f"{len(capitalization_labels)}. Query: '{query}', "
            f"capitalization labels: '{capitalization_labels}'"
        )


def show_prog(queues: Tuple[mp.Queue, ...], total_num_lines: List[int], descriptions: List[str], units: List[str]):
    """
    Show several ``tqdm`` progress bars.
    Args:
        queues: a list of queues by which progress is delivered. Each queue is responsible for one progress bar.
            ``show_prog`` function extracts integers from ``queues`` elements and ands them to progress bars
        total_num_lines: list of values 100% of progress bars. See more in description of ``total`` parameter of
            ``tqdm.tqdm`` function
        descriptions: list of descriptions of progress bars. See more in description of ``desc`` parameter of
            ``tqdm.tqdm`` function
        units: list of progress bar units. See more in description of ``unit`` parameter of ``tqdm.tqdm`` function
    """
    if not all([len(queues) == len(v) for v in [total_num_lines, descriptions, units]]):
        raise ValueError(
            f"All of parameters `queues`, `total_num_lines`, `descriptions`, `units` have to have equal lengths. "
            f"len(queues)={len(queues)}, len(total_num_lines)={len(total_num_lines)}, "
            f"len(descriptions)={len(descriptions)}, len(units)={len(units)}."
        )
    prog = [
        tqdm(total=tt, desc=dd, unit=uu, unit_scale=True, position=i)
        for i, (tt, dd, uu) in enumerate(zip(total_num_lines, descriptions, units))
    ]
    finished = [False] * len(queues)
    while True:
        for i, queue in enumerate(queues):
            stop = False
            to_add = 0
            try:
                v = queue.get(block=False)
                while v != -1:
                    to_add += v
                    v = queue.get(block=False)
                stop = True
            except Empty:
                if to_add == 0 and not stop:
                    continue
            prog[i].n += to_add
            prog[i].update(0)
            if prog[i].n >= total_num_lines[i]:
                finished[i] = True
                prog[i].close()
            if stop:
                if prog[i].n < total_num_lines[i]:
                    logging.warning(
                        f"Progress process with description '{descriptions[i]}' terminated before progress bar "
                        f"reached 100%. prog.n={prog[i].n}, total_num_lines={total_num_lines[i]}"
                    )
                finished[i] = True
                prog[i].close()
        if all(finished):
            break
        sleep(0.1)


class Progress:
    """
    Manages several ``tqdm`` progress bars for multi process tasks. ``Progress`` class can be used as context manager.

    The class starts separate process which creates and updates progress bars. Information is passed to progress
    process via multiprocessing queues. There is a queue for every progress bar

    You can use it as context manager:

    .. code-block:: python
        with Progress([10, 20], ["progress bar 1", "progress bar 2"], ["parrot", "frog"]) as progress_queues:
            num_processes = 10
            with multiprocessing.Pool(num_processes) as pool:
                data = list(zip(my_data, [progress_queues[0]] * num_processes, [progress_queues[1]] * num_processes))
                pool.starmap(worker_func, data)

    Or without context manager:

    .. code-block:: python
        progress = Progress([10, 20], ["progress bar 1", "progress bar 2"], ["parrot", "frog"])
        progress_queues = progress.get_queue()
        num_processes = 10
        with multiprocessing.Pool(num_processes) as pool:
            data = list(zip(my_data, [progress_queues[0]] * num_processes, [progress_queues[1]] * num_processes))
            pool.starmap(worker_func, data)
        progress.finish()

    In a worker function you will have to put number of processed items into progress queues. For example:

    .. code-block:: python
        def worker_func(my_datum, parrot_progress_queue, frog_progress_queue):
            ...
            for i in range(10):
                parrot_progress_queue.put(1)
                frog_progress_queue.put(2)
    """

    def __init__(self, total: Union[int, List[int]], desc: Union[str, List[str]], unit: Union[str, List[str]]):
        """
        Starts progress process and creates queues for passing information to the progress process. Number of progress
        bars is equal to the max length of lists ``total``, ``desc``, ``unit``. If none of these parameters is a list,
        then 1 progress bar is created.

        Args:
            total: a list of ``int`` which length is equal to the number of progress bars OR an ``int`` OR a list of
                one ``int``. Number which comprises 100% of work. When sum of values passed through the queue equals
                ``total`` corresponding progress bar reaches 100%. If ``total`` is an ``int`` or a list of one
                element, then all progress bars have equal ``total`` parameter.
            desc: a list of ``str`` which length is equal to the number of progress bars OR a ``str`` OR a list of one
                ``str``. Description of a progress bar which is showed  as prefix. See more in description of parameter
                ``desc`` of function ``tqdm.tqdm``.
            unit: a list of ``str`` which length is equal to the number of progress bars OR a ``str`` OR a list of one
                ``str``. A unit of progress bar. See more in description of parameter ``unit`` of function
                ``tqdm.tqdm``.
        """
        if not isinstance(total, list):
            total = [total]
        if not isinstance(desc, list):
            desc = [desc]
        if not isinstance(unit, list):
            unit = [unit]
        num_processes = max([len(total), len(desc), len(unit)])
        for param in [total, desc, unit]:
            if len(param) not in [num_processes, 1]:
                raise ValueError(
                    f"If parameter of `Progress.__init__` method is a list, then it has to be the same length as other "
                    f"parameters which are lists"
                )
            if len(param) == 1:
                param *= num_processes
        manager = mp.Manager()
        self.progress_queues = tuple(manager.Queue() for _ in range(num_processes))
        self.progress_process = mp.Process(target=show_prog, args=(self.progress_queues, total, desc, unit))
        self.progress_process.start()

    def __enter__(self):
        return self.get_queues()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def get_queues(self) -> Tuple[mp.Queue, ...]:
        return self.progress_queues

    def finish(self):
        for q in self.progress_queues:
            q.put(-1)
        self.progress_process.join()


class TokenizeCreateMasksClipWorker:
    """A worker for tokenization, encoding labels, creating masks for first token in a word, sequence clipping"""

    def __init__(
        self,
        max_seq_length: int,
        tokenizer: TokenizerSpec,
        punct_label_ids: Optional[Dict[str, int]],
        capit_label_ids: Optional[Dict[str, int]],
        pad_label: str,
        verbose: bool,
        progress_queue: mp.Queue,
    ):
        """
        Args:
            max_seq_length: max number of tokens in input sequence including [CLS] and [SEP] tokens. If number of
                tokens in a sequence exceeds ``max_seq_length``, then excess tokens in the end of the sequence
                are removed
            tokenizer: a tokenizer instance which has properties ``cls_id``, ``pad_id``, ``sep_id``, ``unk_id``
            punct_label_ids: dict to map punctuation labels to label ids.
                Starts with pad_label->0 and then increases in alphabetical order.
                Required for training and evaluation, not needed for inference.
            capit_label_ids: dict to map capitalization labels to label ids. Starts
                with pad_label->0 and then increases in alphabetical order.
                Required for training and evaluation, not needed for inference.
            pad_label: pad value use for labels. By default, it's the neutral label for punctuation and capitalization.
            verbose: whether to show examples of tokenized data and various progress information
            progress_queue: a multiprocessing queue used for reporting progress. Useful for creating tarred dataset
        """
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.punct_label_ids = punct_label_ids
        self.capit_label_ids = capit_label_ids
        self.pad_label = pad_label
        self.verbose = verbose
        self.progress_queue = progress_queue

    def maybe_clip(self, values: List[int], append_value: int) -> List[int]:
        if len(values) > self.max_seq_length:
            return values[: self.max_seq_length - 1] + [append_value]
        return values

    def __call__(
        self,
        queries: List[str],
        punct_label_lines: Optional[Union[List[str], Tuple[str, ...]]],
        capit_label_lines: Optional[Union[List[str], Tuple[str, ...]]],
        split_i: int,
    ) -> Tuple[List[ArrayLike], List[ArrayLike], List[int], List[ArrayLike], List[ArrayLike]]:
        """
        Args:
            queries: text sequences
            punct_label_lines: a list or a tuple of labels for every word in a sequence (str)
            capit_label_lines: a list of a tuple labels for every word in a sequence (str)
            split_i: number of split processed by worker. Used for logging

        Returns:
            input_ids: a list of 1D int32 arrays. Each array contains token ids of corresponding query
            subtokens_mask: a list of 1D boolean arrays. An array element is ``True`` if corresponding token is the
                first token in a word
            sent_lengths: a list of sequences lengths. A sequence length is a length is a length of corresponding
                ``input_ids`` element
            punct_labels: a list of 1D int32 arrays. Encoded punctuation labels for every token in a query. Tokens in
                one word have identical labels. If ``with_label`` is ``False``, then ``punct_labels`` is an empty list
            capit_labels: a list of 1D int32 arrays. Encoded capitalization labels for every token in a query. Tokens in
                one word have identical labels. If ``with_label`` is ``False``, then ``capit_labels`` is an empty list
        """
        all_input_ids, all_subtokens_mask, sent_lengths = [], [], []
        punct_all_labels, capit_all_labels = [], []
        progress_made = 0
        for i, query in enumerate(queries):
            words = query.split()
            input_ids, subtokens_mask = [self.tokenizer.cls_id], [0]
            check_number_of_labels(words, query, i, split_i, punct_label_lines[i], capit_label_lines[i])
            pad_id = self.punct_label_ids[self.pad_label]
            punct_labels = [pad_id]
            punct_query_labels = [self.punct_label_ids[lab] for lab in punct_label_lines[i]]
            capit_labels = [pad_id]
            capit_query_labels = [self.capit_label_ids[lab] for lab in capit_label_lines[i]]
            for j, word in enumerate(words):
                word_ids = self.tokenizer.text_to_ids(word)
                if not word_ids and len(word):
                    word_ids = [self.tokenizer.unk_id]
                input_ids.extend(word_ids)

                subtokens_mask.append(1)
                subtokens_mask.extend([0] * (len(word_ids) - 1))

                if self.with_label:
                    punct_labels.extend([punct_query_labels[j]] * len(word_ids))
                    capit_labels.extend([capit_query_labels[j]] * len(word_ids))

            # add eos token
            input_ids.append(self.tokenizer.sep_id)
            subtokens_mask.append(0)
            sent_lengths.append(len(input_ids))

            all_input_ids.append(np.array(self.maybe_clip(input_ids, self.tokenizer.sep_id), dtype=np.int32))
            all_subtokens_mask.append(np.array(self.maybe_clip(subtokens_mask, 0), dtype=bool))

            punct_labels.append(pad_id)
            punct_all_labels.append(np.array(self.maybe_clip(punct_labels, pad_id), dtype=np.int32))
            capit_labels.append(pad_id)
            capit_all_labels.append(np.array(self.maybe_clip(capit_labels, pad_id), dtype=np.int32))
            progress_made += 1
            if progress_made >= TOKENIZATION_PROGRESS_REPORT_PERIOD:
                self.progress_queue.put(progress_made)
                progress_made = 0
        self.progress_queue.put(progress_made)
        if self.verbose:
            logging.info(f"Finished processing split with number {split_i}")
        return all_input_ids, all_subtokens_mask, sent_lengths, punct_all_labels, capit_all_labels


def tokenize_create_masks_clip_parallel(
    queries: List[str],
    max_seq_length: int,
    tokenizer: TokenizerSpec,
    punct_label_ids: Optional[Dict[str, int]],
    capit_label_ids: Optional[Dict[str, int]],
    punct_label_lines: Optional[Union[List[str], Tuple[str, ...]]],
    capit_label_lines: Optional[Union[List[str], Tuple[str, ...]]],
    pad_label: str,
    verbose: bool,
    n_jobs: Optional[int],
    progress_queue: Optional[mp.Queue],
) -> Tuple[List[ArrayLike], List[ArrayLike], List[int], List[ArrayLike], List[ArrayLike]]:
    """
    Tokenizes data, encodes labels, creates masks of first tokens in words, clips sequences by number of tokens.

    Args:
        queries: text sequences
        max_seq_length: max number of tokens in input sequence including [CLS] and [SEP] tokens. If number of tokens
            in a sequence exceeds ``max_seq_length``, then excess tokens in the end of the sequence are removed
        tokenizer: a tokenizer instance which has properties ``cls_id``, ``pad_id``, ``sep_id``, ``unk_id``
        punct_label_ids: dict to map punctuation labels to label ids.
            Starts with pad_label->0 and then increases in alphabetical order.
            Required for training and evaluation, not needed for inference.
        capit_label_ids: dict to map capitalization labels to label ids. Starts
            with pad_label->0 and then increases in alphabetical order.
            Required for training and evaluation, not needed for inference.
        pad_label: pad value use for labels. By default, it's the neutral label for punctuation and capitalization.
        punct_label_lines: list of labels for every word in a sequence (str)
        capit_label_lines: list of labels for every word in a sequence (str)
        verbose: whether to show examples of tokenized data and various progress information
        n_jobs: a number of workers used for preparing features. If ``n_jobs <= 0``, then do not use multiprocessing
            and run features creation in a calling process. If not set, number of workers will be equal to the number
            of CPUs.

            !!WARNING!!
            There can be deadlocking problems with some tokenizers (e.g. SentencePiece, HuggingFace AlBERT)
            if ``n_jobs > 0``.

        progress_queue: a multiprocessing queue used for reporting progress. Useful for creating tarred dataset

    Returns:
        input_ids: a list of 1D int32 arrays. Each array contains token ids of corresponding query
        subtokens_mask: a list of 1D boolean arrays. An array element is ``True`` if corresponding token is the
            first token in a word
        sent_lengths: a list of sequences lengths. A sequence length is a length is a length of corresponding
            ``input_ids`` element
        punct_labels: a list of 1D int32 arrays. Encoded punctuation labels for every token in a query. Tokens in one
            word have identical labels. If ``with_label`` is ``False``, then ``punct_labels`` is an empty list
        capit_labels: a list of 1D int32 arrays. Encoded capitalization labels for every token in a query. Tokens in
            one word have identical labels. If ``with_label`` is ``False``, then ``capit_labels`` is an empty list
    """
    create_progress_process = progress_queue is None
    if n_jobs is None:
        n_jobs = min(mp.cpu_count(), len(queries))
    if verbose:
        logging.info(f"Running tokenization with {n_jobs} jobs.")

    # Number of queries in split
    split_size = min(len(queries) // max(n_jobs, 1), MAX_NUM_QUERIES_IN_SPLIT)
    n_split = len(queries) // split_size
    split_queries = [queries[split_size * i : split_size * (i + 1)] for i in range(n_split - 1)] + [
        queries[split_size * (n_split - 1) :]
    ]
    split_punct_labels_lines = [
        punct_label_lines[split_size * i : split_size * (i + 1)] for i in range(n_split - 1)
    ] + [punct_label_lines[split_size * (n_split - 1) :]]
    split_capit_labels_lines = [
        capit_label_lines[split_size * i : split_size * (i + 1)] for i in range(n_split - 1)
    ] + [capit_label_lines[split_size * (n_split - 1) :]]
    args = list(zip(split_queries, split_punct_labels_lines, split_capit_labels_lines, range(n_split)))
    if create_progress_process:
        progress = Progress(len(queries), "Tokenization", "query")
        progress_queue = progress.get_queues()[0]
    if n_jobs > 0:
        with mp.Pool(n_jobs) as pool:
            result = pool.starmap(
                TokenizeCreateMasksClipWorker(
                    max_seq_length, tokenizer, punct_label_ids, capit_label_ids, pad_label, verbose, progress_queue
                ),
                args,
            )
    else:
        result = []
        for x in args:
            result.append(
                TokenizeCreateMasksClipWorker(
                    max_seq_length, tokenizer, punct_label_ids, capit_label_ids, pad_label, verbose, progress_queue,
                )(*x)
            )
    if create_progress_process:
        progress.finish()
    result = tuple(list(itertools.chain(*e)) for e in zip(*result))
    assert len(result) == 5
    return result


def get_features(
    queries: Union[List[str], Tuple[str, ...]],
    punct_label_lines: Union[List[str], Tuple[str, ...]],
    capit_label_lines: Union[List[str], Tuple[str, ...]],
    max_seq_length: int,
    tokenizer: TokenizerSpec,
    punct_label_ids: Dict[str, int] = None,
    capit_label_ids: Dict[str, int] = None,
    pad_label: str = 'O',
    verbose: bool = True,
    n_jobs: Optional[int] = 0,
    progress_queue: Optional[mp.Queue] = None,
) -> Tuple[List[ArrayLike], List[ArrayLike], List[ArrayLike], List[ArrayLike]]:
    """
    Tokenizes data, encodes labels, creates masks of first tokens in words, clips sequences by number of tokens.

    Args:
        queries: text sequences
        max_seq_length: max number of tokens in input sequence including [CLS] and [SEP] tokens. If number of tokens
            in a sequence exceeds ``max_seq_length``, then excess tokens in the end of the sequence are removed
        tokenizer: a tokenizer instance which has properties ``cls_id``, ``pad_id``, ``sep_id``, ``unk_id``
        punct_label_ids: dict to map punctuation labels to label ids.
            Starts with pad_label->0 and then increases in alphabetical order.
            Required for training and evaluation, not needed for inference.
        capit_label_ids: dict to map capitalization labels to label ids. Starts
            with pad_label->0 and then increases in alphabetical order.
            Required for training and evaluation, not needed for inference.
        pad_label: pad value use for labels. By default, it's the neutral label for punctuation and capitalization.
        punct_label_lines: a list of a tuple of labels for every word in a sequence (str)
        capit_label_lines: a list or a tuple of labels for every word in a sequence (str)
        verbose: whether to show examples of tokenized data and various progress information
        n_jobs: a number of workers used for preparing features. If ``n_jobs <= 0``, then do not use multiprocessing
            and run features creation in this process. If not set, number of workers will be equal to the number of
            CPUs
        progress_queue: a multiprocessing queue used for reporting progress. Useful for creating tarred dataset

    Returns:
        input_ids: a list of 1D int32 arrays. Each array contains token ids of corresponding query
        subtokens_mask: a list of 1D boolean arrays. An array element is ``True`` if corresponding token is the
            first token in a word
        punct_labels: a list of 1D int32 arrays. Encoded punctuation labels for every token in a query. Tokens in one
            word have identical labels.
        capit_labels: a list of 1D int32 arrays. Encoded capitalization labels for every token in a query. Tokens in
            one word have identical labels
    """
    if verbose:
        logging.info("Start initial tokenization.")
    input_ids, subtokens_mask, sent_lengths, punct_labels, capit_labels = tokenize_create_masks_clip_parallel(
        queries,
        max_seq_length,
        tokenizer,
        punct_label_ids,
        capit_label_ids,
        punct_label_lines,
        capit_label_lines,
        pad_label,
        verbose,
        n_jobs,
        progress_queue,
    )
    if verbose:
        logging.info("Finished initial tokenization.")
        get_stats(sent_lengths)
        logging.info(f"Finished clipping and padding.")
        for i in range(min(len(input_ids), 5)):
            logging.info("*** Example ***")
            logging.info("i: %s" % (i))
            logging.info("subtokens: %s" % " ".join(list(map(str, input_ids[i]))))
            logging.info("subtokens_mask: %s" % " ".join(list(map(str, subtokens_mask[i]))))
            logging.info("punct_labels: %s" % " ".join(list(map(str, punct_labels[i]))))
            logging.info("capit_labels: %s" % " ".join(list(map(str, capit_labels[i]))))
    return input_ids, subtokens_mask, punct_labels, capit_labels


def create_masks_and_segment_ids(
    input_ids: ArrayLike,
    subtokens_mask: ArrayLike,
    pad_id: int,
    cls_id: int,
    sep_id: int,
    ignore_start_end: bool,
    ignore_extra_tokens: bool,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Creates segment ids array, input mask, loss mask.

    Segment ids array is token type ids for BERT in HuggingFace terminology and it is a zeros array for punctuation
    and capitalization task.

    Input mask element is ``True`` if an element of ``input_ids`` is not padding and ``False`` otherwise.

    Loss mask element is always ``True`` for the first token in a word. If ``ignore_start_end=False``, then loss mask
    element is ``True`` for [CLS] and [SEP] tokens. If ``ignore_extra_tokens=False``, then loss mask element is ``True``
    for all word tokens. In all other cases loss mask elements are ``False``.

    Args:
        input_ids: an integer array of shape ``[Batch, Time]`` containing ids of source token ids
        subtokens_mask: a boolean array of shape ``[Batch, Time]`` which elements are ``True`` if they correspond to
            the first token of some word
        pad_id: an id of padding token
        cls_id: an id of [CLS] token
        sep_id: an id of [SEP] token
        ignore_start_end: whether to compute loss for [CLS] and [SEP] tokens
        ignore_extra_tokens: whether to compute loss for not first tokens in words

    Returns:
        segment_ids: int8 array of shape [Batch, Time]
        input_mask: boolean array of shape [Batch, Time]
        loss_mask: boolean array of shape [Batch, Time]
    """
    segment_ids = np.zeros_like(input_ids, dtype=np.int8)
    input_mask = np.not_equal(input_ids, pad_id)
    special_mask = np.equal(input_ids, cls_id) & np.equal(input_ids, sep_id)
    if ignore_start_end:
        if ignore_extra_tokens:
            loss_mask = subtokens_mask
        else:
            loss_mask = input_mask & ~special_mask
    else:
        if ignore_extra_tokens:
            loss_mask = subtokens_mask | special_mask
        else:
            loss_mask = input_mask
    return segment_ids, input_mask, loss_mask


def create_label_ids(unique_labels: Set[str], pad_label: str) -> Dict[str, int]:
    label_ids = {pad_label: 0}
    if pad_label in unique_labels:
        unique_labels.remove(pad_label)
    for label in sorted(unique_labels):
        label_ids[label] = len(label_ids)
    return label_ids


def load_label_ids(file_path: Path) -> Dict[str, int]:
    ids = {}
    with file_path.open() as f:
        for i, line in enumerate(f):
            ids[line.strip()] = i
    return ids


class BertPunctuationCapitalizationDataset(Dataset):
    """
    Creates dataset to use during training for punctuation and capitalization tasks with a pretrained model.
    For dataset to use during inference without labels, see ``BertPunctuationCapitalizationInferDataset``.

    Args:
        text_file: file to sequences, each line should a text without punctuation and capitalization
        label_file: file to labels, each line corresponds to word labels for a sentence in the text_file
        max_seq_length: max number of tokens in a source sequence. ``max_seq_length`` includes for [CLS] and [SEP]
            tokens. Sequences which are too long will be clipped by removal of tokens from the end of the sequence
        tokenizer: a tokenizer instance which has properties ``unk_id``, ``sep_id``, ``bos_id``, ``eos_id``
        num_samples: number of samples you want to use for the dataset. If ``-1``, use all dataset. Useful for testing.
        pad_label: pad value use for labels. It's the neutral label both for punctuation and capitalization.
        punct_label_ids and capit_label_ids: dict to map labels to label ids. Starts with pad_label->0 and then
            increases in alphabetical order. For dev set use label_ids generated during training to support cases when
            not all labels are present in the dev set. For training, it is recommended to set ``punct_label_ids`` to
            ``None`` or load from cache
        ignore_extra_tokens: whether to ignore extra tokens in the loss_mask
        ignore_start_end: whether to ignore bos and eos tokens in the loss_mask
        use_cache: whether to use processed data cache or not
        get_label_frequencies: whether to show label frequencies. Works if ``verbose`` parameter is ``True``
        punct_label_vocab_file and capit_label_vocab_file: paths to .csv files containing punctuation and
            capitalization label vocabularies correspondingly. Each line in such a vocabulary file contains exactly
            one label. The first line has to contain `pad_label`, otherwise error will be raised.
        add_masks_and_segment_ids_to_batch: whether to add ``loss_mask``, ``input_mask``, ``segment_ids`` to batch.
            Useful for creation of tarred dataset and can NOT be used during model training and inference
        verbose: whether to show data examples, label stats and other useful information
        save_label_ids: whether to save punctuation and capitalization label ids into files ``punct_label_ids`` and
            ``capit_label_ids``
        n_jobs: number of workers used for tokenization, encoding labels, creating "first token in word" mask, and
            clipping. If ``n_jobs <= 0`` data preparation is performed without multiprocessing. By default ``n_jobs``
            is equal to the number of CPUs.

            !!WARNING!!
            There can be deadlocking problems with some tokenizers (e.g. SentencePiece, HuggingFace AlBERT)
            if ``n_jobs > 0``.

        tokenization_progress_queue: a queue for reporting tokenization progress. Useful for creation of tarred dataset
        batch_mark_up_progress_queue: a queue for reporting progress in deciding which samples batches will contain
            Useful for creation of tarred dataset
        batch_building_progress_queue: a queue for reporting progress in batch creation (stacking and padding). Useful
            for creation of tarred dataset
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports. """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'subtokens_mask': NeuralType(('B', 'T'), MaskType()),
            'loss_mask': NeuralType(('B', 'T'), MaskType()),
            'punct_labels': NeuralType(('B', 'T'), LabelsType()),
            'capit_labels': NeuralType(('B', 'T'), LabelsType()),
        }

    def __init__(
        self,
        text_file: Union[str, os.PathLike],
        label_file: Union[str, os.PathLike],
        max_seq_length: int,
        tokenizer: TokenizerSpec,
        num_samples: int = -1,
        tokens_in_batch: int = DEFAULT_TOKENS_IN_BATCH,
        pad_label: str = 'O',
        punct_label_ids: Optional[Dict[str, int]] = None,
        capit_label_ids: Optional[Dict[str, int]] = None,
        ignore_extra_tokens: bool = False,
        ignore_start_end: bool = False,
        use_cache: bool = True,
        get_label_frequencies: bool = False,
        punct_label_vocab_file: Union[str, os.PathLike] = None,
        capit_label_vocab_file: Union[str, os.PathLike] = None,
        add_masks_and_segment_ids_to_batch: bool = True,
        verbose: bool = True,
        save_label_ids: bool = True,
        n_jobs: Optional[int] = 0,
        tokenization_progress_queue: Optional[mp.Queue] = None,
        batch_mark_up_progress_queue: Optional[mp.Queue] = None,
        batch_building_progress_queue: Optional[mp.Queue] = None,
    ):
        """ Initializes BertPunctuationCapitalizationDataset. """

        if not (os.path.exists(text_file) and os.path.exists(label_file)):
            raise FileNotFoundError(
                f'{text_file} or {label_file} not found. The data should be split into 2 files: text.txt and'
                f'labels.txt. Each line of the text.txt file contains text sequences, where words are separated with'
                f'spaces. The labels.txt file contains corresponding labels for each word in text.txt, the labels are'
                f'separated with spaces. Each line of the files should follow the format:\n'
                f'   [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and '
                f'   [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).'
            )
        if punct_label_ids is not None and punct_label_vocab_file is not None:
            raise ValueError(
                f'You can provide at most one of parameters `punct_label_ids` and `punct_label_vocab_file` whereas '
                f'both parameters are not `None`.'
            )
        if capit_label_ids is not None and capit_label_vocab_file is not None:
            raise ValueError(
                f'You can provide at most one of parameters `capit_label_ids` and `capit_label_vocab_file` whereas '
                f'both parameters are not `None`.'
            )
        if punct_label_vocab_file is not None:
            punct_label_vocab_file = Path(punct_label_vocab_file).expanduser()
            punct_label_ids = load_label_ids(punct_label_vocab_file)
        if capit_label_vocab_file is not None:
            capit_label_vocab_file = Path(capit_label_vocab_file).expanduser()
            capit_label_ids = load_label_ids(capit_label_vocab_file)
        # Cache features
        text_file, label_file = Path(text_file), Path(label_file)
        data_dir = text_file.parent
        filename = text_file.name

        if not filename.endswith('.txt'):
            raise ValueError("{text_file} should have extension .txt")

        self.tokens_in_batch = tokens_in_batch
        self.tokenizer = tokenizer
        self.pad_label = pad_label
        self.ignore_extra_tokens = ignore_extra_tokens
        self.ignore_start_end = ignore_start_end
        self.add_masks_and_segment_ids_to_batch = add_masks_and_segment_ids_to_batch
        self.verbose = verbose
        self.batch_mark_up_progress_queue = batch_mark_up_progress_queue
        self.batch_building_progress_queue = batch_building_progress_queue
        filename = filename[:-4]
        vocab_size = getattr(self.tokenizer, "vocab_size", 0)
        features_pkl = data_dir / "cached_{}_{}_{}_{}_{}".format(
            filename, self.tokenizer.name, str(max_seq_length), str(vocab_size), str(num_samples)
        )
        for_nemo = "label_id_files_for_nemo_checkpoint"
        self.punct_label_ids_file = data_dir / for_nemo / DEFAULT_PUNCT_LABEL_IDS_NAME
        self.capit_label_ids_file = data_dir / for_nemo / DEFAULT_CAPIT_LABEL_IDS_NAME

        master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        cache_files_exist = all(
            [features_pkl.is_file(), self.punct_label_ids_file.is_file(), self.capit_label_ids_file.is_file()]
        )
        features = None
        if master_device and not (cache_files_exist and use_cache):
            if num_samples == 0:
                raise ValueError(f"Parameter `num_samples` has to be positive whereas `num_samples={num_samples}`")
            if verbose:
                logging.info(f'Processing {text_file}')
            res = self.read_dataset(text_file, label_file, num_samples, verbose)
            text_lines, punct_label_lines, capit_label_lines, punct_unique_labels, capit_unique_labels = res

            # for dev/test sets use label mapping from training set
            if punct_label_ids:
                self.check_label_ids(punct_label_ids, punct_unique_labels, 'punct', 'punctuation', label_file)
            else:
                punct_label_ids = create_label_ids(punct_unique_labels, self.pad_label)
            if capit_label_ids:
                self.check_label_ids(capit_label_ids, capit_unique_labels, 'capit', 'capitalzation', label_file)
            else:
                capit_label_ids = create_label_ids(capit_unique_labels, self.pad_label)
            if save_label_ids:
                self._save_label_ids(punct_label_ids, self.punct_label_ids_file)
                self._save_label_ids(capit_label_ids, self.capit_label_ids_file)
            features = get_features(
                text_lines,
                punct_label_lines,
                capit_label_lines,
                max_seq_length,
                self.tokenizer,
                pad_label=self.pad_label,
                punct_label_ids=punct_label_ids,
                capit_label_ids=capit_label_ids,
                verbose=self.verbose,
                progress_queue=tokenization_progress_queue,
                n_jobs=n_jobs,
            )
            if use_cache:
                pickle.dump(tuple(list(features) + [punct_label_ids, capit_label_ids]), open(features_pkl, "wb"))
                if self.verbose:
                    logging.info(f'Features saved to {features_pkl}')

        # wait until the master process writes to the processed data files
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if features is None:
            features = pickle.load(open(features_pkl, 'rb'))
            punct_label_ids, capit_label_ids = features[-2], features[-1]
            if tokenization_progress_queue is not None:
                tokenization_progress_queue.put(len(features[0]))
            if self.verbose:
                logging.info(f'Features restored from {features_pkl}')

        self.input_ids, self.subtokens_mask, self.punct_labels, self.capit_labels = features[:-2]
        self.punct_label_ids, self.capit_label_ids = punct_label_ids, capit_label_ids
        self.batches = self.pack_into_batches(
            self.input_ids, self.subtokens_mask, self.punct_labels, self.capit_labels
        )

        if get_label_frequencies:
            self.punct_label_frequencies = self._calculate_label_frequencies(self.punct_labels, data_dir, 'punct')
            self.capit_label_frequencies = self._calculate_label_frequencies(self.capit_labels, data_dir, 'capit')

    @staticmethod
    def check_label_ids(
        label_ids: Dict[str, int], unique_labels: Set[str], label_type: str, task: str, label_file: Path
    ):
        if unique_labels - set(label_ids):
            not_present_labels = list(unique_labels - set(label_ids))
            raise ValueError(
                f"{len(not_present_labels)} {task} labels found in {label_file} are not present in "
                f"`{label_type}_label_ids`. Examples of unexpected labels from {label_file}: {not_present_labels[:3]}"
            )

    @staticmethod
    def read_dataset(
        text_file: Path, label_file: Path, num_samples: int, verbose: bool
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Set[str], Set[str]]:
        if num_samples <= 0:
            raise ValueError(f"Parameter `num_samples` has to be positive whereas `num_samples={num_samples}`")
        if verbose:
            logging.info(f'Processing {text_file}')
        with open(text_file, 'r') as f:
            text_lines = f.readlines()
        punct_unique_labels, capit_unique_labels = set(), set()
        punct_labels_lines, capit_labels_lines = [], []
        with label_file.open() as f:
            for i, line in enumerate(f):
                pairs = line.split()
                if not all([len(p) == 2 for p in pairs]):
                    raise ValueError(
                        f"Some label pairs are not pairs but have wrong length (!= 2) in line {i} in label file "
                        f"{label_file}"
                    )
                words = text_lines[i].split()
                if len(pairs) != len(words):
                    raise ValueError(
                        f"In line {i} number of words {len(words)} text file {text_file} is not equal to the number "
                        f"of labels {len(pairs)} in label file {label_file}."
                    )
                punct_line, capit_line = zip(*pairs)
                punct_labels_lines.append(punct_line)
                capit_labels_lines.append(capit_line)
                punct_unique_labels.update(punct_line)
                capit_unique_labels.update(capit_line)

        if len(punct_labels_lines) != len(text_lines):
            raise ValueError(
                f"Number of text lines {len(text_lines)} in text file {text_file} is not equal to the number of lines "
                f"{len(punct_labels_lines)} in label file {label_file}."
            )
        dataset = list(zip(text_lines, punct_labels_lines, capit_labels_lines))
        if len(dataset) == 0:
            raise ValueError(f"Dataset loaded from files {text_file} and {label_file} is empty.")
        if num_samples > 0:
            dataset = dataset[:num_samples]
        text_lines, punct_labels_lines, capit_labels_lines = zip(*dataset)
        return text_lines, punct_labels_lines, capit_labels_lines, punct_unique_labels, capit_unique_labels

    @staticmethod
    def pad(vectors: List[ArrayLike], length: int, value: Union[int, float, bool]) -> ArrayLike:
        """
        Pad vectors to length ``length`` and then stack.
        Args:
            vectors: a list of 1D arrays. Arrays to pad and stack
            length: a length of padded sequence. Has to be greater or equal to the maximum length of an element of
                ``vectors``.
            value: a values used for padding

        Returns:
            an array of padded vectors
        """
        result = []
        for v in vectors:
            result.append(np.concatenate([v, np.full([length - v.shape[0]], value, dtype=v.dtype)]))
        return np.stack(result)

    def mark_up_batches(self, input_ids: List[ArrayLike]) -> Tuple[List[int], List[int], List[int]]:
        """
        Computes indices of first samples in batch, batch sizes, seq lengths for batches. ``input_ids`` has to be
        sorted by number of tokens in ascending order.

        Batches are marked up with respect to following conditions:
            - total number of tokens in batch including paddings is less or equal to ``self.tokens_in_batch``
            - batch size is evenly divisible by 8 (except for the last batch)
            - seq length (elements of the third returned object) is evenly divisible by 8

        If ``self.batch_mark_up_progress_queue`` is not None, then the progress in mark up is reported via
        ``self.batch_mark_up_progress_queue``. Otherwise, ``tqdm`` instance is created in this function.

        Args:
            input_ids: a list of 1D int32 arrays. Elements of ``input_ids`` have to be sorted by length in ascending
                order

        Returns:
            batch_beginnings: a list of indices in ``input_ids`` of first samples of every batch
            batch_sizes: a list of numbers of samples in batches
            batch_seq_lengths: a list of sequence lengths after padding for every batch
        """
        batch_beginnings, batch_sizes, batch_seq_lengths = [], [], []
        current_max_length = 0
        start = 0
        if self.batch_mark_up_progress_queue is None:
            inp_iterator = tqdm(enumerate(input_ids), total=len(input_ids), desc="Batch mark up", unit="query")
        else:
            inp_iterator = enumerate(input_ids)
            progress_made = 0
        for i, inp in inp_iterator:
            current_max_length = max(current_max_length, ceil(len(inp) / 8) * 8)
            if current_max_length * (i + 1 - start) > self.tokens_in_batch:
                batch_size = (i - start) // 8 * 8
                if batch_size == 0:
                    if i > start:
                        batch_size = i - start
                        logging.warning(
                            f"Could not create batch with multiple of 8 size. Probably there is a too long sequence in "
                            f"the dataset. current_max_length={current_max_length}. Batch size will be reduced to "
                            f"{batch_size}. tokens_in_batch={self.tokens_in_batch}. The batch includes sequences from "
                            f"{start} to {i - 1}."
                        )
                    else:
                        logging.warning(
                            f"Input sequence number {i - 1} is too long. Could not fit it into batch with "
                            f"{self.tokens_in_batch} tokens. Sequence number {i - 1} will not be added to batches."
                        )
                        start = i
                        current_max_length = ceil(len(inp) / 8) * 8
                        continue
                seq_length = ceil(max([len(inp) for inp in input_ids[start : start + batch_size]]) / 8) * 8
                batch_beginnings.append(start)
                batch_sizes.append(batch_size)
                batch_seq_lengths.append(seq_length)
                start += batch_size
                current_max_length = ceil(max([len(inp) for inp in input_ids[start : i + 1]]) / 8) * 8
            if self.batch_mark_up_progress_queue is not None:
                progress_made += 1
                if progress_made >= BATCH_MARK_UP_PROGRESS_REPORT_PERIOD:
                    self.batch_mark_up_progress_queue.put(progress_made)
                    progress_made = 0
        if start < len(input_ids):
            seq_length = ceil(max([len(inp) for inp in input_ids[start:]]) / 8) * 8
            batch_beginnings.append(start)
            batch_sizes.append(len(input_ids) - start)
            batch_seq_lengths.append(seq_length)
            if self.batch_mark_up_progress_queue is not None:
                self.batch_mark_up_progress_queue.put(progress_made)
        assert sum(batch_sizes) == len(input_ids)
        for i in range(len(batch_beginnings) - 1):
            assert batch_beginnings[i] + batch_sizes[i] == batch_beginnings[i + 1]
            assert batch_seq_lengths[i] >= max(
                [len(inp) for inp in input_ids[batch_beginnings[i] : batch_beginnings[i] + batch_sizes[i]]]
            )
        return batch_beginnings, batch_sizes, batch_seq_lengths

    def pack_into_batches(
        self,
        input_ids: List[ArrayLike],
        subtokens_mask: List[ArrayLike],
        punct_labels: List[ArrayLike],
        capit_labels: List[ArrayLike],
    ) -> List[Dict[str, ArrayLike]]:
        """
        Shuffle input sequences, sort them by number of tokens, pad, and pack into batches which satisfy following
        conditions:
            - total number of tokens in batch including paddings is less or equal to ``self.tokens_in_batch``
            - batch size is evenly divisible by 8 (except for the last batch)
            - seq length (elements of the third returned object) is evenly divisible by 8
        Created batches are shuffled before returning.

        If ``self.add_masks_and_segment_ids_to_batch`` is ``True``, then ``'segment_ids'``, ``'loss_mask'``, and
        ``'input_mask'`` are added to the batch.

        If ``self.batch_building_progress_queue`` is not ``None``, then padding progress is reported to
        ``self.batch_building_progress_queue``. Otherwise, a new ``tqdm`` instance is created in ``pack_into_batches``
        method.

        Args:
            input_ids: a list of 1D int32 arrays which contain token ids of dataset source
            subtokens_mask: a list of 1D boolean arrays which elements are ``True`` if corresponding token is the
                first token in some word
            punct_labels: a list of 1D int32 arrays which contain encoded punctuation labels
            capit_labels: a list of 1D int32 arrays which contain encoded capitalization labels

        Returns:
            a list of batches. Each batch is a dictionary with items:
              - ``'input_ids'``: a ``np.int32`` numpy array;
              - ``'subtokens_mask'``: a boolean numpy array;
              - ``'punct_labels'``: a ``np.int32`` numpy array;
              - ``'capit_labels'``: a ``np.int32`` numpy array.
            If ``self.add_masks_and_segment_ids_to_batch`` is ``True``, then a batch also contain items
              - ``'segment_ids'``: a ``np.int8`` numpy array;
              - ``'input_mask'``: a boolean numpy array;
              - ``'loss_mask'``: a boolean numpy array.

            The values of a batch dictionary are numpy arrays of identical shape.
        """
        zipped = list(zip(input_ids, subtokens_mask, punct_labels, capit_labels))
        random.shuffle(zipped)
        input_ids, subtokens_mask, punct_labels, capit_labels = zip(*sorted(zipped, key=lambda x: x[0].shape[0]))
        batch_beginnings, batch_sizes, batch_seq_lengths = self.mark_up_batches(input_ids)
        batches = []
        if self.batch_building_progress_queue is None:
            inp_iterator = tqdm(
                zip(batch_beginnings, batch_sizes, batch_seq_lengths),
                total=len(batch_beginnings),
                desc="Batch building",
                unit="batch",
            )
        else:
            # In this case we report number of queries not number of batches
            inp_iterator = zip(batch_beginnings, batch_sizes, batch_seq_lengths)
            progress_made = 0
        for start, size, length in inp_iterator:
            batch_input_ids = self.pad(input_ids[start : start + size], length, self.tokenizer.pad_id)
            batch_subtokens_mask = self.pad(subtokens_mask[start : start + size], length, False)
            batch = {
                "input_ids": batch_input_ids,
                "subtokens_mask": batch_subtokens_mask,
                "punct_labels": self.pad(
                    punct_labels[start : start + size], length, self.punct_label_ids[self.pad_label]
                ).astype(np.int64),
                "capit_labels": self.pad(
                    capit_labels[start : start + size], length, self.capit_label_ids[self.pad_label]
                ).astype(np.int64),
            }
            if self.add_masks_and_segment_ids_to_batch:
                batch_segment_ids, batch_input_mask, batch_loss_mask = create_masks_and_segment_ids(
                    batch_input_ids,
                    batch_subtokens_mask,
                    self.tokenizer.pad_id,
                    self.tokenizer.cls_id,
                    self.tokenizer.sep_id,
                    self.ignore_start_end,
                    self.ignore_extra_tokens,
                )
                batch['segment_ids'] = batch_segment_ids
                batch['input_mask'] = batch_input_mask
                batch['loss_mask'] = batch_loss_mask
            batches.append(batch)
            if self.batch_building_progress_queue is not None:
                progress_made += size
                if progress_made >= BATCH_BUILDING_PROGRESS_REPORT_PERIOD:
                    self.batch_building_progress_queue.put(progress_made)
                    progress_made = 0
        if self.batch_building_progress_queue is not None:
            self.batch_building_progress_queue.put(progress_made)
        random.shuffle(batches)
        return batches

    def repack_batches_with_shuffle(self):
        logging.info("Shuffling training dataset")
        self.batches = self.pack_into_batches(
            self.input_ids, self.subtokens_mask, self.punct_labels, self.capit_labels
        )

    def _calculate_label_frequencies(self, all_labels: List[ArrayLike], data_dir: Path, name: str) -> Dict[str, float]:
        """ Calculates labels frequencies """
        merged_labels = itertools.chain.from_iterable(all_labels)
        if self.verbose:
            logging.info('Three most popular labels')
        _, label_frequencies, _ = get_label_stats(merged_labels, str(data_dir / f'label_count_{name}.tsv'))
        return label_frequencies

    def _save_label_ids(self, label_ids: Dict[str, int], filename: Path):
        """ Saves label ids map to a file """
        with filename.open('w') as out:
            labels, _ = zip(*sorted(label_ids.items(), key=lambda x: x[1]))
            out.write('\n'.join(labels))
            if self.verbose:
                logging.info(f'Labels: {label_ids}')
                logging.info(f'Labels mapping saved to : {out.name}')

    def __len__(self) -> int:
        return len(self.batches)

    def collate_fn(self, batches: List[Dict[str, ArrayLike]]) -> Dict[str, torch.Tensor]:
        """
        Return zeroth batch of ``batches`` passed for collating and casts ``'segment_ids'``, ``'punct_labels'``,
        ``'capit_labels'`` to types supported by ``PunctuationCapitalizationModel``.

        Note: batch size in data loader and sampler has to be 1.
        Args:
            batches: a list of batches passed for collating. Normally ``batches`` contains exactly 1 element

        Returns:
            a batch dictionary with following items:
              - ``'input_ids'``: ``torch.int32`` tensor,
              - ``'subtokens_mask'``: ``torch.bool`` tensor,
              - ``'punct_labels'``: ``torch.int64`` tensor,
              - ``'capit_labels'``: ``torch.int64`` tensor.
              - ``'segment_ids'``: ``torch.int32`` tensor,
              - ``'input_mask'``: ``torch.bool`` tensor,
              - ``'loss_mask'``: ``torch.bool`` tensor.
        """
        batch = {k: torch.as_tensor(v) for k, v in batches[0].items()}
        batch['segment_ids'] = batch['segment_ids'].int()
        batch['punct_labels'] = batch['punct_labels'].long()
        batch['capit_labels'] = batch['capit_labels'].long()
        return batch

    def __getitem__(self, idx: int) -> Dict[str, ArrayLike]:
        """
        Return a batch with index ``idx``.

        Args:
            idx: an index of returned batch

        Returns:
            a dictionary with items:
              - ``'input_ids'``: ``np.int32`` array,
              - ``'subtokens_mask'``: ``bool`` array,
              - ``'punct_labels'``: ``np.int32`` array,
              - ``'capit_labels'``: ``np.int32`` array.
            If ``self.add_masks_and_segment_ids_to_batch`` is ``True``, then a batch also contain items
              - ``'segment_ids'``: ``np.int8`` array,
              - ``'input_mask'``: ``bool`` array,
              - ``'loss_mask'``: ``bool`` array.

            The values of a batch dictionary are numpy arrays of identical shapes.
        """
        return self.batches[idx]
