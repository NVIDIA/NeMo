# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import json
import multiprocessing as mp
import os
import pickle
import re
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import webdataset as wds
from joblib import Parallel, delayed
from torch.utils.data import IterableDataset

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset import (
    BertPunctuationCapitalizationDataset, Progress, create_masks_and_segment_ids
)
from nemo.utils import logging


NUMBER_RE = "(0|[1-9][0-9]*)"
TAR_FRAGMENT_TMPL_1 = "fragment{}.{}.tar"
TAR_FRAGMENT_TMPL_2 = "fragment{}.num_batches{}.{}.tar"
TAR_FRAGMENT_PATTERN_1 = re.compile(f"fragment{NUMBER_RE}.{NUMBER_RE}.tar$")
TAR_FRAGMENT_PATTERN_2 = re.compile(f"fragment{NUMBER_RE}.num_batches{NUMBER_RE}.{NUMBER_RE}.tar$")
EXTRACT_NUM_BATCHES_PATTERN = re.compile(r"fragment\d+.num_batches(\d+).\d+.tar")

DATASET_PARAMETERS_TMPL = "{prefix}.tokens{tokens_in_batch}.max_seq_length{max_seq_length}.{tokenizer}"
TAR_FINAL_TMPL = ".batches{num_batches}.{ctr}.tar"

WRITING_DATASET_PROGRESS_REPORT_PERIOD = 10 ** 4


def count_lines_and_get_fragment_starting_positions(file_name: os.PathLike, lines_per_dataset_fragment: int):
    pos = [0]
    with file_name.open() as f:
        i = 0
        line = f.readline()
        while line:
            i += 1
            if i % lines_per_dataset_fragment == 0:
                pos.append(f.tell())
            line = f.readline()
    return i, pos[:-1] if i % lines_per_dataset_fragment == 0 else pos


def process_fragment(
    text_file: os.PathLike,
    labels_file: os.PathLike,
    output_dir: os.PathLike,
    text_start_pos: int,
    label_start_pos: int,
    lines_per_dataset_fragment: int,
    max_seq_length: int,
    tokens_in_batch: int,
    num_batches_per_tarfile: int,
    tokenizer: TokenizerSpec,
    fragment_idx: int,
    tokenization_progress_queue: mp.Queue,
    batch_mark_up_progress_queue: mp.Queue,
    batch_building_progress_queue: mp.Queue,
    writing_to_tar_progress_queue: mp.Queue,
):
    tmp_text = output_dir / f'tmp_text_{fragment_idx}.txt'
    tmp_labels = output_dir / f'tmp_labels_{fragment_idx}.txt'
    with text_file.open() as tf, labels_file.open() as lf, tmp_text.open('w') as otf, tmp_labels.open('w') as olf:
        tf.seek(text_start_pos)
        lf.seek(label_start_pos)
        for _ in range(lines_per_dataset_fragment):
            text_line = tf.readline()
            if not text_line:
                break
            otf.write(text_line)
            olf.write(lf.readline())
    dataset = BertPunctuationCapitalizationDataset(
        tmp_text,
        tmp_labels,
        max_seq_length,
        tokenizer,
        tokens_in_batch=tokens_in_batch,
        njobs=0,
        use_cache=False,
        add_masks_and_segment_ids_to_batch=False,
        verbose=False,
        pickle_features=False,
        save_label_ids=fragment_idx == 0,
        tokenization_progress_queue=tokenization_progress_queue,
        batch_mark_up_progress_queue=batch_mark_up_progress_queue,
        batch_building_progress_queue=batch_building_progress_queue,
    )
    tmp_text.unlink()
    tmp_labels.unlink()
    tar_ctr = 0
    current_file_name = output_dir / TAR_FRAGMENT_TMPL_1.format(fragment_idx, tar_ctr)
    current_num_batches = 0
    sink = wds.TarWriter(str(current_file_name))
    progress_made = 0
    for batch_i, batch in enumerate(dataset):
        if batch_i % num_batches_per_tarfile == 0 and batch_i > 0:
            sink.close()
            current_file_name.rename(
                output_dir / TAR_FRAGMENT_TMPL_2.format(fragment_idx, current_num_batches, tar_ctr)
            )
            writing_to_tar_progress_queue.put(progress_made)
            progress_made = 0
            tar_ctr += 1
            current_file_name = output_dir / TAR_FRAGMENT_TMPL_1.format(fragment_idx, tar_ctr)
            current_num_batches = 0
            sink = wds.TarWriter(str(current_file_name))
        sink.write({"__key__": f"fragment-{fragment_idx}-batch-{batch_i}", "batch.pyd": batch})
        current_num_batches += 1
        progress_made += len(batch['input_ids'])
    sink.close()
    writing_to_tar_progress_queue.put(progress_made)
    new_file_name = output_dir / TAR_FRAGMENT_TMPL_2.format(fragment_idx, current_num_batches, tar_ctr)
    current_file_name.rename(new_file_name)
    if progress_made > 0:
        new_file_name.unlink()


def remove_unexpected_files(output_dir: os.PathLike, output_file_tmpl: str, metadata_file_name: os.PathLike):
    if not output_dir.is_dir():
        return
    tar_final_pattern = re.compile(output_file_tmpl.format(ctr=NUMBER_RE, num_batches=NUMBER_RE))
    unexpected_tar_files = [
        path for path in output_dir.iterdir()
        if any(
            [
                p.match(path.name) is not None
                for p in [TAR_FRAGMENT_PATTERN_1, TAR_FRAGMENT_PATTERN_2, tar_final_pattern]
            ]
        )
    ]
    if unexpected_tar_files:
        logging.warning(
            f"Found {len(unexpected_tar_files)} unexpected tar files in the output directory {output_dir}. "
            f"All of them are going to be removed. The files match one of 3 patterns: "
            f"'{TAR_FRAGMENT_PATTERN_1.pattern}', '{TAR_FRAGMENT_PATTERN_2.pattern}', "
            f"'{tar_final_pattern.pattern}'. The first 3 unexpected files: {unexpected_tar_files[:3]}"
        )
        for fn in unexpected_tar_files:
            fn.unlink()
    if metadata_file_name.is_file():
        logging.warning(f"Found metadata file {metadata_file_name}. It is going to be removed.")
        metadata_file_name.unlink()


def create_tarred_dataset(
    text_file: Union[os.PathLike, str],
    label_file: Union[os.PathLike, str],
    output_dir: Union[os.PathLike, str],
    max_seq_length: int,
    tokens_in_batch: int,
    lines_per_dataset_fragment: int,
    num_batches_per_tarfile: int,
    tokenizer_name: str,
    tokenizer_model: Optional[Union[os.PathLike, str]] = None,
    vocab_file: Optional[Union[os.PathLike, str]] = None,
    merges_file: Optional[Union[os.PathLike, str]] = None,
    special_tokens: Optional[Dict[str, str]] = None,
    use_fast_tokenizer: Optional[bool] = False,
    tokenizer_bpe_dropout: Optional[float] = 0.0,
    tar_file_prefix: Optional[str] = 'punctuation_capitalization',
    n_jobs: Optional[int] = mp.cpu_count(),
):
    text_file, label_file = Path(text_file).expanduser(), Path(label_file).expanduser()
    output_dir = Path(output_dir).expanduser()
    ds_params_str = DATASET_PARAMETERS_TMPL.format(
        prefix=tar_file_prefix,
        tokens_in_batch=tokens_in_batch,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer_name,
    )
    output_file_tmpl = ds_params_str + TAR_FINAL_TMPL
    metadata_file_name = output_dir / ('metadata.' + ds_params_str + '.json')
    remove_unexpected_files(output_dir, output_file_tmpl, metadata_file_name)
    logging.info(
        f"Counting lines in files {text_file} and {label_file} and creating segment borders. This may take "
        f"considerable time. 86GB, 1.27b lines file was processed in 7 minutes."
    )
    result = Parallel(n_jobs=2)(
        delayed(count_lines_and_get_fragment_starting_positions)(file_name, lines_per_dataset_fragment)
        for file_name in [text_file, label_file]
    )
    if result[0][0] != result[1][0]:
        raise ValueError(
            f"Text file {text_file} and label file {label_file} contain different number of lines. Number of lines "
            f"in text file: {result[0][0]}, number of lines in label file: {result[1][0]}."
        )
    num_lines = result[0][0]
    text_start_bytes, label_start_bytes = result[0][1], result[1][1]
    assert len(text_start_bytes) == len(label_start_bytes)
    if text_start_bytes:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        logging.warning(f"Both {label_file} and {text_file} are empty. Tarred dataset cannot be created.")
        return
    tokenizer_name = get_tokenizer(
        tokenizer_name,
        tokenizer_model=None if tokenizer_model is None else str(Path(tokenizer_model).expanduser()),
        vocab_file=None if vocab_file is None else str(Path(vocab_file).expanduser()),
        merges_file=None if merges_file is None else str(Path(merges_file).expanduser()),
        special_tokens=special_tokens,
        use_fast=use_fast_tokenizer,
        bpe_dropout=tokenizer_bpe_dropout,
    )
    with Progress(
        num_lines,
        ["Tokenization", "Batch mark up", "Batch building", "Writing tarred dataset"],
        "query"
    ) as progress_queues:
        Parallel(n_jobs=min(n_jobs, len(text_start_bytes)))(
            delayed(process_fragment)(
                text_file,
                label_file,
                output_dir,
                text_start_pos,
                label_start_pos,
                lines_per_dataset_fragment,
                max_seq_length,
                tokens_in_batch,
                num_batches_per_tarfile,
                tokenizer_name,
                fragment_idx,
                *progress_queues,
            ) for fragment_idx, (text_start_pos, label_start_pos) in enumerate(zip(text_start_bytes, label_start_bytes))
        )
    metadata = {"num_batches": 0, "tar_files": []}
    for i, fn in enumerate([fn for fn in output_dir.iterdir() if TAR_FRAGMENT_PATTERN_2.match(fn.name)]):
        nb = int(EXTRACT_NUM_BATCHES_PATTERN.match(fn.name).group(1))
        new_name = output_dir / output_file_tmpl.format(ctr=i, num_batches=nb)
        fn.rename(new_name)
        metadata['tar_files'].append(new_name.name)
        metadata["num_batches"] += nb
    with metadata_file_name.open('w') as f:
        json.dump(metadata, f, indent=2)


class BertPunctuationCapitalizationTarredDataset(IterableDataset):
    def __init__(
        self,
        metadata_file: Union[os.PathLike, str],
        tokenizer: TokenizerSpec,
        pad_label: str,
        punct_label_ids_file: str = 'punct_label_ids.csv',
        capit_label_ids_file: str = 'capit_label_ids.csv',
        ignore_extra_tokens: bool = False,
        ignore_start_end: bool = False,
        world_size: int = 1,
        global_rank: int = 0,
        shuffle_n: int = 1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        metadata_file = Path(metadata_file).expanduser()
        with open(metadata_file) as f:
            self.metadata = json.load(f)
        self.ignore_extra_tokens = ignore_extra_tokens
        self.ignore_start_end = ignore_start_end
        self.tar_files = []
        for file_path in self.metadata['tar_files']:
            file_path = Path(file_path).expanduser()
            if file_path.is_absolute():
                self.tar_files.append(str(file_path))
            else:
                self.tar_files.append(str(metadata_file.parent / file_path))
        self.punct_label_ids_file = metadata_file.parent / punct_label_ids_file
        self.punct_label_ids = self.load_label_ids(self.punct_label_ids_file)
        self.capit_label_ids_file = metadata_file.parent / capit_label_ids_file
        self.capit_label_ids = self.load_label_ids(self.capit_label_ids_file)
        self.pad_label = pad_label
        self.check_pad_label()
        begin_idx = (len(self.tar_files) // world_size) * global_rank
        end_idx = begin_idx + (len(self.tar_files) // world_size)
        logging.info(
            "Partitioning tarred dataset: process (%d) taking shards [%d, %d)", global_rank, begin_idx, end_idx
        )
        self.tar_files = self.tar_files[begin_idx: end_idx]
        self.length = self.metadata['num_batches'] // world_size
        self._dataset = wds.WebDataset(urls=self.tar_files, nodesplitter=None).decode(
            wds.handle_extension('.pyd', self.decode_pyd)
        )
        if shuffle_n > 0:
            self._dataset.shuffle(shuffle_n)
        else:
            logging.info("WebDataset will not shuffle files within the tar files.")
        self._dataset = self._dataset.to_tuple('__key__', 'batch.pyd').map(f=self._build_sample)

    def check_pad_label(self):
        for label_ids, label_file, task in [
            (self.punct_label_ids, self.punct_label_ids_file, "punctuation"),
            (self.capit_label_ids, self.capit_label_ids_file, "capitalization")
        ]:
            if self.punct_label_ids[self.pad_label] != 0:
                raise ValueError(
                    f"Pad label '{self.pad_label}' has non zero id {label_ids[self.pad_label]} in {task} "
                    f"ids dictionary loaded from {label_file}."
                )

    @staticmethod
    def load_label_ids(file_path: os.PathLike):
        ids = {}
        with file_path.open() as f:
            for i, line in enumerate(f):
                ids[line.strip()] = i
        return ids

    def decode_pyd(self, key, value):
        return pickle.loads(value)

    def _build_sample(self, batch):
        _, batch = batch
        batch_segment_ids, batch_input_mask, batch_loss_mask = create_masks_and_segment_ids(
            batch['input_ids'],
            batch['subtokens_mask'],
            self.tokenizer.pad_id,
            self.tokenizer.cls_id,
            self.tokenizer.sep_id,
            self.ignore_start_end,
            self.ignore_extra_tokens,
        )
        batch['segment_ids'] = batch_segment_ids
        batch['input_mask'] = batch_input_mask
        batch['loss_mask'] = batch_loss_mask
        return batch

    def __iter__(self):
        return self._dataset.__iter__()

    def __len__(self):
        return self.length

    def collate_fn(self, batch):
        batch = {k: torch.as_tensor(v) for k, v in batch[0].items()}
        batch['segment_ids'] = batch['segment_ids'].int()
        batch['punct_labels'] = batch['punct_labels'].long()
        batch['capit_labels'] = batch['capit_labels'].long()
        return batch
