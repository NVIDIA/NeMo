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
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import webdataset as wds
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from torch.utils.data import IterableDataset

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset import (
    DEFAULT_CAPIT_LABEL_IDS_NAME,
    DEFAULT_PUNCT_LABEL_IDS_NAME,
    BertPunctuationCapitalizationDataset,
    Progress,
    create_label_ids,
    create_masks_and_segment_ids,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.neural_types import ChannelType, LabelsType, MaskType, NeuralType
from nemo.utils import logging

NUMBER_RE = "(0|[1-9][0-9]*)"
TAR_FRAGMENT_TMPL_IN_PROGRESS = "fragment{}.{}.tar"
TAR_FRAGMENT_TMPL_FINISHED = "fragment{}.num_batches{}.{}.tar"
TAR_FRAGMENT_TMPL_TO_REPACK = "fragment{}.num_batches{}.{}.tar.to_repack"
TAR_FRAGMENT_PATTERN_IN_PROGRESS = re.compile(f"fragment{NUMBER_RE}.{NUMBER_RE}.tar$")
TAR_FRAGMENT_PATTERN_FINISHED = re.compile(f"fragment{NUMBER_RE}.num_batches({NUMBER_RE}).{NUMBER_RE}.tar$")
TAR_FRAGMENT_PATTERN_TO_REPACK = re.compile(f"fragment{NUMBER_RE}.num_batches{NUMBER_RE}.{NUMBER_RE}.tar.to_repack$")

DATASET_PARAMETERS_TMPL = "{prefix}.tokens{tokens_in_batch}.max_seq_length{max_seq_length}.{tokenizer}"
TAR_FINAL_TMPL = ".batches{num_batches}.{ctr}.tar"

PROGRESS_REPORT_PERIOD = 10 ** 4


def count_lines_and_get_fragment_starting_positions(
    file_name: Path, lines_per_dataset_fragment: int
) -> Tuple[int, List[int]]:
    """
    Returns number of lines in a file and indices of fragment starting bytes.

    Args:
        file_name: a path to a text or label file
        lines_per_dataset_fragment: number of lines in a dataset fragment. The last fragment can contain less lines

    Returns:
        num_lines: number of lines in a file
        start_bytes: indices of fragment starting bytes
    """
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


def get_fragment_start_bytes(
    text_file: Path, label_file: Path, lines_per_dataset_fragment: int
) -> Tuple[int, List[int], List[int]]:
    """
    A function for calculating borders of dataset fragments. The function is used to split ``text_file`` and
    ``label_file`` for processing them in parallel.

    Args:
        text_file: a path to a dataset source file
        label_file: a path to a dataset label file
        lines_per_dataset_fragment: a number of lines in one fragment

    Returns:
        num_lines: total number of elements in the dataset (number of lines in ``text_file``` and ``label_file``)
        text_start_bytes: indices of the first bytes of fragments in ``text_file``
        label_start_bytes: indices of the first bytes of fragments in ``label_file``
    """
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
    return num_lines, text_start_bytes, label_start_bytes


def process_fragment(
    text_file: Path,
    labels_file: Path,
    output_dir: Path,
    text_start_pos: int,
    label_start_pos: int,
    lines_per_dataset_fragment: int,
    max_seq_length: int,
    tokens_in_batch: int,
    num_batches_per_tarfile: int,
    tokenizer_name: str,
    tokenizer_model: Optional[Path],
    vocab_file: Optional[Path],
    merges_file: Optional[Path],
    special_tokens: Dict[str, str],
    use_fast_tokenizer: Optional[bool],
    tokenizer_bpe_dropout: Optional[bool],
    pad_label: str,
    punct_label_ids: Dict[str, int],
    capit_label_ids: Dict[str, int],
    fragment_idx: int,
    tokenization_progress_queue: mp.Queue,
    batch_mark_up_progress_queue: mp.Queue,
    batch_building_progress_queue: mp.Queue,
    writing_to_tar_progress_queue: mp.Queue,
):
    tokenizer = get_tokenizer(
        tokenizer_name,
        tokenizer_model=str(tokenizer_model),
        vocab_file=str(vocab_file),
        merges_file=str(merges_file),
        special_tokens=special_tokens,
        use_fast=use_fast_tokenizer,
        bpe_dropout=tokenizer_bpe_dropout,
    )
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
        pad_label=pad_label,
        punct_label_ids=punct_label_ids,
        capit_label_ids=capit_label_ids,
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
    current_file_name = output_dir / TAR_FRAGMENT_TMPL_IN_PROGRESS.format(fragment_idx, tar_ctr)
    current_num_batches = 0
    sink = wds.TarWriter(str(current_file_name))
    progress_made = 0
    for batch_i, batch in enumerate(dataset):
        sink.write({"__key__": f"fragment-{fragment_idx}-batch-{batch_i}", "batch.pyd": batch})
        current_num_batches += 1
        progress_made += len(batch['input_ids'])
        if current_num_batches % num_batches_per_tarfile == 0:
            sink.close()
            current_file_name.rename(
                output_dir / TAR_FRAGMENT_TMPL_FINISHED.format(fragment_idx, current_num_batches, tar_ctr)
            )
            writing_to_tar_progress_queue.put(progress_made)
            progress_made = 0
            tar_ctr += 1
            current_file_name = output_dir / TAR_FRAGMENT_TMPL_IN_PROGRESS.format(fragment_idx, tar_ctr)
            current_num_batches = 0
            sink = wds.TarWriter(str(current_file_name))
    sink.close()
    writing_to_tar_progress_queue.put(progress_made)
    if progress_made > 0:
        new_file_name = output_dir / TAR_FRAGMENT_TMPL_TO_REPACK.format(fragment_idx, current_num_batches, tar_ctr)
        current_file_name.rename(new_file_name)
    else:
        current_file_name.unlink()


def remove_unexpected_files_and_dirs(output_dir: Path, output_file_tmpl: str, metadata_file_name: Path):
    """
    This function removes all files with names which may be used in the dataset creation.

    Args:
        output_dir: a path to directory where removal is performed
        output_file_tmpl: a format string for a name of final tar file. Must include fields ``ctr`` for number of the
            file and ``num_batches`` for number of batches in the file.
        metadata_file_name: a metadata file name
    """
    if not output_dir.is_dir():
        return
    tar_final_pattern = re.compile(output_file_tmpl.format(ctr=NUMBER_RE, num_batches=NUMBER_RE))
    unexpected_tar_files = [
        path
        for path in output_dir.iterdir()
        if any(
            [
                p.match(path.name) is not None
                for p in [
                    TAR_FRAGMENT_PATTERN_IN_PROGRESS,
                    TAR_FRAGMENT_PATTERN_FINISHED,
                    TAR_FRAGMENT_PATTERN_TO_REPACK,
                    tar_final_pattern,
                ]
            ]
        )
    ]
    if unexpected_tar_files:
        logging.warning(
            f"Found {len(unexpected_tar_files)} unexpected tar files in the output directory {output_dir}. "
            f"All of them are going to be removed. The files match one of 3 patterns: "
            f"'{TAR_FRAGMENT_PATTERN_IN_PROGRESS.pattern}', '{TAR_FRAGMENT_PATTERN_FINISHED.pattern}', "
            f"'{tar_final_pattern.pattern}'. The first 3 unexpected files: {unexpected_tar_files[:3]}"
        )
        for fn in unexpected_tar_files:
            fn.unlink()
    if metadata_file_name.exists():
        logging.warning(f"Found metadata file {metadata_file_name}. It is going to be removed.")
        metadata_file_name.unlink()
    punct_label_ids = output_dir / DEFAULT_PUNCT_LABEL_IDS_NAME
    capit_label_ids = output_dir / DEFAULT_CAPIT_LABEL_IDS_NAME
    if punct_label_ids.exists():
        logging.warning(f"Found unexpected punctuation label file {punct_label_ids}. It is going to be removed.")
        punct_label_ids.unlink()
    if capit_label_ids.exists():
        logging.warning(f"Found unexpected capitalization label file {capit_label_ids}. It is going to be removed.")
        capit_label_ids.unlink()


def collect_unique_labels_from_fragment(
    label_file: Path, start_pos: int, lines_per_dataset_fragment: int, progress_queue: mp.Queue, fragment_idx: int
) -> Tuple[Set[str], Set[str]]:
    """
    Returns a set of unique punctuation labels and a set of unique capitalization labels.

    Args:
        label_file: a path to a file with labels
        start_pos: an index of the first byte of fragment in ``label_file``
        lines_per_dataset_fragment: number of lines in dataset fragment. In the last fragment there can be less lines.
        progress_queue: a queue for reporting number of processed lines
        fragment_idx: a processed fragment index

    Returns:
        unique_punct: a set of unique punctuation labels
        unique_capit: a set of unique capitalization labels
    """
    unique_punct, unique_capit = set(), set()
    with label_file.open() as f:
        f.seek(start_pos)
        progress_report = 0
        for i in range(lines_per_dataset_fragment):
            line = f.readline()
            if not line:
                break
            pairs = line.split()
            if not all([len(p) == 2 for p in pairs]):
                broken_pairs = [i for i, p in enumerate(pairs) if len(p) != 2]
                raise ValueError(
                    f"Found broken labels line in number {fragment_idx * lines_per_dataset_fragment + i} in file "
                    f"{label_file}. Indices of broken pairs of labels: {broken_pairs}"
                )
            punct, capit = zip(*pairs)
            unique_punct.update(punct)
            unique_capit.update(capit)
            progress_report += 1
            if progress_report >= PROGRESS_REPORT_PERIOD:
                progress_queue.put(progress_report)
                progress_report = 0
        progress_queue.put(progress_report)
    return unique_punct, unique_capit


def create_label_dictionaries(
    label_file: Path,
    text_start_bytes: List[int],
    num_lines: int,
    lines_per_dataset_fragment: int,
    pad_label: str,
    n_jobs: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Creates punctuation and capitalization label ids dictionaries based on labels present in ``label_file``.

    Args:
        label_file: a path to file with labels
        text_start_bytes: indices of first bytes in fragments
        num_lines: total number of lines in ``label_file``
        lines_per_dataset_fragment: number of lines in dataset fragments. The last fragment can have less lines
        pad_label: a label used for padding and for absence of punctuation and capitalization
        n_jobs: a number of fragments processed in parallel

    Returns:
        punct_label_ids: a dictionary with punctuation label ids
        capit_label_ids: a dictionary with capitalization label ids
    """
    with Progress(num_lines, "Creating label dictionary", "line") as progress_queues:
        result = Parallel(n_jobs=min(n_jobs, len(text_start_bytes)))(
            delayed(collect_unique_labels_from_fragment)(
                label_file, start_pos, lines_per_dataset_fragment, *progress_queues, fragment_idx
            )
            for fragment_idx, start_pos in enumerate(text_start_bytes)
        )
    unique_punct, unique_capit = zip(*result)
    unique_punct = set().union(*unique_punct)
    unique_capit = set().union(*unique_capit)
    return create_label_ids(unique_punct, pad_label), create_label_ids(unique_capit, pad_label)


def check_label_ids(pad_label: str, punct_label_ids: Dict[str, int], capit_label_ids: Dict[str, int]):
    """
    A function for checking that pad label has zeroth id in ``punct_label_dis`` and ``capit_label_ids`` dictionaries.
    Args:
        pad_label: a pad label
        punct_label_ids: a dictionary with punctuation label ids
        capit_label_ids: a dictionary with capitalization label ids
    """
    msg = "Parameter `pad_label` has to have id 0 in dictionary `{param_name}` whereas it has id {id_}." + (
        '' if len(pad_label) > 10 else f" pad_label='{pad_label}'"
    )
    if punct_label_ids is not None:
        if punct_label_ids[pad_label] != 0:
            raise ValueError(msg.format(param_name='punct_label_ids', id_=punct_label_ids[pad_label]))
    if capit_label_ids is not None:
        if capit_label_ids[pad_label] != 0:
            raise ValueError(msg.format(param_name='capit_label_ids', id_=capit_label_ids[pad_label]))


def process_error(msg: str, error_class_or_function: Union[Type[Exception], Callable[[str], Any]]):
    if issubclass(error_class_or_function, Exception):
        raise error_class_or_function(msg)
    if callable(error_class_or_function):
        error_class_or_function(msg)
    raise ValueError(
        f"Parameter `error_class_or_function` has to be a subclass of `Exception` or a function."
        f"Given {type(error_class_or_function)}"
    )


def check_before_building_label_ids(
    pad_label: str,
    other_labels: List[str],
    pad_label_name: str,
    other_labels_name: str,
    error_class_or_function: Union[Type[Exception], Callable[[str], Any]],
):
    """
    A function for checking that that all labels are unique.
    Args:
        pad_label: a pad label
        other_labels: a list of labels except for the pad label
        pad_label_name: a name of the pad label used in error message
        other_labels_name: a name of other labels used in error message
        error_class_or_function: a class of an exception which is raised if there is a problem with labels.
            Alternatively it can be a function for handling exceptions, for example ``argparse.ArgumentParser.error``.
            Such a function has to take one argument -- error message.
    """
    for i, lbl in enumerate(other_labels):
        if lbl == pad_label:
            msg = f"Label number {i} in parameter `{other_labels_name}` is equal to `{pad_label_name}`."
            process_error(msg, error_class_or_function)
    for i in range(len(other_labels) - 1):
        for lbl in other_labels[i + 1 :]:
            if lbl == other_labels[i]:
                msg = f"Label number {i} occurs at least 2 times in parameter `{other_labels_name}`."
                process_error(msg, error_class_or_function)


def build_label_ids_from_list_of_labels(pad_label: str, other_labels: List[str]) -> Dict[str, int]:
    """
    Builds label ids dictionary from pad label and list of other labels. Used for parsing command line arguments.
    Args:
        pad_label: a pad label
        other_labels: list of labels except for the pad label

    Returns:
        a dictionary with label ids
    """
    check_before_building_label_ids(pad_label, other_labels, 'pad_label', 'other_labels', ValueError)
    ids = {pad_label: 0}
    for lbl in other_labels:
        ids[lbl] = len(ids)
    return ids


def load_label_ids(ids_file: Path) -> Dict[str, int]:
    ids = {}
    with ids_file.open() as f:
        for i, line in enumerate(f):
            ids[line.strip()] = i
    return ids


def get_label_dictionaries(
    label_file: Path,
    start_bytes: List[int],
    num_lines: int,
    lines_per_dataset_fragment: int,
    pad_label: str,
    punct_label_ids: Optional[Dict[str, int]],
    capit_label_ids: Optional[Dict[str, int]],
    punct_label_ids_file: Optional[Path],
    capit_label_ids_file: Optional[Path],
    n_jobs: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Return label ids if the label ids are present in passed in variables ``punct_label_ids``, ``capit_label_ids``,
    ``punct_label_ids_file``, ``capit_label_ids_file``. Otherwise, label ids are created using ``label_file``.

    Args:
        label_file: a path to file with labels. Labels have to be given in the format described in
            https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html#nemo-data-format
        start_bytes: a list of positions in ``label_file`` at which fragment start. Parameter ``start_bytes`` is used
            for creating labels in parallel for several fragments
        num_lines: total number of lines in ``label_file``. Parameter ``num_lines`` is used for showing progress of
            label ids collection
        lines_per_dataset_fragment: number of lines in a dataset fragment
        pad_label: a label used for padding showing there is no punctuation and capitalization. Label ``pad_label``
            has to have id ``0`` in parameters ``punct_label_ids``, ``capit_label_ids``, ``punct_label_ids_file``,
            ``capit_label_ids_file`` if these parameters are provided.
        punct_label_ids: a dictionary with punctuation label ids. Pad label has to have id ``0``. No more than 1 of
            parameters ``punct_label_ids`` and ``punct_label_ids_file`` can be provided.
        capit_label_ids: a dictionary with capitalization label ids. Pad label has to have id ``0``. No more than 1 of
            parameters ``capit_label_ids`` and ``capit_label_ids_file`` can be provided.
        punct_label_ids_file: a text file with punctuation labels. Every line in the file contains 1 label. Pad label
            has to be in the first line. No more than 1 of parameters ``punct_label_ids`` and ``punct_label_ids_file``
            can be provided.
        capit_label_ids_file: a text file with capitalization labels. Every line in the file contains 1 label. Pad label
            has to be in the first line. No more than 1 of parameters ``capit_label_ids`` and ``capit_label_ids_file``
            can be provided.
        n_jobs: a number of fragments processed in parallel

    Returns:
        punct_label_ids: a dictionary with punctuation label ids
        capit_label_ids: a dictionary with capitalization label ids
    """
    if punct_label_ids is not None and punct_label_ids_file is not None:
        raise ValueError("You can provide at most one of parameters `punct_label_ids` and `punct_label_ids_file`.")
    if capit_label_ids is not None and capit_label_ids_file is not None:
        raise ValueError("You can provide at most one of parameters `capit_label_ids` and `capit_label_ids_file`.")
    if punct_label_ids is None and punct_label_ids_file is not None:
        punct_label_ids = load_label_ids(punct_label_ids_file)
    if capit_label_ids is None and capit_label_ids_file is not None:
        capit_label_ids = load_label_ids(capit_label_ids_file)
    check_label_ids(pad_label, punct_label_ids, capit_label_ids)
    if punct_label_ids is None or capit_label_ids is None:
        _punct_label_ids, _capit_label_ids = create_label_dictionaries(
            label_file, start_bytes, num_lines, lines_per_dataset_fragment, pad_label, n_jobs
        )
        if punct_label_ids is None:
            punct_label_ids = _punct_label_ids
        if capit_label_ids is None:
            capit_label_ids = _capit_label_ids
    return punct_label_ids, capit_label_ids


def decode_pyd(key: str, value: bytes) -> Any:
    """
    Used for decoding batch loaded by ``webdataset`` from tar files.
    Args:
        key: name of a batch
        value: pickled batch

    Returns:
        decoded batch
    """
    return pickle.loads(value)


def repack_tar_files_with_not_enough_batches(output_dir: Path, num_batches_per_tarfile: int):
    f"""
    It is possible that number of batches in a fragment is not evenly divisible by ``num_batches_per_tarfile``.
    In such a case excess batches are put in a tar file which matches a pattern
    ``'{TAR_FRAGMENT_PATTERN_TO_REPACK.pattern}'``. Such files are repacked by
    ``repack_tar_files_with_not_enough_batches`` function into tar files with correct ``num_batches_per_tarfile``
    batches each. If there is no enough batches in repacked files, then up to ``num_batches_per_tarfile - 1``
    remaining batches may be discarded.
    
    Args:
        output_dir: a path to the output directory which contains files to repack and where new files are saved
        num_batches_per_tarfile: a number of batches in 1 tar file. If number of batches in files matching a pattern
            ``'{TAR_FRAGMENT_PATTERN_TO_REPACK.pattern}'`` is not evenly divisible by ``num_batches_per_tarfile``
            excess batches are discarded.
    """
    files_to_repack_with_matches = [
        (path, TAR_FRAGMENT_PATTERN_TO_REPACK.match(path.name))
        for path in output_dir.iterdir()
        if TAR_FRAGMENT_PATTERN_TO_REPACK.match(path.name) is not None
    ]
    files_to_repack_with_matches = sorted(files_to_repack_with_matches, key=lambda x: int(x[1].group(3)))
    files_to_repack = [f for f, m in files_to_repack_with_matches]
    logging.info(f"Found files for repacking: {files_to_repack}")
    files_to_repack = deque(files_to_repack)
    number_of_write_ops = 0
    initial_number_of_files_to_repack = len(files_to_repack)
    pop_file_ds = None
    new_file_sink = None
    new_file_num_batches = 0
    while files_to_repack:
        assert pop_file_ds is None or new_file_sink is None
        if new_file_sink is None:
            append_file = files_to_repack.popleft()
            new_file = append_file.parent / append_file.stem
            logging.info(f"Opening new file sink {new_file}")
            new_file_sink = wds.TarWriter(str(new_file))
            append_ds_to_rewrite = (
                wds.WebDataset(urls=[str(append_file)], nodesplitter=None)
                .decode(wds.handle_extension('.pyd', decode_pyd))
                .to_tuple('__key__', 'batch.pyd')
            )
            for key, batch in iter(append_ds_to_rewrite):
                new_file_sink.write({"__key__": key, "batch.pyd": batch})
                new_file_num_batches += 1
                number_of_write_ops += 1
                assert number_of_write_ops < initial_number_of_files_to_repack * num_batches_per_tarfile
            logging.info(f"{new_file_num_batches} batches were rewritten to new file {new_file}")
        if files_to_repack and pop_file_ds is None:
            pop_file = files_to_repack.pop()
            logging.info(f"Popped file {pop_file}")
            pop_file_ds = (
                wds.WebDataset(urls=[str(pop_file)], nodesplitter=None)
                .decode(wds.handle_extension('.pyd', decode_pyd))
                .to_tuple('__key__', 'batch.pyd')
            )
            pop_file_ds = iter(pop_file_ds)
        if pop_file_ds is not None and new_file_sink is not None:
            while new_file_num_batches < num_batches_per_tarfile:
                try:
                    key, batch = next(pop_file_ds)
                except StopIteration:
                    logging.info(f"Finished extracting from {pop_file}")
                    pop_file_ds = None
                    pop_file.unlink()
                    break
                new_file_sink.write({"__key__": key, "batch.pyd": batch})
                number_of_write_ops += 1
                assert number_of_write_ops < initial_number_of_files_to_repack * num_batches_per_tarfile
                new_file_num_batches += 1
            if new_file_num_batches >= num_batches_per_tarfile:
                logging.info(f"Finished filling file {new_file}")
                assert new_file_num_batches == num_batches_per_tarfile
                new_file_sink.close()
                new_file_sink = None
                new_file_num_batches = 0
    if new_file_sink is not None:
        new_file_sink.close()
        logging.info(f"Removing file {new_file}")
        new_file.unlink()
    if pop_file_ds is not None:
        pop_file.unlink()
    logging.info(f"Wrote totally {number_of_write_ops} batches")


def create_metadata_file(
    output_dir: Path, output_file_tmpl: str, metadata_file_name: Path, num_batches_per_tarfile: int
):
    """
    Rename tar files according to template ``output_file_tmpl`` and save metadata file.
    Args:
        output_dir: a path to directory which contains initial tar files and where renamed tar files are saved
        output_file_tmpl: a template of new tar file name
        metadata_file_name: a path to a file into which metadata is going to be saved
        num_batches_per_tarfile: a required number of batches in tar files. Used for checking present tar files
    """
    metadata = {"num_batches": 0, "tar_files": []}
    for i, fn in enumerate([fn for fn in output_dir.iterdir() if TAR_FRAGMENT_PATTERN_FINISHED.match(fn.name)]):
        nb = int(TAR_FRAGMENT_PATTERN_FINISHED.match(fn.name).group(1))
        assert nb == num_batches_per_tarfile
        new_name = output_dir / output_file_tmpl.format(ctr=i, num_batches=nb)
        fn.rename(new_name)
        metadata['tar_files'].append(new_name.name)
        metadata["num_batches"] += nb
    with metadata_file_name.open('w') as f:
        json.dump(metadata, f, indent=2)


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
    pad_label: str = 'O',
    punct_label_ids: Optional[Dict[str, int]] = None,
    capit_label_ids: Optional[Dict[str, int]] = None,
    punct_label_ids_file: Optional[Union[os.PathLike, str]] = None,
    capit_label_ids_file: Optional[Union[os.PathLike, str]] = None,
    tar_file_prefix: Optional[str] = 'punctuation_capitalization',
    n_jobs: Optional[int] = mp.cpu_count(),
):
    """
    A tarred dataset allows to train on large amounts without storing it all into memory simultaneously.

    Tarred dataset is a directory which contains metadata file, tar files with batches, punct_label_ids.csv and
    capit_label_ids.csv files.

    Metadata file is a JSON file with 2 fields: 'num_batches' and 'tar_files'. 'num_batches' (int) is a total number
    of batches in tarred dataset. 'tar_files' is a list of paths to tar files given relatively to directory
    containing the metadata file.

    Every tar file contains objects written using ``webdataset.TarWriter``. Each object is a dictionary with two items:
    '__key__' and 'batch.pyd'. '__key__' is a name of a batch and 'batch.pyd' is a pickled dictionary which contains
    'input_ids', 'subtokens_mask', 'punct_labels', 'capit_labels'. 'input_ids' is an array containing ids of source
    tokens, 'subtokens_mask' is a boolean array showing first tokens in words, 'punct_labels' and 'capit_labels' are
    arrays with ids of labels.

    Metadata file should be passed to constructor of
    ``nemo.collections.nlp.data.token_classification.PunctuationCapitalizationTarredDataset`` and the instance of the
    class will handle iteration and constructing masks and token types for BERT model.

    Args:
        text_file: a path to a file with dataset source. Dataset source is lowercased text without punctuation. Number
            of lines in ``text_file`` has to be equal to the number of lines in ``label_file``.
        label_file: a path to a file with labels. Labels are given in the format described in
            https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html#nemo-data-format
        output_dir: a path to a directory where metadata file, tar files and 'punct_label_ids.csv' and
            'capit_label_ids.csv' files are saved.
        max_seq_length: Maximum number of subtokens in an input sequence. A source sequence which contain too many
            subtokens are clipped to ``max_seq_length - 2`` subtokens and then [CLS] token is prepended to the clipped
            sequence and [SEP] token is appended to the clipped sequence. The clipping is performed via removal of
            subtokens in the beginning of a source sequence.
        tokens_in_batch: maximum number of tokens in a batch including [CLS], [SEP], [UNK], and [PAD] tokens. Before
            packing into batches source sequences are sorted by number of tokens in order to reduce number of pad
            tokens. So the number of sequences in a batch may be different.
        lines_per_dataset_fragment: a number of lines processed by one worker during creation of tarred dataset.
            A worker tokenizes ``lines_per_dataset_fragment`` keeps in RAM tokenized text labels before packing them
            into batches. Reducing ``lines_per_dataset_fragment`` leads to reducing of the amount of memory required
            by this script.
        num_batches_per_tarfile: a number of batches saved in a tar file. If you increase ``num_batches_per_tarfile``
            there will be less tar files in the dataset. There cannot be less then ``num_batches_per_tarfile`` batches
            in a tar file, and all excess batches are removed. Maximum number of discarded batches is
            ``num_batches_per_tarfile - 1``.
        tokenizer_name: name of the tokenizer used for tokenization of source sequences. Possible options are
            ``'sentencepiece'``, ``'word', ``'char'``, HuggingFace tokenizers. For more options see function
            ``nemo.collections.nlp.modules.common.get_tokenizer``. The tokenizer has have properties ``cls_id``,
            ``pad_id``, ``sep_id``, ``unk_id``."
        tokenizer_model: a path to tokenizer model required for ``'sentencepiece'`` tokenizer.
        vocab_file: a path to vocabulary file which is used in ``'word'``, ``'char'``, and HuggingFace tokenizers.
        merges_file: a path to merges file which maybe used in HuggingFace tokenizers.
        special_tokens: a dictionary with special tokens passed to constructors of ``'char'``, ``'word'``,
            ``'sentencepiece'``, and various HuggingFace tokenizers.
        use_fast_tokenizer: whether to use fast HuggingFace tokenizer.
        tokenizer_bpe_dropout: BPE dropout for YouTokenToMe tokenizer. Currently YouTokenToMe tokenizer is not
            supported because it lacks `cls_id` and `sep_id` properties.
        pad_label: a pad label both for punctuation and capitalization. This label is also is used for marking words
            which do not need punctuation and capitalization.
        punct_label_ids: a dictionary which keys are punctuation labels and values are label ids. The pad label has
            to have id ``0``. You can provide at most one of parameters ``punct_label_ids`` and
            ``punct_label_ids_file``. If none of parameters ``punct_label_ids`` and ``punct_label_ids_file`` are
            provided, then punctuation label ids will be inferred from ``labels`` file.
        capit_label_ids: same as ``punct_label_ids`` for capitalization labels.
        punct_label_ids_file: a path to file with punctuation labels. These labels include pad label. Pad label has to
            be the first label in the file. Each label is written on separate line. Alternatively you can use
            ``punct_labels_ids`` parameter. If none of parameters ``punct_labels_ids`` and ``punct_label_ids_file`` are
            provided, then punctuation label ids will be inferred from ``labels`` file.
        capit_label_ids_file: same as ``punct_label_ids_file`` for capitalization labels.
        tar_file_prefix: a string from which tar file names start
        n_jobs: a number of workers for creating tarred dataset
    """
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
    remove_unexpected_files_and_dirs(output_dir, output_file_tmpl, metadata_file_name)
    num_lines, text_start_bytes, label_start_bytes = get_fragment_start_bytes(
        text_file, label_file, lines_per_dataset_fragment
    )
    if text_start_bytes:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        logging.warning(f"Both {label_file} and {text_file} are empty. Tarred dataset cannot be created.")
        return
    punct_label_ids, capit_label_ids = get_label_dictionaries(
        label_file,
        label_start_bytes,
        num_lines,
        lines_per_dataset_fragment,
        pad_label,
        punct_label_ids,
        capit_label_ids,
        punct_label_ids_file,
        capit_label_ids_file,
        n_jobs,
    )

    with Progress(
        num_lines, ["Tokenization", "Batch mark up", "Batch building", "Writing tarred dataset"], "query"
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
                None if tokenizer_model is None else Path(tokenizer_model).expanduser(),
                None if vocab_file is None else Path(vocab_file).expanduser(),
                None if merges_file is None else Path(merges_file).expanduser(),
                special_tokens,
                use_fast_tokenizer,
                tokenizer_bpe_dropout,
                pad_label,
                punct_label_ids,
                capit_label_ids,
                fragment_idx,
                *progress_queues,
            )
            for fragment_idx, (text_start_pos, label_start_pos) in enumerate(zip(text_start_bytes, label_start_bytes))
        )
    repack_tar_files_with_not_enough_batches(output_dir, num_batches_per_tarfile)
    create_metadata_file(output_dir, output_file_tmpl, metadata_file_name, num_batches_per_tarfile)


class BertPunctuationCapitalizationTarredDataset(IterableDataset):
    """
    Punctuation capitalization dataset for training which allows not to load all data in memory. Tarred dataset is
    created from text and label files using
    examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py script or
    ``nemo.collections.nlp.data.token_classification.punctuation_capitalization_tarred_dataset.create_tarred_dataset``
    function.

    Args:
        metadata_file: a path to tarred dataset metadata file. Metadata file is a JSON file which contains
            ``'num_batches'`` and ``'tar_files'`` items. The first item is total number of batches in a dataset and
            the second is a list of paths to tar files relative to directory containing ``metadata_file``.
        tokenizer: a tokenizer instance used for tokenization of dataset source. A tokenizer instance is used for
            getting ids of [CLS], [PAD], and [SEP] tokens which are used for masks creation.
        pad_label: a label that is used for padding and for absence of both punctuation and capitalization. Used for
            checking parameters ``punct_label_ids_file`` and ``capit_label_ids_file`` parameters.
        punct_label_ids_file: name of a file with punctuation label ids. A file with this name has to be present in
            the same directory with ``metadata_file``. Each line in the file contains a punctuation label. The first
            line has to contain ``pad_label``.
        capit_label_ids_file: name of a file with capitalization label ids. A file with this name has to be present in
            the same directory with ``metadata_file``. Each lien in the file contains a capitalization label. The first
            line has to contain ``pad_label``.
        ignore_extra_tokens: whether to use only first token in a word for loss computation and training. If set to
            ``True``, then loss will be computed only for the first token of a word.
        ignore_start_end: whether to compute loss for [CLS] and [SEP] tokens. If set to ``True``, then loss will not
            be computed for [CLS] and [SEP] tokens.
        world_size: a number of processes used for model training. It is used together with ``global_rank`` parameter
            to decide which tar files will be used in current process.
        global_rank: a number of current process in the pool of workers used for model training. It is used together
            with ``world_size`` parameter to decide which tar files will be used in current process.
        shuffle_n: a number of shuffled batches in a buffer. ``shuffle_n`` batches are loaded into memory, shuffled,
            and then yielded by dataset instance.
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
        self.tar_files = self.tar_files[begin_idx:end_idx]
        self.length = self.metadata['num_batches'] // world_size
        self._dataset = wds.WebDataset(urls=self.tar_files, nodesplitter=None).decode(
            wds.handle_extension('.pyd', decode_pyd)
        )
        if shuffle_n > 0:
            self._dataset.shuffle(shuffle_n)
        else:
            logging.info("WebDataset will not shuffle files within the tar files.")
        self._dataset = self._dataset.to_tuple('__key__', 'batch.pyd').map(f=self._build_sample)

    def check_pad_label(self):
        """Checks the condition that pad label has 0 id in both ``self.punct_label_ids`` and ``self.capit_label_ids``"""
        for label_ids, label_file, task in [
            (self.punct_label_ids, self.punct_label_ids_file, "punctuation"),
            (self.capit_label_ids, self.capit_label_ids_file, "capitalization"),
        ]:
            if label_ids[self.pad_label] != 0:
                raise ValueError(
                    f"Pad label '{self.pad_label}' has non zero id {label_ids[self.pad_label]} in {task} "
                    f"ids dictionary loaded from {label_file}."
                )

    @staticmethod
    def load_label_ids(file_path: Path):
        ids = {}
        with file_path.open() as f:
            for i, line in enumerate(f):
                ids[line.strip()] = i
        return ids

    def _build_sample(self, batch: Tuple[str, Dict[str, ArrayLike]]):
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

    @staticmethod
    def collate_fn(batch: List[Dict[str, ArrayLike]]):
        """Batches are ready in dataset, so ``collate_fn`` is used to take zeroth element of ``batch``."""
        batch = {k: torch.as_tensor(v) for k, v in batch[0].items()}
        batch['segment_ids'] = batch['segment_ids'].int()
        batch['punct_labels'] = batch['punct_labels'].long()
        batch['capit_labels'] = batch['capit_labels'].long()
        return batch
