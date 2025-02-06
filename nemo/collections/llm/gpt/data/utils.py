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

import copy
import datetime
import multiprocessing as mp
import os
import pickle
import subprocess
import time
from functools import partial
from typing import Any

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero

PREFIX_STR = (
    "\x00"  # the prefix string used in the tokenizer to deal with the added empty token for some of the tokenizers
)

IGNORE_INDEX = -100
SYSTEM_TOKEN = "System"

TYPE_INSTRUCTION = {
    'TEXT_TO_VALUE': "",
    'VALUE_TO_TEXT': '',
}

__idx_version__ = "0.2"  # index file version
__idx_suffix__ = "idx"  # index file suffix


def build_index_from_memdata(fn, newline_int):
    """
    Build index of delimiter positions between samples in memmap.
    Can be provided externally.

    Returns a 1D array of ints.
    """
    # use memmap to read file
    mdata = np.memmap(fn, dtype=np.uint8, mode="r")
    # find newline positions
    midx = np.where(mdata == newline_int)[0]
    midx_dtype = midx.dtype
    # make sure to account for all data
    midx = midx.tolist()
    # add last item in case there is no new-line at the end of the file
    if (len(midx) == 0) or (midx[-1] + 1 != len(mdata)):
        midx = midx + [len(mdata) + 1]

    # remove empty lines from end of file
    while len(midx) > 1 and (midx[-1] - midx[-2]) < 2:
        midx.pop(-1)
    midx = np.asarray(midx, dtype=midx_dtype)

    # free memmap
    mdata._mmap.close()
    del mdata

    return midx


def build_index_files(
    dataset_paths,
    newline_int,
    workers=None,
    build_index_fn=build_index_from_memdata,
    index_mapping_dir: str = None,
):
    """Auxiliary method to build multiple index files"""
    if len(dataset_paths) < 1:
        raise ValueError("files_list must contain at leat one file name")

    if workers is None:
        workers = max(1, os.cpu_count() // 2)

    logging.info(f"Processing {len(dataset_paths)} data files using {workers} workers")
    # load all files into memmap
    start_time = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as p:
        build_status = p.map(
            partial(
                _build_memmap_index_files,
                newline_int,
                build_index_fn,
                index_mapping_dir=index_mapping_dir,
            ),
            dataset_paths,
        )

    logging.info(
        f"Time building {sum(build_status)} / {len(build_status)} "
        f"mem-mapped files: {datetime.timedelta(seconds=time.time() - start_time)}"
    )


def handle_index(dataset, idx):
    """
    Remaps negative indices and handles numpy int indices.

    Arguments:
        dataset (Dataset): dataset to index into
        idx (int): Index. Can include negative indices.
    Returns:
        int: Remapped and fully qualified index.

    Raises:
        IndexError: If a negative index is out of range.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from torch.utils.data import TensorDataset
        >>> from nemo_chem.data.fasta_dataset import handle_index
        >>> dataset = TensorDataset(torch.tensor(-np.arange(5)))
        >>> handle_index(dataset, 1)
        1
        >>> handle_index(dataset, -2)
        3

    """
    if idx < 0 and idx > -len(dataset) - 1:
        idx = len(dataset) + idx
    elif idx < 0:
        raise IndexError(f'Index out of range: {idx}')

    return idx


def lightning_prepare_data():
    """
    This function checks whether it is invoked in lightning's hook "prepare_data", which is run only on rank 0.
    TextMemMapDataset contains a torch.distributed.barrier operation, so when run inside the single-process hook
    prepare_data, the barrier operation would hang forever.
    """
    import inspect

    return any(
        [
            frame.function == 'prepare_data' and 'prepare_packed_sequence_data' in frame.code_context[0]
            for frame in inspect.stack()
        ]
    )


def get_samples_mapping(
    indexed_dataset,
    data_prefix,
    num_epochs,
    max_num_samples,
    max_seq_length,
    short_seq_prob,
    seed,
    name,
    binary_head,
    index_mapping_dir: str = None,
    samples_mapping: Any = None,
    sanity_check_dist_workers: bool = True,
):
    """Get a list that maps a sample index to a starting sentence index, end sentence index, and length"""

    from megatron.core import parallel_state

    if not num_epochs:
        if not max_num_samples:
            raise ValueError("Need to specify either max_num_samples " "or num_epochs")
        num_epochs = np.iinfo(np.int32).max - 1
    if not max_num_samples:
        max_num_samples = np.iinfo(np.int64).max - 1

    # Filename of the index mapping
    if index_mapping_dir is not None:
        indexmap_filename = os.path.join(index_mapping_dir, os.path.basename(data_prefix))
    else:
        indexmap_filename = data_prefix
    indexmap_filename += '_{}_indexmap'.format(name)
    if num_epochs != (np.iinfo(np.int32).max - 1):
        indexmap_filename += '_{}ep'.format(num_epochs)
    if max_num_samples != (np.iinfo(np.int64).max - 1):
        indexmap_filename += '_{}mns'.format(max_num_samples)
    indexmap_filename += '_{}msl'.format(max_seq_length)
    indexmap_filename += '_{:0.2f}ssp'.format(short_seq_prob)
    indexmap_filename += '_{}s'.format(seed)
    indexmap_filename += '.npy'

    # Build the indexed mapping if not exist and not provided externally.
    if samples_mapping is None and torch.distributed.get_rank() == 0 and not os.path.isfile(indexmap_filename):
        # Fake index mapping if missing
        if (getattr(indexed_dataset, 'doc_idx', None) is None) and (getattr(indexed_dataset, 'sizes', None) is None):
            _make_indexed_dataset_compatibility(indexed_dataset)

        print(
            ' > WARNING: could not find index map file {}, building '
            'the indices on rank 0 ...'.format(indexmap_filename)
        )

        # Make sure the types match the helpers input types.
        assert indexed_dataset.doc_idx.dtype == np.int64
        assert indexed_dataset.sizes.dtype == np.int32

        # Build samples mapping
        verbose = torch.distributed.get_rank() == 0
        start_time = time.time()
        logging.info(' > building samples index mapping for {} ...'.format(name))
        # First compile and then import.
        try:
            if is_global_rank_zero():
                _compile_helper()
            from megatron.core.datasets import helpers_cpp
        except ImportError:
            raise ImportError(
                'Could not compile megatron dataset C++ helper functions '
                'and therefore cannot import helpers python file.'
            )
        samples_mapping = helpers_cpp.build_mapping(
            indexed_dataset.doc_idx,
            indexed_dataset.sizes,
            num_epochs,
            max_num_samples,
            max_seq_length,
            short_seq_prob,
            seed,
            verbose,
            2 if binary_head else 1,
        )
        logging.info(' > done building samples index maping')
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
        logging.info(' > saved the index mapping in {}'.format(indexmap_filename))
        # Make sure all the ranks have built the mapping
        logging.info(
            ' > elasped time to build and save samples mapping ' '(seconds): {:4f}'.format(time.time() - start_time)
        )

    if sanity_check_dist_workers:
        torch.distributed.barrier()
        counts = torch.cuda.LongTensor([1])
        torch.distributed.all_reduce(counts, group=parallel_state.get_data_parallel_group(with_context_parallel=True))
        torch.distributed.all_reduce(counts, group=parallel_state.get_pipeline_model_parallel_group())
        assert counts[0].item() == (
            torch.distributed.get_world_size()
            // torch.distributed.get_world_size(group=parallel_state.get_tensor_model_parallel_group())
        )
    # Load indexed dataset if not given externally.
    if samples_mapping is None:
        logging.info(' > loading indexed mapping from {}'.format(indexmap_filename))
        start_time = time.time()
        samples_mapping = np.load(indexmap_filename, allow_pickle=True, mmap_mode='r')
        logging.info('    loaded indexed file in {:3.3f} seconds'.format(time.time() - start_time))
        logging.info('    total number of samples: {}'.format(samples_mapping.shape[0]))

    # Deallocate temporary numpy arrays that were created for `get_samples_mapping()` when needed
    if hasattr(indexed_dataset, 'doc_idx') and hasattr(indexed_dataset, 'sizes'):
        _deallocate_indexed_dataset_memory(indexed_dataset)

    return samples_mapping


def _make_indexed_dataset_compatibility(dataset):
    """Make any dataset compatible with IndexedDataset for Megatron samples mapping."""
    if (getattr(dataset, 'doc_idx', None) is not None) or (getattr(dataset, 'sizes', None) is not None):
        raise AttributeError("Dataset already has doc_idx or sizes attributes.")

    dataset.doc_idx = np.arange(len(dataset) + 1, dtype=np.int64)
    dataset.sizes = np.ones(len(dataset), dtype=np.int32)

    return dataset


def preprocess(
    source: dict,
    tokenizer: TokenizerSpec,
    name_end_token_ids: int,
    label_start_ids: list,
    special_tokens: dict,
    num_turn_start_tokens: int,
):
    """
    Given a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    header, conversation, data_type, mask_role = _get_header_conversation_type_mask_role(source, special_tokens)
    # tokenize conversations
    input_ids = tokenizer.text_to_ids(conversation)
    target = copy.deepcopy(input_ids)
    header_tokens = tokenizer.text_to_ids(header)
    header_len = len(header_tokens)

    ids = []
    tokenized_lens = []
    assert torch.equal(torch.tensor(target[:header_len]), torch.tensor(header_tokens))
    for s in source['conversations']:
        # hack to remove the extra empty token in front
        id1 = tokenizer.text_to_ids(PREFIX_STR + s["value"])
        id2 = tokenizer.text_to_ids(PREFIX_STR)
        tokenized_sentence = id1[len(id2) :]
        ids.append(torch.tensor(tokenized_sentence))
        tokenized_lens.append(len(tokenized_sentence))
    speakers = [sentence["from"] for sentence in source['conversations']]
    # assert mask_role in speakers, "mask role not in the conversation"
    split_mask = mask_role.split(',')
    for s in split_mask:
        assert s in speakers, "mask role not in the conversation"

    target = torch.LongTensor(target)
    # not going to train on the header
    target[:header_len] = IGNORE_INDEX
    input_ids = torch.LongTensor(input_ids)
    _mask_targets(
        target,
        tokenized_lens,
        speakers,
        header_len,
        ids,
        tokenizer,
        mask_role,
        data_type,
        name_end_token_ids,
        special_tokens,
        label_start_ids,
        num_turn_start_tokens,
    )
    mask = (target != IGNORE_INDEX).bool()
    assert mask.sum().item() != 0, "mask is empty"
    # Choose the last conversation as answer other history are context
    last_ignore_index_pos = torch.nonzero(target == IGNORE_INDEX)[-1].item() + 1
    context_ids = input_ids[:last_ignore_index_pos]
    answer_ids = input_ids[last_ignore_index_pos:]

    return dict(input_ids=input_ids, mask=mask, context_ids=context_ids, answer_ids=answer_ids)


def _mask_targets(
    target,
    tokenized_lens,
    speakers,
    header_len,
    s_ids,
    tokenizer,
    mask_role,
    gtype,
    name_end_token_ids,
    special_tokens,
    label_start_ids,
    num_turn_start_tokens,
):
    """This function masks the tokens so the loss is computed only on the non-masked role's responses.
    For 'TEXT_TO_VALUE' type, the loss is computed on the value attributes.

    Args:
        target (Tensor): input ids
        tokenized_lens (List[int]): array of lengths of each turns
        speakers (List[str]): array of speakers of each turns
        header_len (int): the system prompt length
        s_ids (List[Tensor]): array of tokenized ids of each turns
        tokenizer (TokenizerSpec): tokenizer object
        mask_role (str): the speaker id to be masked from loss computation.
        gtype (str): either 'TEXT_TO_VALUE' or 'VALUE_TO_TEXT'
        name_end_token_ids (int): end of name token ids
        special_tokens (dict): special tokens used for the chat prompt.
        label_start_ids (list): list of label start token ids,
        num_turn_start_tokens (int): number of tokens of the turn_start str
    """
    TURN_TOKEN = special_tokens['turn_start']
    END_NAME_SIGNAL = special_tokens['end_of_name']
    label_start_ids = torch.tensor(label_start_ids)
    name_end_token_ids = torch.tensor(name_end_token_ids)

    cur_idx = header_len
    tgt_len = target.shape[0]
    for i, (tokenized_len, speaker, s_id) in enumerate(zip(tokenized_lens, speakers, s_ids)):
        # note, sentence piece will add extra empty token in front. has to compute the diff
        id1 = tokenizer.text_to_ids(PREFIX_STR)
        id2 = tokenizer.text_to_ids(PREFIX_STR + TURN_TOKEN + speaker + END_NAME_SIGNAL)
        skip_name_len = len(id2) - len(
            id1
        )  # s_ids[:skip_name_len] is the name part of the prompt 'TURN_TOKEN + speaker + END_NAME_SIGNAL'
        # get the position of the label start string in this turn
        location = _identify_start_index_of_subsequence(label_start_ids, s_id)

        if location >= 0:
            # if it contains the label start tokens
            if gtype == 'VALUE_TO_TEXT':
                # handles the case that condition on labels to generate respone
                # the next token after the name part of the prompt is the beginning of the label start tokens
                assert skip_name_len == location
                # find the first new line token after the label part, which indicates the end of the whole label string
                # newline_loc = torch.where((s_id[skip_name_len:] == name_end_token_ids))[0]
                newline_loc = _identify_start_index_of_subsequence(name_end_token_ids, s_id[skip_name_len:])
                if newline_loc < 0:
                    # cannot find new line token, which means the the whole turn
                    # is just a partial label string. Mask the whole turn
                    target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
                    continue
                # skip the label part and the new line token
                more_skip_len = newline_loc + len(name_end_token_ids)
                # skip the name part and the label part
                skip_name_len += more_skip_len
            elif gtype == 'TEXT_TO_VALUE':
                # handles the case that condition on response to generate label
                # skip the name part, response and the label start tokens part,
                # the remainder is the label string without label start, e.g. 'quality:9,toxicity:8...'
                skip_name_len = location + len(label_start_ids)
        if cur_idx >= tgt_len:
            break
        elif cur_idx + tokenized_len < tgt_len:
            # Check whether the mask is applied to the correct position, the first token is turn start tokens
            if not torch.equal(target[cur_idx + 1 : cur_idx + tokenized_len], s_id[1:]):
                logging.warning("a sentence mismatches the corresponding piece " "in the conversation")
        if i == 0 and (gtype == 'VALUE_TO_TEXT' or gtype is None):
            # mask the first turn completely to provide at least one turn as context for the rest
            target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker in mask_role and i == 1 and gtype == 'TEXT_TO_VALUE':
            # leave the first turn start tag unmasked, servers severs as the end of turn signal
            target[cur_idx + num_turn_start_tokens : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker in mask_role and (i > 1):
            # leave the first turn start tag unmasked, which severs as the end of turn signal
            target[cur_idx + num_turn_start_tokens : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker in mask_role and (i <= 1):
            # mask out everything in the second turn
            target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
        else:
            # mask up to name part, label part for VALUE_TO_TEXT, or name part,
            # response and label start tokens for TEXT_TO_VALUE, or just the name part if gtype is None
            target[cur_idx : cur_idx + skip_name_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _get_header_conversation_type_mask_role(source, special_tokens):
    END_SIGNAL = special_tokens['end_of_turn']
    END_NAME_SIGNAL = special_tokens['end_of_name']

    data_type = None
    if 'type' in source:
        data_type = source['type']
        if data_type is not None:
            assert data_type in TYPE_INSTRUCTION, f"source type {data_type} not supported"
    # add end signal and concatenate together
    conversation = source['system']
    if data_type is not None:
        if TYPE_INSTRUCTION[data_type] != '':
            conversation = conversation + '\n' + TYPE_INSTRUCTION[data_type]
    mask_role = source.get('mask', 'User')
    header = f"{special_tokens['system_turn_start']}{SYSTEM_TOKEN}{END_NAME_SIGNAL}{conversation}{END_SIGNAL}"
    conversation = _add_speaker_and_signal(header, source['conversations'], mask_role, data_type, special_tokens)

    return header, conversation, data_type, mask_role


def _add_speaker_and_signal(header, source, mask_role, gtype, special_tokens):
    TURN_TOKEN = special_tokens['turn_start']
    END_SIGNAL = special_tokens['end_of_turn']
    LABEL_START = special_tokens['label_start']
    END_NAME_SIGNAL = special_tokens['end_of_name']

    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = ""
    conversation = header
    for i, sentence in enumerate(source):
        sentence_from = sentence["from"]
        role_token = TURN_TOKEN
        if gtype is None:
            sentence["value"] = (
                BEGIN_SIGNAL + role_token + sentence_from + END_NAME_SIGNAL + sentence["value"] + END_SIGNAL
            )
        elif gtype == "VALUE_TO_TEXT":
            sentence["value"] = (
                BEGIN_SIGNAL
                + role_token
                + sentence_from
                + END_NAME_SIGNAL
                + (
                    _response_value_formater(sentence['label'], LABEL_START, END_NAME_SIGNAL)
                    if 'label' in sentence
                    else ''
                )
                + sentence["value"]
                + END_SIGNAL
            )
        elif gtype == "TEXT_TO_VALUE":
            sentence["value"] = (
                BEGIN_SIGNAL
                + role_token
                + sentence_from
                + END_NAME_SIGNAL
                + sentence["value"]
                + END_SIGNAL
                + (
                    _response_value_formater(sentence['label'], LABEL_START, END_NAME_SIGNAL)
                    if 'label' in sentence
                    else ''
                )
            )
        else:
            raise ValueError(
                f"source type {gtype} not supported, only 'VALUE_TO_TEXT' and 'TEXT_TO_VALUE' are supported"
            )
        conversation += sentence["value"]
        # if the last turn is not masked, add next token start token to the end,
        # which will be included for loss calculation
        if sentence_from not in mask_role and i == len(source) - 1:
            conversation += TURN_TOKEN

    return conversation


def _response_value_formater(label, label_start, end_signal):
    if isinstance(label, str):
        return label_start + label + end_signal
    elif label is None:
        return ''
    else:
        raise ValueError(f'Unknown label type {type(label)}, only str type is supported')


def _identify_start_index_of_subsequence(subsequence, sequence):
    """find the location of the small tensor in the large tensor.
        e.g.  small = [1,3], large = [2,3,1,3], returns 2
              small = [3,2], large = [2,3,1,3], returns -1
    Args:
        small (tensor): small tensor
        large (tensor): large tensor
    """
    for i in range(sequence.size(0) - subsequence.size(0) + 1):
        if torch.equal(sequence[i : i + subsequence.size(0)], subsequence):
            return i
    return -1


def _build_memmap_index_files(newline_int, build_index_fn, fn, index_mapping_dir: str):
    """Helper function to build an index file"""
    idx_fn = _index_fn(fn, index_mapping_dir)

    # create data map
    if _index_file_exists(idx_fn):
        return False
    else:
        logging.info(f"Building indexing for fn = {fn}")
        # find all newline positions
        midx = build_index_fn(fn, newline_int)
        # validate midx
        midx = np.asarray(midx)
        if not np.issubdtype(midx.dtype, np.integer):
            raise TypeError(f"midx must be an integer array, but got type = {midx.dtype}")

        # create e metadata file
        data = dict(newline_int=newline_int, version=__idx_version__)

        # save index as numpy array to enable memmap reading
        logging.info(f"Saving idx file = {idx_fn}.npy")
        np.save(idx_fn + ".npy", midx, allow_pickle=True)
        logging.info(f"Saving metadata file = {idx_fn}.info")
        pickle.dump(data, open(idx_fn + ".info", "wb"))

        return True


def _index_fn(fn: str, index_mapping_dir: str) -> str:
    """Return base file name of index files.

    This returns the base file name associated with specified index
    files. This base name is the base on top of which suffixes
    like .npy or .info are added.

    The parent directory is created if it does not already exist.

    fn may be specified in multiple ways:
    1. file name: data.jsonl,
    2. relative path to a file: relative/path/to/data.jsonl,
    3. absolute path to a file: /absolute/path/to/data.jsonl.

    This function returns paths in the pattern of:
    1. /path/to/input_mapping_dir/data.jsonl.idx
    2. /path/to/input_mapping_dir/relative/path/to/data.jsonl.idx
    3. /path/to/input_mapping_dir/absolute/path/to/data.jsonl.idx

    Args:
        fn: filename to get base name for.
        index_mapping_dir: directory to save the index mapping to.
                If None, will write to the same folder as the dataset.
    """
    if index_mapping_dir:
        # Remove leading "/" and "..".
        while fn.startswith(("/", "..")):
            if fn.startswith(".."):
                fn = fn.lstrip("..")
            if fn.startswith("/"):
                fn = fn.lstrip("/")
        idx_fn = f"{os.path.join(index_mapping_dir, fn)}.{__idx_suffix__}"
        # Create parent directory if needed.
        os.makedirs(os.path.dirname(idx_fn), exist_ok=True)
    else:
        idx_fn = f"{fn}.{__idx_suffix__}"
    return idx_fn


def _index_file_exists(idx_fn):
    """Helper function to test if index file exists"""
    if os.path.exists(idx_fn + ".npy") and os.path.exists(idx_fn + ".info"):
        return True
    else:
        return False


def _compile_helper():
    """Compile helper function ar runtime. Make sure this
    is invoked on a single process."""

    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(['make', '-C', path])
    if ret.returncode != 0:
        logging.error("Making C++ dataset helpers module failed, exiting.")
        import sys

        sys.exit(1)


def _deallocate_indexed_dataset_memory(indexed_dataset):
    """Deallocate memory of an IndexedDataset."""
    if isinstance(indexed_dataset):
        # for MMapIndexedDataset we cannot release any memory of sizes
        indexed_dataset._index._doc_idx = None
    else:
        indexed_dataset.sizes = None
        indexed_dataset.doc_idx = None
