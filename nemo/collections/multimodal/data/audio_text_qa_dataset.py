# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import io
import json
import math
import multiprocessing
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import braceexpand
import numpy as np
import torch
import webdataset as wd
from omegaconf import DictConfig
from torch.utils.data import ChainDataset
from tqdm import tqdm

from nemo.collections.asr.data.audio_to_text import (
    cache_datastore_manifests,
    expand_sharded_filepaths,
    shard_manifests_if_needed,
)
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.common import tokenizers
from nemo.collections.common.parts.preprocessing import collections, parsers
from nemo.collections.multimodal.parts.utils.data_utils import ceil_to_nearest, maybe_cast_to_list
from nemo.core.classes import Dataset, IterableDataset
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.data_utils import (
    DataStoreObject,
    datastore_object_get,
    datastore_path_to_webdataset_url,
    is_datastore_cache_shared,
    is_datastore_path,
    is_tarred_path,
)
from nemo.utils.get_rank import is_global_rank_zero

__all__ = [
    'AudioQuestionAnswerDataset',
    'TarredAudioQuestionAnswerDataset',
]


def _speech_collate_fn(audio_signals, audio_lengths):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        audio_signals: List[Tensor]
        audio_lengths: List[Tensor]
    """

    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()

    audio_signals_padded = []
    for sig, sig_len in zip(audio_signals, audio_lengths):
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signals_padded.append(sig)

    if has_audio:
        audio_signals_padded = torch.stack(audio_signals_padded)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signals_padded, audio_lengths = None, None

    return audio_signals_padded, audio_lengths


class TokenizerWrapper:
    def __init__(self, tokenizer):
        if isinstance(tokenizer, tokenizers.aggregate_tokenizer.AggregateTokenizer):
            self.is_aggregate = True
        else:
            self.is_aggregate = False
        self._tokenizer = tokenizer

    def __call__(self, *args):
        if isinstance(args[0], List) and self.is_aggregate:
            t = []
            for span in args[0]:
                t.extend(self._tokenizer.text_to_ids(span['str'], span['lang']))
            return t

        t = self._tokenizer.text_to_ids(*args)
        return t


class AudioQuestionAnswerDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        parser: Str for a language specific preprocessor or a callable.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor object used to augment loaded
            audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include in dataset
        max_utts: Limit number of utterances
        trim: whether or not to trim silence. Defaults to False
        channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
        --------- NLP SPECIFIC ARGS -------------
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        add_sep (bool): Whether to add a separation token to each data example (goes between prompt and answer)
        tokens_to_generate (int): (inference only) Number of tokens to generate during inference
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        seed: int = 1234,
        input_key: Key to use for the context in your JSONL file
        output_key: Key to use for the label in your JSONL file
        separate_prompt_and_response_with_newline: Adds a newline between prompt and response.
        answer_only_loss: If True, will compute the loss only on the answer part of the input. If False, will compute the loss on the entire input.
        truncation_field: Field to use for truncation. (Options: "answer", "context"). Field to be used for truncation if the combined length exceeds the max sequence length.
        pad_to_max_length: Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        index_mapping_dir: Directory to save the index mapping to. If None, will write to the same folder as the dataset.
        prompt_template: Prompt template to inject via an fstring. Formatted like Q: {input}\n\nA: {output}
    """

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: Optional[int] = None,
        max_num_samples: Optional[int] = None,
        seed: int = 1234,
        separate_prompt_and_response_with_newline: bool = False,
        answer_only_loss: bool = True,
        truncation_field: str = "answer",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        index_mapping_dir: str = None,
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        index_by_file_id: bool = False,
        input_key: str = 'input',
        output_key: str = 'output',
        end_string: Optional[str] = None,
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.max_num_samples = max_num_samples
        self.seed = seed
        self.separate_prompt_and_response_with_newline = separate_prompt_and_response_with_newline
        self.answer_only_loss = answer_only_loss
        self.truncation_field = truncation_field
        self.pad_to_max_length = pad_to_max_length
        self.index_mapping_dir = index_mapping_dir
        self.prompt_template = prompt_template
        self.virtual_tokens = virtual_tokens
        self.tokens_to_generate = tokens_to_generate
        self.tokens_to_generate = tokens_to_generate
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_sep = add_sep
        self.end_string = end_string

        if add_bos and hasattr(tokenizer, "bos_id") and tokenizer.bos_id > 0:
            self.bos_id = tokenizer.bos_id
        else:
            self.bos_id = None

        if add_eos and hasattr(tokenizer, "eos_id") and tokenizer.eos_id > 0:
            self.eos_id = tokenizer.eos_id
        else:
            self.eos_id = None

        if hasattr(tokenizer, "pad_id"):
            self.pad_id = tokenizer.pad_id
        else:
            self.pad_id = self.eos_id if self.eos_id is not None else 0

        self.sep_id = sep_id if add_sep else None

        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id > 0:
            self.pad_id = tokenizer.pad_id
        else:
            self.pad_id = 0

        if self.prompt_template is not None:
            # When providing things like newlines in the prompt template via the CLI, they are escaped. This line unescapes them.
            self.prompt_template = self.prompt_template.encode('utf-8').decode('unicode_escape')
        assert self.truncation_field in ["answer", "context"]

        if type(manifest_filepath) == str:
            manifest_filepath = manifest_filepath.split(",")

        # If necessary, cache manifests and audio from object store
        cache_datastore_manifests(manifest_filepaths=manifest_filepath, cache_audio=True)

        self.collection = collections.ALMAudioQA(
            manifests_files=manifest_filepath,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
            index_by_file_id=index_by_file_id,
            max_num_samples=max_num_samples,
        )

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.channel_selector = channel_selector

    def get_manifest_sample(self, sample_id):
        return self.collection[sample_id]

    def __getitem__(self, index):
        output = {"idx": index}
        sample = self.collection[index]
        offset = sample.offset

        if offset is None:
            offset = 0

        features = self.featurizer.process(
            sample.audio_file,
            offset=offset,
            duration=sample.duration,
            trim=self.trim,
            orig_sr=sample.orig_sr,
            channel_selector=self.channel_selector,
        )
        f, fl = features, torch.tensor(features.shape[0]).long()
        output["audio_signal"] = f
        output["audio_length"] = fl

        text_data = self._process_example(context=sample.question, output=sample.answer)

        output.update(text_data)

        return output

    def __len__(self):
        return len(self.collection)

    def _process_example(self, context: str, output: str):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.

        function copied from nemo/collections/nlp/data/language_modelling/megatron/gpt_sft_dataset.py
        """

        if self.end_string:
            output += self.end_string

        if self.prompt_template is not None:
            assert f'{{{self.input_key}}}' in self.prompt_template
            assert f'{{{self.output_key}}}' in self.prompt_template
            # Make sure that '{output}' always occurs at the end of the prompt template string
            assert self.prompt_template.index(f'{{{self.output_key}}}') == len(self.prompt_template) - len(
                f'{{{self.output_key}}}'
            )
            # Get the context by replacing only the input
            original_context = context
            context = (
                self.prompt_template.replace(f'{{{self.input_key}}}', context)
                .replace(f'{{{self.output_key}}}', '')
                .strip(' ')
            )
            # Replace the input and output placeholders with the actual input and output
            text = self.prompt_template.replace(f'{{{self.input_key}}}', original_context).replace(
                f'{{{self.output_key}}}', output
            )

        if self.separate_prompt_and_response_with_newline and self.prompt_template is None:
            text = context + '\n' + output
        elif not self.separate_prompt_and_response_with_newline and self.prompt_template is None:
            text = context + ' ' + output

        if self.virtual_tokens:
            # (@adithyare) we are going to insert "pad/eos" tokens in the beginning of the text and context
            # these pad/eos tokens are placeholders for virtual tokens
            pre_pad = [self.tokenizer.eos_id] * self.virtual_tokens
        else:
            pre_pad = []
        tokenized_text = pre_pad + self.tokenizer.text_to_ids(text)
        context_ids = pre_pad + self.tokenizer.text_to_ids(context)
        answer_ids = tokenized_text[len(context_ids) :]

        # for the long context cases, collate_fn includes self.tokens_to_generate for padding
        total_ids = len(context_ids) + max(len(answer_ids), self.tokens_to_generate)
        if self.add_bos:
            total_ids += 1
        if self.add_sep:
            total_ids += 1
        # Only training need to consider eos token
        if self.add_eos and self.tokens_to_generate == 0:
            total_ids += 1

        # If the total number of token is greater than the max, we will try to truncate the answer
        if total_ids > self.max_seq_length:
            truncation_length = total_ids - self.max_seq_length
            if self.truncation_field == "answer":
                answer_ids = answer_ids[: -min(truncation_length, len(answer_ids))]
            elif self.truncation_field == "context":
                context_ids = context_ids[: -min(truncation_length, len(context_ids))]

        if len(context_ids) > self.max_seq_length:
            context_ids = context_ids[: self.max_seq_length]

        assert len(context_ids) <= self.max_seq_length
        input_ids = context_ids

        answer_start_idx = len(input_ids)
        # Adds sep token between text/prompt and answer
        if self.add_sep:
            input_ids = input_ids + [self.sep_id]
            answer_start_idx += 1

        input_ids = input_ids + answer_ids

        if self.add_bos:
            input_ids = [self.tokenizer.bos_id] + input_ids
            answer_start_idx += 1
        if self.add_eos:
            input_ids = input_ids + [self.tokenizer.eos_id]

        if len(input_ids) < self.min_seq_length or len(input_ids) > self.max_seq_length:
            input_ids = input_ids[: self.max_seq_length]

        processed_example = {
            'input_ids': input_ids,
            'answer_start_idx': answer_start_idx,
            'context_ids': context_ids,
            'context_length': len(context_ids),
            'answer_ids': answer_ids,
        }

        return processed_example

    def _build_loss_mask(self, processed_example):
        """ Pad input_ids in batch to max batch length while building loss mask """
        # function copied from nemo/collections/nlp/data/language_modelling/megatron/gpt_sft_dataset.py
        input_ids = processed_example['input_ids']
        answer_start_idx = processed_example['answer_start_idx']
        if self.answer_only_loss:
            loss_mask = [float(idx >= answer_start_idx) for idx in range(len(input_ids))]
        else:
            loss_mask = [1.0] * len(input_ids)

        return loss_mask

    def _collate_item(self, item, max_length, pad_id):
        # function copied from nemo/collections/nlp/data/language_modelling/megatron/gpt_sft_dataset.py
        item = maybe_cast_to_list(item)
        # max_length = max([len(x) for x in item]) if item else 0
        # here [0] should be tokenizer.pad_id
        item = [x + [pad_id] * (max_length - len(x)) for x in item]
        return item

    def _collate_fn(self, batch):
        text_pad_id = self.pad_id
        sample_ids = [x["idx"] for x in batch]
        sample_ids = torch.tensor(sample_ids, dtype=torch.int32)

        audio_signal = [x["audio_signal"] for x in batch]
        audio_lengths = [x["audio_length"] for x in batch]
        audio_signal, audio_lengths = _speech_collate_fn(audio_signal, audio_lengths)

        input_ids = [item['input_ids'][:-1] for item in batch]
        labels = [item['input_ids'][1:] for item in batch]
        contexts = [item['context_ids'] for item in batch]
        context_lengths = torch.LongTensor([item['context_length'] for item in batch])
        answers = [item['answer_ids'] for item in batch]
        loss_mask = [self._build_loss_mask(item)[1:] for item in batch]

        max_length = max([len(x) for x in input_ids]) + self.tokens_to_generate
        # increase max length to nearest multiple of 4 or 8
        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            max_length = min(self.max_seq_length, ceil_to_nearest(max_length, 8))
        assert max_length <= self.max_seq_length

        position_ids = [list(range(max_length)) for _ in batch]
        position_ids = torch.LongTensor(position_ids)
        input_length = torch.LongTensor([len(x) for x in input_ids])
        input_ids = torch.LongTensor(self._collate_item(input_ids, max_length=max_length, pad_id=text_pad_id))
        labels = torch.LongTensor(self._collate_item(labels, max_length=max_length, pad_id=text_pad_id))
        loss_mask = torch.LongTensor(self._collate_item(loss_mask, max_length=max_length, pad_id=0))
        contexts = torch.LongTensor(self._collate_item(contexts, max_length=max_length, pad_id=text_pad_id))
        answers = torch.LongTensor(self._collate_item(answers, max_length=max_length, pad_id=text_pad_id))

        batch = {
            'sample_ids': sample_ids,
            'audio_signal': audio_signal,
            'audio_signal_length': audio_lengths,
            'tokens': input_ids,
            'tokens_length': input_length,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'contexts': contexts,
            'context_lengths': context_lengths,
            'answers': answers,
            'max_length': torch.LongTensor(max_length),
        }

        return batch


class _TarredAudioToTextDataset(IterableDataset):
    """
    A similar Dataset to the AudioToCharDataset/AudioToBPEDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToCharDataset/AudioToBPEDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    Note: For brace expansion in (1), there may be cases where `{x..y}` syntax cannot be used due to shell interference.
    This occurs most commonly inside SLURM scripts. Therefore we provide a few equivalent replacements.
    Supported opening braces - { <=> (, [, < and the special tag _OP_.
    Supported closing braces - } <=> ), ], > and the special tag _CL_.
    For SLURM based tasks, we suggest the use of the special tags for ease of use.

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple workers the number of shards should be divisible by world_size to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToCharDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        parser (callable): A callable which is used to pre-process the text output.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        blank_index (int): Blank character index, defaults to -1.
        unk_index (int): Unknown character index, defaults to -1.
        normalize (bool): Dataset parameter.
            Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        bos_id (id): Dataset parameter.
            Beginning of string symbol id used for seq2seq models.
            Defaults to None.
        eos_id (id): Dataset parameter.
            End of string symbol id used for seq2seq models.
            Defaults to None.
        pad_id (id): Token used to pad when collating samples in batches.
            If this is None, pads using 0s.
            Defaults to None.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                .. warning::
                    Replicated strategy allows every node to sample the entire set of available tarfiles,
                    and therefore more than one node may sample the same tarfile, and even sample the same
                    data points! As such, there is no assured guarantee that all samples in the dataset will be
                    sampled at least once during 1 epoch. Scattered strategy, on the other hand, on specific
                    occasions (when the number of shards is not divisible with ``world_size``), will not sample
                    the entire dataset. For these reasons it is not advisable to use tarred datasets as validation
                    or test datasets.
        shard_manifests (bool): Whether or not to try / shard manifests. Defaults to False.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        return_sample_id (bool): whether to return the sample_id as a part of each sample
        --------- NLP SPECIFIC ARGS -------------
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        add_sep (bool): Whether to add a separation token to each data example (goes between prompt and answer)
        tokens_to_generate (int): (inference only) Number of tokens to generate during inference
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        seed: int = 1234,
        input_key: Key to use for the context in your JSONL file
        output_key: Key to use for the label in your JSONL file
        separate_prompt_and_response_with_newline: Adds a newline between prompt and response.
        answer_only_loss: If True, will compute the loss only on the answer part of the input. If False, will compute the loss on the entire input.
        truncation_field: Field to use for truncation. (Options: "answer", "context"). Field to be used for truncation if the combined length exceeds the max sequence length.
        pad_to_max_length: Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        index_mapping_dir: Directory to save the index mapping to. If None, will write to the same folder as the dataset.
        prompt_template: Prompt template to inject via an fstring. Formatted like Q: {input}\n\nA: {output}
    """

    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        parser: Callable,
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional['nemo.collections.asr.parts.perturb.AudioAugmentor'] = None,
        shuffle_n: int = 0,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        shard_strategy: str = "scatter",
        shard_manifests: bool = False,
        global_rank: int = 0,
        world_size: int = 0,
        return_sample_id: bool = False,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: int = None,
        max_num_samples: int = None,
        seed: int = 1234,
        separate_prompt_and_response_with_newline: bool = False,
        answer_only_loss: bool = True,
        truncation_field: str = "answer",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        index_mapping_dir: str = None,
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
    ):
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_sep = add_sep
        self.sep_id = sep_id
        self.max_num_samples = max_num_samples
        self.seed = seed
        self.separate_prompt_and_response_with_newline = separate_prompt_and_response_with_newline
        self.answer_only_loss = answer_only_loss
        self.truncation_field = truncation_field
        self.pad_to_max_length = pad_to_max_length
        self.index_mapping_dir = index_mapping_dir
        self.prompt_template = prompt_template
        self.virtual_tokens = virtual_tokens
        self.tokens_to_generate = tokens_to_generate
        self.tokens_to_generate = tokens_to_generate
        if self.prompt_template is not None:
            # When providing things like newlines in the prompt template via the CLI, they are escaped. This line unescapes them.
            self.prompt_template = self.prompt_template.encode('utf-8').decode('unicode_escape')
        assert self.truncation_field in ["answer", "context"]

        self.shard_manifests = shard_manifests

        # Shard manifests if necessary and possible and then expand the paths
        manifest_filepath = shard_manifests_if_needed(
            shard_manifests=shard_manifests,
            shard_strategy=shard_strategy,
            manifest_filepaths=manifest_filepath,
            world_size=world_size,
            global_rank=global_rank,
        )

        # If necessary, cache manifests from object store
        cache_datastore_manifests(manifest_filepaths=manifest_filepath)

        self.manifest_processor = ASRManifestProcessor(
            manifest_filepath=manifest_filepath,
            parser=parser,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=0,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            index_by_file_id=True,  # Must set this so the manifest lines can be indexed by file ID
        )

        self.len = self._compute_len()

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.return_sample_id = return_sample_id

        audio_tar_filepaths = expand_sharded_filepaths(
            sharded_filepaths=audio_tar_filepaths,
            shard_strategy=shard_strategy,
            world_size=world_size,
            global_rank=global_rank,
        )

        # Put together WebDataset
        self._dataset = wd.WebDataset(urls=audio_tar_filepaths, nodesplitter=None)

        if shuffle_n > 0:
            self._dataset = self._dataset.shuffle(shuffle_n)
        else:
            logging.info("WebDataset will not shuffle files within the tar files.")

        self._dataset = (
            self._dataset.rename(audio='wav;ogg;flac', key='__key__')
            .to_tuple('audio', 'key')
            .pipe(self._filter)
            .pipe(self._loop_offsets)
            .map(f=self._build_sample)
        )

    def _filter(self, iterator):
        """This function is used to remove samples that have been filtered out by ASRAudioText already.
        Otherwise, we would get a KeyError as _build_sample attempts to find the manifest entry for a sample
        that was filtered out (e.g. for duration).
        Note that if using multi-GPU training, filtering may lead to an imbalance in samples in each shard,
        which may make your code hang as one process will finish before the other.
        """

        class TarredAudioFilter:
            def __init__(self, collection):
                self.iterator = iterator
                self.collection = collection

            def __iter__(self):
                return self

            def __next__(self):
                while True:
                    audio_bytes, audio_filename = next(self.iterator)
                    file_id, _ = os.path.splitext(os.path.basename(audio_filename))
                    if file_id in self.collection.mapping:
                        return audio_bytes, audio_filename

        return TarredAudioFilter(self.manifest_processor.collection)

    def _loop_offsets(self, iterator):
        """This function is used to iterate through utterances with different offsets for each file.
        """

        class TarredAudioLoopOffsets:
            def __init__(self, collection):
                self.iterator = iterator
                self.collection = collection
                self.current_fn = None
                self.current_bytes = None
                self.offset_id = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.current_fn is None:
                    self.current_bytes, self.current_fn = next(self.iterator)
                    self.offset_id = 0
                else:
                    offset_list = self.collection.mapping[self.current_fn]
                    if len(offset_list) == self.offset_id + 1:
                        self.current_bytes, self.current_fn = next(self.iterator)
                        self.offset_id = 0
                    else:
                        self.offset_id += 1

                return self.current_bytes, self.current_fn, self.offset_id

        return TarredAudioLoopOffsets(self.manifest_processor.collection)

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, self.pad_id)

    def _build_sample(self, tup):
        """Builds the training sample by combining the data from the WebDataset with the manifest info.
        """
        audio_bytes, audio_filename, offset_id = tup

        # Grab manifest entry from self.manifest_preprocessor.collection
        file_id, _ = os.path.splitext(os.path.basename(audio_filename))
        manifest_idx = self.manifest_processor.collection.mapping[file_id][offset_id]
        manifest_entry = self.manifest_processor.collection[manifest_idx]

        offset = manifest_entry.offset
        if offset is None:
            offset = 0

        # Convert audio bytes to IO stream for processing (for SoundFile to read)
        audio_filestream = io.BytesIO(audio_bytes)
        features = self.featurizer.process(
            audio_filestream,
            offset=offset,
            duration=manifest_entry.duration,
            trim=self.trim,
            orig_sr=manifest_entry.orig_sr,
        )
        audio_filestream.close()

        # Audio features
        f, fl = features, torch.tensor(features.shape[0]).long()

        # Text features
        t, tl = manifest_entry.text_tokens, len(manifest_entry.text_tokens)

        self.manifest_processor.process_text_by_sample(sample=manifest_entry)

        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        if self.return_sample_id:
            return f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), manifest_idx
        else:
            return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def get_manifest_sample(self, sample_id):
        return self.manifest_processor.collection[sample_id]

    def __iter__(self):
        return self._dataset.__iter__()

    def _compute_len(self):
        if self.shard_manifests and torch.distributed.is_available() and torch.distributed.is_initialized():
            my_len = torch.tensor(len(self.manifest_processor.collection), dtype=torch.int32).cuda()
            torch.distributed.all_reduce(my_len)
            my_len = my_len.int()
            logging.info(f'Sharded manifests: Total length: {my_len}')
        else:
            my_len = len(self.manifest_processor.collection)

        return my_len

    def __len__(self):
        return self.len


class TarredAudioToBPEDataset(_TarredAudioToTextDataset):
    """
    A similar Dataset to the AudioToBPEDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToBPEDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple workers the number of shards should be divisible by world_size to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToBPEDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        tokenizer (TokenizerSpec): Either a Word Piece Encoding tokenizer (BERT),
            or a Sentence Piece Encoding tokenizer (BPE). The CTC blank
            symbol is automatically added later for models using ctc.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        use_start_end_token: Boolean which dictates whether to add [BOS] and [EOS]
            tokens to beginning and ending of speech respectively.
        pad_id (id): Token used to pad when collating samples in batches.
            If this is None, pads using 0s.
            Defaults to None.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.

            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                .. warning::

                    Replicated strategy allows every node to sample the entire set of available tarfiles,
                    and therefore more than one node may sample the same tarfile, and even sample the same
                    data points! As such, there is no assured guarantee that all samples in the dataset will be
                    sampled at least once during 1 epoch. Scattered strategy, on the other hand, on specific
                    occasions (when the number of shards is not divisible with ``world_size``), will not sample
                    the entire dataset. For these reasons it is not advisable to use tarred datasets as validation
                    or test datasets.

        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        return_sample_id (bool): whether to return the sample_id as a part of each sample
    """

    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional['nemo.collections.asr.parts.perturb.AudioAugmentor'] = None,
        shuffle_n: int = 0,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        trim: bool = False,
        use_start_end_token: bool = True,
        shard_strategy: str = "scatter",
        shard_manifests: bool = False,
        global_rank: int = 0,
        world_size: int = 0,
        return_sample_id: bool = False,
    ):
        if use_start_end_token and hasattr(tokenizer, "bos_id") and tokenizer.bos_id > 0:
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, "eos_id") and tokenizer.eos_id > 0:
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id > 0:
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                if isinstance(tokenizer, tokenizers.aggregate_tokenizer.AggregateTokenizer):
                    self.is_aggregate = True
                else:
                    self.is_aggregate = False
                self._tokenizer = tokenizer

            def __call__(self, *args):
                if isinstance(args[0], List) and self.is_aggregate:
                    t = []
                    for span in args[0]:
                        t.extend(self._tokenizer.text_to_ids(span['str'], span['lang']))
                    return t

                t = self._tokenizer.text_to_ids(*args)
                return t

        super().__init__(
            audio_tar_filepaths=audio_tar_filepaths,
            manifest_filepath=manifest_filepath,
            parser=TokenizerWrapper(tokenizer),
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            shuffle_n=shuffle_n,
            min_duration=min_duration,
            max_duration=max_duration,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            shard_strategy=shard_strategy,
            shard_manifests=shard_manifests,
            global_rank=global_rank,
            world_size=world_size,
            return_sample_id=return_sample_id,
        )


class BucketingDataset(IterableDataset):
    """
    A Dataset which wraps another IterableDataset and adopts it for bucketing
    Args:
        dataset (IterableDataset): The IterableDataset to get wrapped
        bucketing_batch_size (int): Number of samples to build a batch
    """

    def __init__(
        self, dataset: IterableDataset, bucketing_batch_size: int,
    ):
        self.wrapped_dataset = dataset
        self.bucketing_batch_size = bucketing_batch_size
        super().__init__()

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch[0], self.wrapped_dataset.pad_id)

    def __iter__(self):
        return BucketingIterator(
            wrapped_ds=self.wrapped_dataset._dataset, bucketing_batch_size=self.bucketing_batch_size
        ).__iter__()

    def __len__(self):
        return int(math.ceil(len(self.wrapped_dataset) / float(self.bucketing_batch_size)))


class BucketingIterator:
    def __init__(self, wrapped_ds, bucketing_batch_size):
        self.wrapped_ds = wrapped_ds
        self.wrapped_iter = None
        self.bucketing_batch_size = bucketing_batch_size

    def __iter__(self):
        self.wrapped_iter = iter(self.wrapped_ds)
        return self

    def __next__(self):
        batches = []
        for idx in range(self.bucketing_batch_size):
            try:
                sample = next(self.wrapped_iter)
            except StopIteration:
                break
            batches.append(sample)
        if len(batches) == 0:
            raise StopIteration
        return batches


class RandomizedChainDataset(ChainDataset):
    def __init__(self, datasets: Iterable[Dataset], rnd_seed=0) -> None:
        super(RandomizedChainDataset, self).__init__(list(datasets))
        self.rnd_gen = np.random.RandomState(rnd_seed)

    def __iter__(self):
        shuffled_order = self.rnd_gen.permutation(len(self.datasets))
        for dataset_idx in shuffled_order:
            d = self.datasets[dataset_idx]
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            for idx, x in enumerate(d):
                yield x
                # in case d is an infinite dataset, we want to break the loop
                # so that the other datasets get a chance to yield too
                if idx >= len(d) - 1:
                    break


def get_aqa_dataset_from_config(
    manifest_filepath: str,
    config: DictConfig,
    tokenizer,
    augmentor,
    sep_id: Optional[int] = None,
    answer_only_loss: bool = True,
    virtual_tokens: int = 0,
    num_samples: Optional[List[int]] = None,
):
    dataset = AudioQuestionAnswerDataset(
        manifest_filepath=manifest_filepath,
        tokenizer=tokenizer,
        sample_rate=config.sample_rate,
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=getattr(config, 'max_duration', None),
        min_duration=getattr(config, 'min_duration', None),
        max_utts=getattr(config, 'max_utts', -1),
        trim=getattr(config, 'trim_silence', False),
        channel_selector=getattr(config, 'channel_selector', None),
        max_seq_length=config.max_seq_length,
        min_seq_length=config.min_seq_length,
        add_bos=config.get('add_bos', False),
        add_eos=config.get('add_eos', True),
        add_sep=config.get('add_sep', False),
        sep_id=sep_id,
        max_num_samples=num_samples,
        seed=config.get('seed', 1234),
        separate_prompt_and_response_with_newline=config.get('separate_prompt_and_response_with_newline', True),
        answer_only_loss=answer_only_loss,
        truncation_field=config.get('truncation_field', 'context'),
        pad_to_max_length=config.get('pad_to_max_length', False),
        prompt_template=config.get('prompt_template', None),
        virtual_tokens=virtual_tokens,
        tokens_to_generate=config.get(
            'tokens_to_generate', 0
        ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
        input_key=config.get('input_key', 'input'),
        output_key=config.get('output_key', 'output'),
        end_string=config.get('end_string', None),
    )
    return dataset
