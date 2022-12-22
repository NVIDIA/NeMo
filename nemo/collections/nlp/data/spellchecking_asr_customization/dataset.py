# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


from typing import Dict, List, Optional

import numpy as np
import torch
import braceexpand
import webdataset as wd
import pickle
from io import BytesIO

from nemo.collections.nlp.data.spellchecking_asr_customization.bert_example import BertExampleBuilder, read_input_file
from nemo.core.classes.dataset import Dataset, IterableDataset
from nemo.core.neural_types import ChannelType, IntType, LabelsType, MaskType, NeuralType
from nemo.utils import logging

__all__ = ["SpellcheckingAsrCustomizationDataset", "SpellcheckingAsrCustomizationTestDataset"]


class SpellcheckingAsrCustomizationDataset(Dataset):
    """
    Dataset as used by the SpellcheckingAsrCustomizationModel for training, validation, and inference
    pipelines.

    Args:
        input_file (str): path to tsv-file with data
        example_builder: instance of BertExampleBuilder
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), MaskType()),
            "segment_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_ids_for_subwords": NeuralType(('B', 'T'), ChannelType()),
            "input_mask_for_subwords": NeuralType(('B', 'T'), MaskType()),
            "segment_ids_for_subwords": NeuralType(('B', 'T'), ChannelType()),
            "character_pos_to_subword_pos": NeuralType(('B', 'T'), ChannelType()),
            "labels_mask": NeuralType(('B', 'T'), MaskType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
            "spans": NeuralType(('B', 'T', 'C'), IntType()),
        }

    def __init__(self, input_file: str, example_builder: BertExampleBuilder) -> None:
        self.examples = read_input_file(example_builder, input_file, infer=False)
        self.example_builder = example_builder
        self.pad_id = self.example_builder._pad_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        example = self.examples[idx]
        #-example.pad_to_max_length(
        #    self.example_builder._max_seq_length, self.example_builder._max_spans_length, self.example_builder._pad_id
        #)
        input_ids = np.array(example.features["input_ids"])
        input_mask = np.array(example.features["input_mask"])
        segment_ids = np.array(example.features["segment_ids"])
        input_ids_for_subwords = np.array(example.features["input_ids_for_subwords"])
        input_mask_for_subwords = np.array(example.features["input_mask_for_subwords"])
        segment_ids_for_subwords = np.array(example.features["segment_ids_for_subwords"])
        character_pos_to_subword_pos = np.array(example.features["character_pos_to_subword_pos"])
        labels_mask = np.array(example.features["labels_mask"])
        labels = np.array(example.features["labels"])
        spans = np.array(example.features["spans"])
        return (
            input_ids,
            input_mask,
            segment_ids,
            input_ids_for_subwords,
            input_mask_for_subwords,
            segment_ids_for_subwords,
            character_pos_to_subword_pos,
            labels_mask,
            labels,
            spans
        )

    def _collate_fn(self, batch):
        """collate batch of items
        Args:
            batch:  A list of tuples of (input_ids, input_mask, segment_ids, input_ids_for_subwords, input_mask_for_subwords, segment_ids_for_subwords, character_pos_to_subword_pos, labels_mask, labels, spans).
        """
        max_length = 0
        max_length_for_subwords = 0
        max_length_for_spans = 0
        for (
                input_ids,
                input_mask,
                segment_ids,
                input_ids_for_subwords,
                input_mask_for_subwords,
                segment_ids_for_subwords,
                character_pos_to_subword_pos,
                labels_mask,
                labels,
                spans
            ) in batch:
            if len(input_ids) > max_length:
                max_length = len(input_ids)
            if len(input_ids_for_subwords) > max_length_for_subwords:
                max_length_for_subwords = len(input_ids_for_subwords)
            if len(spans) > max_length_for_spans:
                max_length_for_spans = len(spans)

        padded_input_ids = []
        padded_input_mask = []
        padded_segment_ids = []
        padded_input_ids_for_subwords = []
        padded_input_mask_for_subwords = []
        padded_segment_ids_for_subwords = []
        padded_character_pos_to_subword_pos = []
        padded_labels_mask = []
        padded_labels = []
        padded_spans = []
        for (
                input_ids,
                input_mask,
                segment_ids,
                input_ids_for_subwords,
                input_mask_for_subwords,
                segment_ids_for_subwords,
                character_pos_to_subword_pos,
                labels_mask,
                labels,
                spans
            ) in batch:
            if len(input_ids) < max_length:
                pad_length = max_length - len(input_ids)
                padded_input_ids.append(np.pad(input_ids, pad_width=[0, pad_length], constant_values=self.example_builder._pad_id))
                padded_input_mask.append(np.pad(input_mask, pad_width=[0, pad_length], constant_values=0))
                padded_segment_ids.append(np.pad(segment_ids, pad_width=[0, pad_length], constant_values=0))
                padded_labels_mask.append(np.pad(labels_mask, pad_width=[0, pad_length], constant_values=0))
                padded_labels.append(np.pad(labels, pad_width=[0, pad_length], constant_values=0))
                padded_character_pos_to_subword_pos.append(np.pad(character_pos_to_subword_pos, pad_width=[0, pad_length], constant_values=0))
            else:
                padded_input_ids.append(input_ids)
                padded_input_mask.append(input_mask)
                padded_segment_ids.append(segment_ids)
                padded_labels_mask.append(labels_mask)
                padded_labels.append(labels)
                padded_character_pos_to_subword_pos.append(character_pos_to_subword_pos)

            if len(input_ids_for_subwords) < max_length_for_subwords:
                pad_length = max_length_for_subwords - len(input_ids_for_subwords)
                padded_input_ids_for_subwords.append(np.pad(input_ids_for_subwords, pad_width=[0, pad_length], constant_values=self.example_builder._pad_id))
                padded_input_mask_for_subwords.append(np.pad(input_mask_for_subwords, pad_width=[0, pad_length], constant_values=0))
                padded_segment_ids_for_subwords.append(np.pad(segment_ids_for_subwords, pad_width=[0, pad_length], constant_values=0))
            else:
                padded_input_ids_for_subwords.append(input_ids_for_subwords)
                padded_input_mask_for_subwords.append(input_mask_for_subwords)
                padded_segment_ids_for_subwords.append(segment_ids_for_subwords)

            if len(spans) < max_length_for_spans:
                pad_length = max_length_for_spans - len(spans)
                padded_spans.append(np.ones(max_length_for_spans, 3) * -1)    # pad value is [-1, -1, -1]
                padded_spans[-1][:spans.shape[0],:spans.shape[1]] = spans   # copy actual spans to the beginning
            else:
                padded_spans.append(spans)

        return (
            torch.IntTensor(padded_input_ids),
            torch.IntTensor(padded_input_mask),
            torch.IntTensor(padded_segment_ids),
            torch.IntTensor(padded_input_ids_for_subwords),
            torch.IntTensor(padded_input_mask_for_subwords),
            torch.IntTensor(padded_segment_ids_for_subwords),
            torch.IntTensor(padded_character_pos_to_subword_pos),
            torch.IntTensor(padded_labels_mask),
            torch.IntTensor(padded_labels),
            torch.IntTensor(padded_spans)
        )


class SpellcheckingAsrCustomizationTestDataset(Dataset):
    """
    Dataset for inference pipeline.

    Args:
        sents: list of strings
        example_builder: instance of BertExampleBuilder
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), MaskType()),
            "segment_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_ids_for_subwords": NeuralType(('B', 'T'), ChannelType()),
            "input_mask_for_subwords": NeuralType(('B', 'T'), MaskType()),
            "segment_ids_for_subwords": NeuralType(('B', 'T'), ChannelType()),
            "character_pos_to_subword_pos": NeuralType(('B', 'T'), ChannelType()),
        }

    def __init__(self, sents: List[str], example_builder: BertExampleBuilder) -> None:
        self.example_builder = example_builder
        self.examples = []
        for sent in sents:
            parts = sent.split("\t")
            if len(parts) < 2:
                print("Skip input with bad format: " + sent)
                continue
            hyp, ref = parts
            example = self.example_builder.build_bert_example(hyp=hyp, ref=ref, infer=True)
            if example is None:
                raise ValueError("Cannot build example from: " + sent)
            self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        example = self.examples[idx]
        example.pad_to_max_length(
            self.example_builder._max_seq_length, self.example_builder._max_spans_length, self.example_builder._pad_id
        )
        input_ids = np.array(example.features["input_ids"])
        input_mask = np.array(example.features["input_mask"])
        segment_ids = np.array(example.features["segment_ids"])
        input_ids_for_subwords = np.array(example.features["input_ids_for_subwords"])
        input_mask_for_subwords = np.array(example.features["input_mask_for_subwords"])
        segment_ids_for_subwords = np.array(example.features["segment_ids_for_subwords"])
        character_pos_to_subword_pos = np.array(example.features["character_pos_to_subword_pos"], dtype=np.int64)
        return (
            input_ids,
            input_mask,
            segment_ids,
            input_ids_for_subwords,
            input_mask_for_subwords,
            segment_ids_for_subwords,
            character_pos_to_subword_pos
        )


class TarredSpellcheckingAsrCustomizationDataset(IterableDataset):
    """
    This Dataset loads tarred tokenized pickle files.
    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    Additionally, please note that the len() of this DataLayer is assumed to be the number of tokens
    of the text data. Shard strategy is scatter - each node gets a unique set of shards, which are permanently
    pre-allocated and never changed at runtime.
    Args:
        text_tar_filepaths: a string (can be brace-expandable).
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 1.
        pad_token_id: id of pad token (used in collate_fn)
    """

    def __init__(
        self,
        text_tar_filepaths: str,
        shuffle_n: int = 1,
        global_rank: int = 0,
        world_size: int = 1,
        pad_token_id: int = -1,  # use real value or get error
    ):
        super(TarredSpellcheckingAsrCustomizationDataset, self).__init__()
        if pad_token_id < 0:
            raise(ValueError, "use non-negative pad_token_id: " + str(pad_token_id))

        self.pad_token_id = pad_token_id

        # Replace '(', '[', '<' and '_OP_' with '{'
        brace_keys_open = ['(', '[', '<', '_OP_']
        for bkey in brace_keys_open:
            if bkey in text_tar_filepaths:
                text_tar_filepaths = text_tar_filepaths.replace(bkey, "{")

        # Replace ')', ']', '>' and '_CL_' with '}'
        brace_keys_close = [')', ']', '>', '_CL_']
        for bkey in brace_keys_close:
            if bkey in text_tar_filepaths:
                text_tar_filepaths = text_tar_filepaths.replace(bkey, "}")

        # Brace expand
        text_tar_filepaths = list(braceexpand.braceexpand(text_tar_filepaths))

        logging.info("Tarred dataset shards will be scattered evenly across all nodes.")
        if len(text_tar_filepaths) % world_size != 0:
            logging.warning(
                f"Number of shards in tarred dataset ({len(text_tar_filepaths)}) is not divisible "
                f"by number of distributed workers ({world_size}). "
                f"Some shards will not be used ({len(text_tar_filepaths) % world_size})."
            )
        begin_idx = (len(text_tar_filepaths) // world_size) * global_rank
        end_idx = begin_idx + (len(text_tar_filepaths) // world_size)
        logging.info('Begin Index : %d' % (begin_idx))
        logging.info('End Index : %d' % (end_idx))
        text_tar_filepaths = text_tar_filepaths[begin_idx:end_idx]
        logging.info(
            "Partitioning tarred dataset: process (%d) taking shards [%d, %d)", global_rank, begin_idx, end_idx
        )

        self.tarpath = text_tar_filepaths

        # Put together WebDataset
        self._dataset = wd.WebDataset(urls=text_tar_filepaths, nodesplitter=None)

        if shuffle_n > 0:
            self._dataset = self._dataset.shuffle(shuffle_n, initial=shuffle_n)
        else:
            logging.info("WebDataset will not shuffle files within the tar files.")

        self._dataset = self._dataset.rename(pkl='pkl', key='__key__').to_tuple('pkl', 'key').map(f=self._build_sample)

    def _build_sample(self, fname):
        # Load file
        pkl_file, _ = fname
        pkl_file = BytesIO(pkl_file)
        data = pickle.load(pkl_file)  # loads np.int64 vector
        pkl_file.close()
        input_ids = data["input_ids"]
        input_mask = data["input_mask"]
        segment_ids = data["segment_ids"]
        input_ids_for_subwords = data["input_ids_for_subwords"]
        input_mask_for_subwords = data["input_mask_for_subwords"]
        segment_ids_for_subwords = data["segment_ids_for_subwords"]
        character_pos_to_subword_pos = data["character_pos_to_subword_pos"]
        labels_mask = data["labels_mask"]
        labels = data["labels"]
        spans = data["spans"]

        return (
            input_ids,
            input_mask,
            segment_ids,
            input_ids_for_subwords,
            input_mask_for_subwords,
            segment_ids_for_subwords,
            character_pos_to_subword_pos,
            labels_mask,
            labels,
            spans
        )

    def __iter__(self):
        return self._dataset.__iter__()

    def _collate_fn(self, batch):
        """collate batch of items
        Args:
            batch:  A list of tuples of (input_ids, input_mask, segment_ids, input_ids_for_subwords, input_mask_for_subwords, segment_ids_for_subwords, character_pos_to_subword_pos, labels_mask, labels, spans).
        """
        max_length = 0
        max_length_for_subwords = 0
        max_length_for_spans = 0
        for (
                input_ids,
                input_mask,
                segment_ids,
                input_ids_for_subwords,
                input_mask_for_subwords,
                segment_ids_for_subwords,
                character_pos_to_subword_pos,
                labels_mask,
                labels,
                spans
            ) in batch:
            if len(input_ids) > max_length:
                max_length = len(input_ids)
            if len(input_ids_for_subwords) > max_length_for_subwords:
                max_length_for_subwords = len(input_ids_for_subwords)
            if len(spans) > max_length_for_spans:
                max_length_for_spans = len(spans)

        padded_input_ids = []
        padded_input_mask = []
        padded_segment_ids = []
        padded_input_ids_for_subwords = []
        padded_input_mask_for_subwords = []
        padded_segment_ids_for_subwords = []
        padded_character_pos_to_subword_pos = []
        padded_labels_mask = []
        padded_labels = []
        padded_spans = []
        for (
                input_ids,
                input_mask,
                segment_ids,
                input_ids_for_subwords,
                input_mask_for_subwords,
                segment_ids_for_subwords,
                character_pos_to_subword_pos,
                labels_mask,
                labels,
                spans
            ) in batch:
            if len(input_ids) < max_length:
                pad_length = max_length - len(input_ids)
                padded_input_ids.append(np.pad(input_ids, pad_width=[0, pad_length], constant_values=self.pad_token_id))
                padded_input_mask.append(np.pad(input_mask, pad_width=[0, pad_length], constant_values=0))
                padded_segment_ids.append(np.pad(segment_ids, pad_width=[0, pad_length], constant_values=0))
                padded_labels_mask.append(np.pad(labels_mask, pad_width=[0, pad_length], constant_values=0))
                padded_labels.append(np.pad(labels, pad_width=[0, pad_length], constant_values=0))
                padded_character_pos_to_subword_pos.append(np.pad(character_pos_to_subword_pos, pad_width=[0, pad_length], constant_values=0))
            else:
                padded_input_ids.append(input_ids)
                padded_input_mask.append(input_mask)
                padded_segment_ids.append(segment_ids)
                padded_labels_mask.append(labels_mask)
                padded_labels.append(labels)
                padded_character_pos_to_subword_pos.append(character_pos_to_subword_pos)

            if len(input_ids_for_subwords) < max_length_for_subwords:
                pad_length = max_length_for_subwords - len(input_ids_for_subwords)
                padded_input_ids_for_subwords.append(np.pad(input_ids_for_subwords, pad_width=[0, pad_length], constant_values=self.pad_token_id))
                padded_input_mask_for_subwords.append(np.pad(input_mask_for_subwords, pad_width=[0, pad_length], constant_values=0))
                padded_segment_ids_for_subwords.append(np.pad(segment_ids_for_subwords, pad_width=[0, pad_length], constant_values=0))
            else:
                padded_input_ids_for_subwords.append(input_ids_for_subwords)
                padded_input_mask_for_subwords.append(input_mask_for_subwords)
                padded_segment_ids_for_subwords.append(segment_ids_for_subwords)

            if len(spans) < max_length_for_spans:
                pad_length = max_length_for_spans - len(spans)
                padded_spans.append(np.ones(max_length_for_spans, 3) * -1)   # pad value is [-1, -1, -1]
                padded_spans[-1][:spans.shape[0],:spans.shape[1]] = spans   # copy actual spans to the beginning
            else:
                padded_spans.append(spans)

        return (
            torch.IntTensor(padded_input_ids),
            torch.IntTensor(padded_input_mask),
            torch.IntTensor(padded_segment_ids),
            torch.IntTensor(padded_input_ids_for_subwords),
            torch.IntTensor(padded_input_mask_for_subwords),
            torch.IntTensor(padded_segment_ids_for_subwords),
            torch.IntTensor(padded_character_pos_to_subword_pos),
            torch.IntTensor(padded_labels_mask),
            torch.IntTensor(padded_labels),
            torch.IntTensor(padded_spans)
        )

