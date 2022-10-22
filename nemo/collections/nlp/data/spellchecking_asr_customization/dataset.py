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

from nemo.collections.nlp.data.spellchecking_asr_customization.bert_example import BertExampleBuilder, read_input_file
from nemo.core.classes.dataset import Dataset
from nemo.core.neural_types import ChannelType, IntType, LabelsType, MaskType, NeuralType

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
            "labels_mask": NeuralType(('B', 'T'), MaskType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
            "spans": NeuralType(('B', 'T', 'C'), IntType()),
        }

    def __init__(self, input_file: str, example_builder: BertExampleBuilder) -> None:
        self.examples = read_input_file(example_builder, input_file, infer=False)
        self.example_builder = example_builder

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
        labels_mask = np.array(example.features["labels_mask"])
        labels = np.array(example.features["labels"])
        spans = np.array(example.features["spans"])
        return input_ids, input_mask, segment_ids, labels_mask, labels, spans


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
        return input_ids, input_mask, segment_ids
