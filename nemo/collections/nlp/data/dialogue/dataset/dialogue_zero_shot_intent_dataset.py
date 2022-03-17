# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2019 The Google Research Authors.
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


from typing import Dict, Optional

from nemo.collections.nlp.data.glue_benchmark.data_processors import InputExample
from nemo.collections.nlp.data.glue_benchmark.glue_benchmark_dataset import GLUEDataset
from nemo.core.neural_types import CategoricalValuesType, ChannelType, LabelsType, MaskType, NeuralType

__all__ = ['DialogueZeroShotIntentDataset']


class DialogueZeroShotIntentDataset(GLUEDataset):
    """
    Dataset for training a NLI model for zero shot intent recognition. Similar to GLUE/MNLI
    dataset, but allows the user to specify which columns in the data files contain the
    premise, hypothesis, and gold label.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'labels': NeuralType(tuple('B'), CategoricalValuesType()),
        }

    def __init__(self, dataset_split: str, dialogues_processor: object, tokenizer, cfg):
        """
        Args:
            dataset_split: dataset split
            dialogues_processor: Data generator for dialogues
            tokenizer: tokenizer to split text into sub-word tokens
            cfg: config dict for dataset
                num_classes: number of classes in the data (should be either 2 or 3, corresponding to
                labels ['entailment', 'not_entailment'] or ["contradiction", "entailment", "neutral"])
        """
        self.cfg = cfg
        self.tokenizer = tokenizer
        if self.cfg.num_classes not in [2, 3]:
            raise ValueError("num_classes must be either 2 or 3!")
        self.label_list = (
            ["contradiction", "entailment", "neutral"]
            if self.cfg.num_classes == 3
            else ['not_entailment', 'entailment']
        )
        token_params = {
            'bos_token': None,
            'eos_token': tokenizer.eos_token,
            'pad_token': tokenizer.pad_token,
            'cls_token': tokenizer.cls_token,
            'sep_token_extra': tokenizer.eos_token if 'roberta' in tokenizer.name.lower() else None,
        }

        self.raw_features = dialogues_processor.get_dialog_examples(dataset_split)
        self.examples = self._create_examples(self.raw_features, dataset_split)
        self.features = self.convert_examples_to_features(
            self.examples,
            [0, 1, 2, 3],
            self.cfg.max_seq_length,
            tokenizer,
            output_mode="classification",
            **token_params,
        )

    def _create_examples(self, raw_features, dataset_split: str):
        """Creates examples for the training and dev sets."""
        examples = []
        for idx in range(len(raw_features)):
            ex = self.raw_features[idx].data
            user_utterance = ex["utterance"]
            intent = ex["labels"]["intent"]
            for candidate_idx, candidate_intent in enumerate(ex["possible_labels"]["intent"]):
                guid = "{}-{}-{}".format(dataset_split, idx, candidate_idx)
                text_a = user_utterance
                text_b = "{} {}".format(self.cfg.prompt_template, candidate_intent)
                label = 1 if candidate_intent == intent else 0
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
