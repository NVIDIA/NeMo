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

import os

import pandas as pd

from nemo.collections.nlp.data.dialogue.data_processor.data_processor import DialogueDataProcessor
from nemo.collections.nlp.data.dialogue.input_example.input_example import DialogueInputExample

__all__ = ['DialogueMellonQADataProcessor']


class DialogueMellonQADataProcessor(DialogueDataProcessor):
    """Data Processor for Mellon QA dialogues. 
    """

    def __init__(self, data_dir: str, tokenizer: object):
        """
        Constructs DialogueMSMarcoDataProcessor
        Args:
            data_dir: path to data directory
            tokenizer: tokenizer object
        """
        self.data_dir = data_dir
        self._tokenizer = tokenizer

    def open_csv(self, filename):
        """
        Reads file into a list
        """
        filename = os.path.join(self.data_dir, filename)
        with open(filename, "r", encoding="UTF-8") as f:
            df = pd.read_csv(filename)
        return df.to_dict(orient='index')

    def get_dialog_examples(self, dataset_split: str):
        """
        Process raw files into DialogueInputExample
        Args: 
            dataset_split: {train, dev, test}
        For the assistant dataset, there is no explicit dev set (instead uses the test set as the dev set)
        Therefore, this function creates a dev set and a new train set from the train set.
        This is done by taking every 10th example and putting it into the dev set,
        with all other examples going into the new train set.
        """

        examples = []

        raw_examples = self.open_csv('mellon_qa_data.csv')
        raw_examples = list(raw_examples.values())
        # filter out answers with no answer
        raw_examples = [
            example
            for example in raw_examples
            if example['Non Generative Question Answering '] and example['Generative Question Answering ']
        ]

        if dataset_split == "train":
            idxs = []
            for idx in range(len(raw_examples)):
                if idx % 10 != 0:
                    idxs.append(idx)
        elif dataset_split == "dev":
            idxs = []
            for idx in range(len(raw_examples)):
                if idx % 10 == 0:
                    idxs.append(idx)
        elif dataset_split == "test":
            idxs = list(range(len(raw_examples)))

        for i in idxs:
            utterance = str(raw_examples[i]['Question'])
            answer = str(raw_examples[i]['Non Generative Question Answering '])
            well_formed_answer = str(raw_examples[i]['Generative Question Answering '])
            passage = raw_examples[i]['Passage']
            input_example = {
                "utterance": utterance,
                "example_id": i,
                "labels": {"response": answer, "fluent_response": well_formed_answer, "passage": passage,},
            }
            example = DialogueInputExample(input_example)
            examples.append(example)
        return examples

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        return self.get_dialog_examples("train")

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.get_dialog_examples("dev")

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        return self.get_dialog_examples("test")
