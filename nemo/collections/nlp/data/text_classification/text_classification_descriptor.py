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

from typing import List

from nemo import logging
from nemo.collections.nlp.data.data_utils.data_preprocessing import (
    fill_class_weights,
    get_freq_weights,
    get_label_stats,
    if_exist,
)

__all__ = ['TextClassificationDataDesc']


class TextClassificationDataDesc:
    def __init__(self, data_dir: str, modes: List[str] = ['train', 'test', 'dev']):
        """A descriptor class that reads all the data and calculates some stats of the data and also calculates the class weights to be used for class balancing
        Args:
            data_dir: the path to the data folder
            modes: list of the modes to read, it can be from ["train", "test", "dev"] by default. It is going to look for the data files at {data_dir}/{mode}.tsv
        """
        self.data_dir = data_dir

        class_weights_dict = None
        max_label_id = 0
        for mode in modes:
            if not if_exist(self.data_dir, [f'{mode}.tsv']):
                logging.info(f'Stats calculation for {mode} mode is skipped as {mode}.tsv was not found.')
                continue

            input_file = f'{self.data_dir}/{mode}.tsv'
            with open(input_file, 'r') as f:
                input_lines = f.readlines()[1:]  # Skipping headers at index 0

            try:
                int(input_lines[0].strip().split()[-1])
            except ValueError:
                logging.warning(f'No numerical labels found for {mode}.tsv.')
                raise

            queries, raw_sentences = [], []
            for input_line in input_lines:
                parts = input_line.strip().split()
                label = int(parts[-1])
                raw_sentences.append(label)
                queries.append(' '.join(parts[:-1]))

            infold = input_file[: input_file.rfind('/')]

            logging.info(f'Three most popular classes in {mode} dataset')
            total_sents, sent_label_freq, max_id = get_label_stats(
                raw_sentences, infold + f'/{mode}_sentence_stats.tsv'
            )
            max_label_id = max(max_label_id, max_id)

            if mode == 'train':
                class_weights_dict = get_freq_weights(sent_label_freq)
                logging.info(f'Class Weights: {class_weights_dict}')

            logging.info(f'Total Sentences: {total_sents}')
            logging.info(f'Sentence class frequencies - {sent_label_freq}')

        if class_weights_dict is None:
            raise FileNotFoundError(f"Could not find any of the data files!")

        self.class_weights = fill_class_weights(class_weights_dict, max_label_id)

        self.num_labels = max_label_id + 1
