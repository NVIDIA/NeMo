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

import csv
import string
from abc import abstractmethod
from dataclasses import MISSING, dataclass, field
from os.path import exists, expanduser
from typing import List, Union

import nltk
from hydra.core.config_store import ConfigStore
from hydra.types import ObjectConf

from nemo.collections.vis.transforms.transforms import Transform
from nemo.utils import logging

# from nemo.utils.configuration_parsing import get_value_list_from_dictionary

nltk.download('punkt')

# Create the config store instance.
cs = ConfigStore.instance()


class TextTransform(Transform):
    """ Abstract class defining the string transform interface. """

    def __call__(self, input_batch: Union[str, List[str], List[List[str]]]):
        output_batch = []
        if type(input_batch) is list:
            for sentence in input_batch:
                # Process sentence as list of words.
                if type(sentence) is list:
                    output_sentence = [self.process_text(word) for word in sentence]
                # Process sentence as a single string.
                else:
                    output_sentence = self.process_text(sentence)
                output_batch.append(output_sentence)
            return output_batch
        # Process batch as a single string.
        else:
            return self.process_text(input_batch)

    @abstractmethod
    def process_text(self, sample: str):
        pass


class LowerCase(TextTransform):
    """ Transform responsible for turning text to lowercase. """

    def process_text(self, sample: str) -> str:
        return sample.lower()


@dataclass
class LowerCaseConfig:
    # Target class name.
    _target_: str = "nemo.collections.vis.transforms.LowerCase"


# Register the config.
cs.store(
    group="nemo.collections.vis.transforms",
    name="LowerCase",
    node=ObjectConf(target="nemo.collections.vis.transforms.LowerCase", params=LowerCaseConfig()),
)


class RemoveCharacters(TextTransform):
    """ Transform responsible for removing the characters. """

    def __init__(self, characters_to_remove: List[str]):
        self.characters_to_remove = characters_to_remove

    def process_text(self, sample: str) -> str:
        # Remove characters.
        for char in self.characters_to_remove:
            sample = sample.replace(char, ' ')
        return sample


@dataclass
class RemoveCharactersConfig:
    # Target class name.
    _target_: str = "nemo.collections.vis.transforms.RemoveCharacters"
    characters_to_remove: List[str] = field(default_factory=list)


# Register the config.
cs.store(
    group="nemo.collections.vis.transforms",
    name="RemoveCharacters",
    node=ObjectConf(target="nemo.collections.vis.transforms.RemoveCharacters", params=RemoveCharactersConfig()),
)


class RemovePunctuation(TextTransform):
    """ Transform responsible for removing punctuation. """

    def __init__(self):
        self.translator = str.maketrans('', '', string.punctuation)

    def process_text(self, sample: str) -> str:
        # Remove characters.
        return sample.translate(self.translator)


@dataclass
class RemovePunctuationConfig:
    # Target class name.
    _target_: str = "nemo.collections.vis.transforms.RemovePunctuation"


# Register the config.
cs.store(
    group="nemo.collections.vis.transforms",
    name="RemovePunctuation",
    node=ObjectConf(target="nemo.collections.vis.transforms.RemovePunctuation", params=RemovePunctuationConfig()),
)


class Tokenizer(TextTransform):
    """ Transform responsible for tokenization. """

    def __init__(self):
        # Tokenizer.
        self.tokenizer = nltk.tokenize.WhitespaceTokenizer()

    def process_text(self, sample: str) -> List[str]:
        # Tokenize.
        return self.tokenizer.tokenize(sample)


@dataclass
class TokenizerConfig:
    # Target class name.
    _target_: str = "nemo.collections.vis.transforms.Tokenizer"


# Register the config.
cs.store(
    group="nemo.collections.vis.transforms",
    name="Tokenizer",
    node=ObjectConf(target="nemo.collections.vis.transforms.Tokenizer", params=TokenizerConfig()),
)


class WordToIndex(TextTransform):
    """ Transform responsible for turning text index using the provided word mappings. """

    def __init__(self, word_mappings_filepath: str):
        """
        Loads (word:index) mappings from csv file.
        .. warning::
                There is an assumption that file will contain key,value pairs (no content checking for now!)

        Args:
            filepath: Path an name of file with encodings (absolute path + filename).

        Returns:
            Dictionary with word:index.
        """
        self.word_to_ix = {}

        filepath = expanduser(word_mappings_filepath)

        if not exists(filepath):
            logging.error("Cannot load word mappings from '{}' because the file does not exist".format(filepath))

        sniffer = csv.Sniffer()
        with open(filepath, mode='rt') as csvfile:
            # Check the presence of the header.
            first_bytes = str(csvfile.read(256))
            has_header = sniffer.has_header(first_bytes)
            # Rewind.
            csvfile.seek(0)
            reader = csv.reader(csvfile)
            # Skip the header row.
            if has_header:
                next(reader)
            # Read the remaining rows.
            for row in reader:
                if len(row) == 2:
                    self.word_to_ix[row[0]] = int(row[1])

        logging.info("Loaded mappings of size {}".format(len(self.word_to_ix)))

    def process_text(self, sample: str) -> int:
        return self.word_to_ix[sample]


@dataclass
class WordToIndexConfig:
    word_mappings_filepath: str = MISSING
    # Target class name.
    _target_: str = "nemo.collections.vis.transforms.WordToIndex"


# Register the config.
# cs.store(
#    group="nemo.collections.vis.transforms",
#    name="WordToIndex",
#    node=ObjectConf(target="nemo.collections.vis.transforms.WordToIndex", params=WordToIndexConfig()),
# ) # Problem: MISSING must be set.
