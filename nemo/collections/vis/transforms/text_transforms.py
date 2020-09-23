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

from abc import abstractmethod, ABC
from typing import Optional, Any, List
from dataclasses import dataclass, field, MISSING

from hydra.types import ObjectConf
from hydra.core.config_store import ConfigStore

import string
import nltk

# from nemo.utils.configuration_parsing import get_value_list_from_dictionary

nltk.download('punkt')

# Create the config store instance.
cs = ConfigStore.instance()


class TextTransform(ABC):
    """ Abstract class defining the text transform interface. """

    @abstractmethod
    def __call__(self, batch):
        pass


class Compose:
    """ Class for composing text transforms. """

    def __init__(self, transforms: List[TextTransform]):
        self.transforms = transforms

    def __call__(self, batch):
        """ Processes the input batch by transforms - one by one. """
        for t in self.transforms:
            batch = t(batch)
        return batch


class StringTransform(TextTransform):
    """ Abstract class defining the string transform interface. """

    def __call__(self, input_batch):
        output_batch = []
        if type(input_batch) is list:
            for sentence in input_batch:
                # Process sentence as list of words.
                if type(sentence) is list:
                    output_sentence = [self.process_string(word) for word in sentence]
                # Process sentence as a single string.
                else:
                    output_sentence = self.process_string(sentence)
                output_batch.append(output_sentence)
            return output_batch
        # Process batch as a single string.
        else:
            return self.process_string(input_batch)

    @abstractmethod
    def process_string(self, sample: str):
        pass


class RemoveCharacters(StringTransform):
    """ Transform responsible for removing the characters. """

    def __init__(self, characters_to_remove):
        self.characters_to_remove = characters_to_remove

    def process_string(self, sample: str) -> str:
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


class RemovePunctuation(StringTransform):
    """ Transform responsible for removing punctuation. """

    def __init__(self):
        self.translator = str.maketrans('', '', string.punctuation)

    def process_string(self, sample: str) -> str:
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


class Tokenizer(StringTransform):
    """ Transform responsible for tokenization. """

    def __init__(self):
        # Tokenizer.
        self.tokenizer = nltk.tokenize.WhitespaceTokenizer()

    def process_string(self, sample: str) -> List[str]:
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
