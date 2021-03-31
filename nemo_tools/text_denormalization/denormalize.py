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

import itertools
import sys
from collections import OrderedDict
from typing import List

import pynini
from nemo_tools.text_denormalization.taggers.tokenize_and_classify_final import ClassifyFinalFst
from nemo_tools.text_denormalization.token_parser import PRESERVE_ORDER_KEY, TokenParser
from nemo_tools.text_denormalization.verbalizers.verbalize_final import VerbalizeFinalFst
from tqdm import tqdm

tagger = ClassifyFinalFst()
verbalizer = VerbalizeFinalFst()
parser = TokenParser()


def _permute(d: OrderedDict) -> List[str]:
    """
    Creates reorderings of dictionary elements and serializes as strings

    Args:
        d: (nested) dictionary of key value pairs

    Return permutations of different string serializations of key value pairs
    """
    l = []
    if PRESERVE_ORDER_KEY in d.keys():
        d_permutations = [d.items()]
    else:
        d_permutations = itertools.permutations(d.items())
    for perm in d_permutations:
        subl = [""]
        for k, v in perm:
            if isinstance(v, str):
                subl = ["".join(x) for x in itertools.product(subl, [f"{k}: \"{v}\" "])]
            elif isinstance(v, OrderedDict):
                rec = _permute(v)
                subl = ["".join(x) for x in itertools.product(subl, [f" {k} {{ "], rec, [f" }} "])]
            elif isinstance(v, bool):
                subl = ["".join(x) for x in itertools.product(subl, [f"{k}: true "])]
            else:
                raise ValueError()
        l.extend(subl)
    return l


def generate_permutations(tokens: List[dict]):
    """
    Generates permutations of string serializations of list of dictionaries

    Args:
        tokens: list of dictionaries

    Returns string serialization of list of dictionaries
    """

    def _helper(prefix: str, tokens: List[dict], idx: int):
        """
        Generates permutations of string serializations of given dictionary

        Args:
            tokens: list of dictionaries
            prefix: prefix string
            idx:    index of next dictionary

        Returns string serialization of dictionary
        """
        if idx == len(tokens):
            yield prefix
            return
        token_options = _permute(tokens[idx])
        for token_option in token_options:
            yield from _helper(prefix + token_option, tokens, idx + 1)

    return _helper("", tokens, 0)


def find_tags(text: str) -> pynini.FstLike:
    """
    Given text use tagger Fst to tag text

    Args:
        text: sentence

    Returns: tagged lattice
    """
    lattice = text @ tagger.fst
    return lattice


def select_tag(lattice: pynini.FstLike) -> str:
    """
    Given tagged lattice return shortest path

    Args:
        tagged_text: tagged text

    Returns: shortest path
    """
    tagged_text = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
    return tagged_text


def find_verbalizer(tagged_text: str) -> pynini.FstLike:
    """
    Given tagged text, e.g. token {name: ""} token {money {fractional: ""}}, creates verbalization lattice
    This is context-independent.

    Args:
        tagged_text: input text

    Returns: verbalized lattice
    """
    lattice = tagged_text @ verbalizer.fst
    return lattice


def select_verbalizer(lattice: pynini.FstLike) -> str:
    """
    Given verbalized lattice return shortest path

    Args:
        lattice: verbalization lattice

    Returns: shortest path
    """
    output = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
    return output


def denormalize(text: str, verbose: bool) -> str:
    """
    main function. normalizes spoken tokens in given text to its written form
        e.g. twelve kilograms -> 12 kg

    Args:
        text: string that may include semiotic classes.

    Returns: written form
    """

    text = pynini.escape(text)
    tagged_lattice = find_tags(text)
    tagged_text = select_tag(tagged_lattice)
    parser(tagged_text)
    tokens = parser.parse()
    tags_reordered = generate_permutations(tokens)
    for tagged_text in tags_reordered:
        tagged_text = pynini.escape(tagged_text)
        verbalizer_lattice = find_verbalizer(tagged_text)
        if verbalizer_lattice.num_states() == 0:
            continue
        output = select_verbalizer(verbalizer_lattice)
        if verbose:
            print(output)
        return output
    raise ValueError()


def denormalize_identity(texts: List[str], verbose=False) -> List[str]:
    """
    Identity function. Returns input unchanged

    Args:
        texts: input strings

    Returns input strings
    """
    return texts


def denormalize_nemo(texts: List[str], verbose=False) -> List[str]:
    """
    NeMo inverse text normalizer

    Args:
        texts: input strings

    Returns converted input strings
    """
    res = []
    for input in tqdm(texts):
        try:
            text = denormalize(input, verbose=verbose)
        except:
            raise Exception
        res.append(text)
    return res


DENORMALIZERS = {
    "identity": denormalize_identity,
    "nemo": denormalize_nemo,
}


if __name__ == "__main__":
    s = sys.argv[1]  # input string
    denormalize(s, verbose=True)
