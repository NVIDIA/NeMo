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

import itertools
import sys
from collections import OrderedDict
from typing import List

import pynini
import regex as re
from denormalization.taggers.tokenize_and_classify import ClassifyFst
from denormalization.token_parser import TokenParser
from denormalization.verbalizers.verbalize import VerbalizeFst
from pynini.export import export
from tqdm import tqdm

tagger = ClassifyFst()
verbalizer = VerbalizeFst()
parser = TokenParser()


def reorder(d: OrderedDict) -> List[str]:
    l = []
    d_permutations = itertools.permutations(d.items())
    for perm in d_permutations:
        subl = [""]
        for k, v in perm:
            if isinstance(v, str):
                subl = ["".join(x) for x in itertools.product(subl, [f"{k}: \"{v}\" "])]
            elif isinstance(v, OrderedDict):
                rec = reorder(v)
                subl = ["".join(x) for x in itertools.product(subl, [f" {k} {{ "], rec, [f" }} "])]
            else:
                raise ValueError()
        l.extend(subl)
    return l


def create_tags_from_string(text: str) -> List[str]:
    parser(text)
    tokens = parser.parse()
    l = [""]
    for i in range(len(tokens)):
        reorders = reorder(tokens[i])
        l = ["".join(x) for x in itertools.product(l, reorders)]
    return list(l)


def find_tags(text: str) -> pynini.FstLike:
    """
    Given text use composite TaggerGraph to tag text
    Args:
        text: sentence
    Returns: Fst
    """
    lattice = text @ tagger.fst
    return lattice


def select_tag(lattice: pynini.FstLike) -> str:
    """
    Given tagged lattice return selected path
    Args:
        tagged_text: tagged text
    Returns: output string, list of tags
    """
    tagged_text = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
    return tagged_text


def find_verbalizer(tagged_text: str) -> pynini.FstLike:
    """
    Given tagged text, e.g. token {name: ""} token {money {fractional: ""}}, creates verbalization lattice
    This is context-independent.
    Args:
        tagged_text: input text
    Returns: Fst
    """
    lattice = tagged_text @ verbalizer.fst
    return lattice


def select_verbalizer(lattice: pynini.FstLike) -> str:
    """
    Given verbalized lattice return transduced string
    Args:
        lattice: verbalization lattice
    Returns: output string
    """
    output = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
    return output


def denormalize(text: str, verbose: bool) -> str:
    """
    main function. normalizes alphanumerical tokens in given text to its verbalized form:
    e.g. "12kg -> twelve kilograms"
    Args:
        text: string that may include semiotic classes.
    Returns: verbalized form in string format
    """

    text = pynini.escape(text)
    tagged_lattice = find_tags(text)
    tagged_text = select_tag(tagged_lattice)
    tags_reordered = create_tags_from_string(tagged_text)
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


def denormalize_identity(un_normalized: List[str], verbose=False) -> List[str]:
    """
    Identity normalizer. Returns input unchanged
    Args:
        un_normalized: input string
    Returns input string
    """
    return un_normalized


def denormalize_nemo(un_normalized: List[str], verbose=False) -> List[str]:
    res = []
    for input in tqdm(un_normalized):
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
    # Example usage:
    s = sys.argv[1]  # input string
    denormalize(s, verbose=True)
