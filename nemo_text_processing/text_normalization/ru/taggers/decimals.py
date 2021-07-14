# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from collections import defaultdict

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, NEMO_SPACE, GraphFst, insert_space
from nemo_text_processing.text_normalization.en.utils import load_labels
from nemo_text_processing.text_normalization.ru.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil

    delete_space = pynutil.delete(" ")

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


def prepare_labels_for_insertion(file_path: str):
    """
    Read the file and creates a union insertion graph

    Args:
    file_path: path to a file (single column)

    Returns fst that inserts labels from the file
    """
    labels = load_labels(file_path)
    map = defaultdict(list)
    for k, v in labels:
        map[k].append(v)

    for k in map:
        map[k] = insert_space + pynini.union(*[pynutil.insert(end) for end in map[k]])
    return map


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g. 
        -12.5006 billion -> decimal { negative: "true" integer_part: "12"  fractional_part: "five o o six" quantity: "billion" }
        1 billion -> decimal { integer_part: "one" quantity: "billion" }

    cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = False):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        integer_part = cardinal.cardinal_numbers | ordinal.ordinal_numbers
        cardinal_numbers_with_leading_zeros = cardinal.cardinal_numbers_with_leading_zeros

        delimiter_map = prepare_labels_for_insertion(get_abs_path("data/decimal_delimiter.tsv"))
        delimiter = (
            pynini.cross(",", "") + delimiter_map['@@decimal_delimiter@@'] + pynini.closure(pynutil.insert(" и"), 0, 1)
        ).optimize()

        decimal_endings_map = prepare_labels_for_insertion(get_abs_path("data/decimal_endings.tsv"))

        self.integer_part = integer_part + delimiter
        graph_integer = pynutil.insert("integer_part: \"") + self.integer_part + pynutil.insert("\"")

        graph_fractional = NEMO_DIGIT @ cardinal_numbers_with_leading_zeros + decimal_endings_map['10']
        graph_fractional |= (NEMO_DIGIT + NEMO_DIGIT) @ cardinal_numbers_with_leading_zeros + decimal_endings_map[
            '100'
        ]
        graph_fractional |= (
            NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT
        ) @ cardinal_numbers_with_leading_zeros + decimal_endings_map['1000']
        graph_fractional |= (
            NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT
        ) @ cardinal_numbers_with_leading_zeros + decimal_endings_map['10000']

        quantity = pynini.union("1000", "1000000", "1000000000")
        self.optional_quantity = (
            (pynini.string_file(get_abs_path("data/cardinals.tsv")).invert() @ quantity).project("input").optimize()
        )
        optional_quantity = pynutil.insert("quantity: \"") + self.optional_quantity + pynutil.insert("\"")
        optional_quantity = pynini.closure(pynini.accep(NEMO_SPACE) + optional_quantity, 0, 1)

        self.graph_fractional = graph_fractional
        graph_fractional = pynutil.insert("fractional_part: \"") + graph_fractional + pynutil.insert("\"")
        self.final_graph = (
            cardinal.optional_graph_negative + graph_integer + insert_space + graph_fractional + optional_quantity
        )

        self.final_graph = self.add_tokens(self.final_graph)
        self.fst = self.final_graph.optimize()

        # from pynini.lib.rewrite import top_rewrites
        # import pdb; pdb.set_trace()
        # print(top_rewrites("две целых и пять десятых тысячи", self.final_graph, 5))


if __name__ == '__main__':
    from nemo_text_processing.text_normalization.ru.taggers.cardinal import CardinalFst
    from pynini.lib import rewrite

    fst = DecimalFst(CardinalFst())
    print(rewrite.rewrites("2,3", fst.fst))
    import pdb

    pdb.set_trace()
    print()
