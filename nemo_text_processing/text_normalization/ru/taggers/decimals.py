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

from nemo_text_processing.text_normalization.data_loader_utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    delete_space = pynutil.delete(" ")

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g. 
        -12.5006 billion -> decimal { negative: "true" integer_part: "12"  fractional_part: "five o o six" quantity: "billion" }
        1 billion -> decimal { integer_part: "one" quantity: "billion" }

    cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = False):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.cardinal_numbers

        delimiter = (pynini.cross(",", " целых") | pynini.cross(",", " целых и")).optimize()
        decimal_endings = load_labels(get_abs_path("ru/data/decimal_endings.tsv"))
        optional_end = pynini.closure(
            insert_space + pynini.union(*[pynutil.insert(end[0]) for end in decimal_endings]), 0, 1
        )

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("минус", "\"true\"") + delete_extra_space, 0, 1
        )

        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + delimiter + pynutil.insert("\"")
        graph_fractional = pynutil.insert("fractional_part: \"") + cardinal_graph + optional_end + pynutil.insert("\"")

        self.final_graph_wo_negative = graph_integer + insert_space + graph_fractional
        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()


if __name__ == '__main__':
    from nemo_text_processing.text_normalization.ru.taggers.cardinal import CardinalFst
    from pynini.lib import rewrite

    fst = DecimalFst(CardinalFst())
    print(rewrite.rewrites("2,3", fst.fst))
