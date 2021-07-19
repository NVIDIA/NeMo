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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.ru.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    # Add placeholders for global variables
    graph_teen = None
    graph_digit = None
    ties_graph = None

    PYNINI_AVAILABLE = True


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g. 
        jan. 5, 2012 -> date { month: "january" day: "five" year: "twenty twelve" preserve_order: true }
        jan. 5 -> date { month: "january" day: "five" preserve_order: true }
        5 january 2012 -> date { day: "five" month: "january" year: "twenty twelve" preserve_order: true }
        2012-01-05 -> date { year: "twenty twelve" month: "january" day: "five" }
        2012.01.05 -> date { year: "twenty twelve" month: "january" day: "five" }
        2012/01/05 -> date { year: "twenty twelve" month: "january" day: "five" }
        2012 -> date { year: "twenty twelve" }

    Args:
        ordinal: OrdinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, number_names: GraphFst, ordinal: GraphFst, deterministic: bool):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        # Ru format: DD-MM-YYYY or DD-MM-YY
        month_abbr_to_names = pynini.string_file(get_abs_path("data/whitelist.tsv")).optimize()

        delete_sep = pynutil.add_weight(pynini.cross("-", " "), 1.09) | pynutil.add_weight(
            pynini.cross(pynini.union("/", "."), " "), 1.1
        )

        numbers = number_names.ordinal_number_names

        day = (
            (pynutil.add_weight(pynini.cross("0", ""), -0.1) | pynini.union("1", "2", "3")) + NEMO_DIGIT
        ) | NEMO_DIGIT
        day = (pynutil.insert("day: \"") + pynini.compose(day, numbers) + pynutil.insert("\"")).optimize()

        month_number_to_abbr = pynini.string_file(get_abs_path("data/months/numbers.tsv")).optimize()
        month_number_to_abbr = (
            ((pynini.union("0", "1") + NEMO_DIGIT) | NEMO_DIGIT).optimize() @ month_number_to_abbr
        ).optimize()

        month_name = ((month_number_to_abbr @ month_abbr_to_names) | month_abbr_to_names).optimize()
        month = (pynutil.insert("month: \"") + month_name + pynutil.insert("\"")).optimize()
        year = pynini.compose(((NEMO_DIGIT ** 4) | (NEMO_DIGIT ** 2)), numbers).optimize()
        year_word_singular = ["год", "года", "году", "годом", "годе"]
        year_word_plural = ["годы", "годов", "годам", "годами", "годам"]

        year_word = pynini.cross("г.", pynini.union(*year_word_singular))
        year_word |= pynini.cross("гг.", pynini.union(*year_word_plural))
        year_word = pynini.closure(insert_space + year_word, 0, 1)
        year = pynini.closure(
            delete_sep + pynutil.insert("year: \"") + year + year_word + pynutil.insert("\""), 0, 1
        ).optimize()

        tagger_graph = day + delete_sep + month + year

        # Verbalizer
        day = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        year = (
            pynutil.delete("year:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + delete_space
            + pynutil.delete("\"")
        )
        year = pynini.closure(delete_extra_space + year, 0, 1)
        graph_dmy = day + delete_extra_space + month + year
        verbalizer_graph = graph_dmy + delete_space

        self.final_graph = pynini.compose(tagger_graph, verbalizer_graph).optimize()
        self.fst = pynutil.insert("day: \"") + self.final_graph + pynutil.insert("\"")
        self.fst = self.add_tokens(self.fst).optimize()

        # from pynini.lib.rewrite import top_rewrites
        # import pdb;
        # pdb.set_trace()
        # print(top_rewrites("01.02.2001", self.fst, 5))
        # print()
