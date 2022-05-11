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

import os

from nemo_text_processing.text_normalization.en.graph_utils import (
    MIN_NEG_WEIGHT,
    NEMO_CHAR,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    generator_main,
)
from nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst

from nemo.utils import logging

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class PostProcessingFst:
    """
    Finite state transducer that post-processing an entire sentence after verbalization is complete, e.g.
    removes extra spaces around punctuation marks " ( one hundred and twenty three ) " -> "(one hundred and twenty three)"

    Args:
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
    """

    def __init__(self, cache_dir: str = None, overwrite_cache: bool = False):

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, "en_tn_post_processing.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["post_process_graph"]
            logging.info(f'Post processing graph was restored from {far_file}.')
        else:
            self.fst = self.punct_postprocess_graph()

            if far_file:
                generator_main(far_file, {"post_process_graph": self.fst})

    def punct_postprocess_graph(self):
        punct_marks_all = PunctuationFst().punct_marks

        # no_space_before_punct assume no space before them
        quotes = ["'", "\"", "``", "«"]
        dashes = ["-", "—"]
        open_close_symbols = [("<", ">"), ("{", "}"), ('"', '"'), ("``", "``"), ("``", "``"), ("(", ")"), ("“", "”")]
        allow_space_before_punct = ["&"] + quotes + dashes + [k[0] for k in open_close_symbols]
        no_space_before_punct = [m for m in punct_marks_all if m not in allow_space_before_punct]
        no_space_before_punct = pynini.union(*no_space_before_punct)
        delete_space = pynutil.delete(" ")

        # non_punct allows space
        # delete space before no_space_before_punct marks, if present
        non_punct = pynini.difference(NEMO_CHAR, no_space_before_punct).optimize()
        graph = (
            pynini.closure(non_punct)
            + pynini.closure(
                no_space_before_punct | pynutil.add_weight(delete_space + no_space_before_punct, MIN_NEG_WEIGHT)
            )
            + pynini.closure(non_punct)
        ).optimize()
        graph = pynini.closure(graph).optimize()

        open_close_marks_graph = (
            pynini.accep("(")
            + pynini.closure(delete_space, 0, 1)
            + NEMO_NOT_SPACE
            + NEMO_SIGMA
            + pynini.closure(delete_space, 0, 1)
            + pynini.accep(")")
        )
        for item in open_close_symbols:
            open, close = item
            open_close_marks_graph |= (
                pynini.accep(open)
                + pynini.closure(delete_space, 0, 1)
                + NEMO_NOT_SPACE
                + NEMO_SIGMA
                + pynini.closure(delete_space, 0, 1)
                + pynini.accep(close)
            )

        open_close_marks_graph = pynutil.add_weight(open_close_marks_graph, MIN_NEG_WEIGHT)
        open_close_marks_graph = NEMO_SIGMA + pynini.closure(open_close_marks_graph + NEMO_SIGMA)
        # graph = pynini.compose(graph.project("output").rmepsilon(), open_close_marks_graph).optimize()
        graph = pynini.compose(graph, open_close_marks_graph).optimize()
        # graph = graph.project("output").rmepsilon()
        return graph.optimize()
