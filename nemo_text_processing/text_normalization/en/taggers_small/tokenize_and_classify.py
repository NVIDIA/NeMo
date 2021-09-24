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

import os

from nemo_text_processing.text_normalization.en import taggers, taggers_small
from nemo_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)

from nemo.utils import logging

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class ClassifyFstSmall(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence including punctuation.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File. 
    More details to deployment at NeMo/tools/text_processing_deployment.
    
    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
    """

    def __init__(
        self, input_case: str, deterministic: bool = True, cache_dir: str = None, overwrite_cache: bool = False
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"_{input_case}_en_tn_small_{deterministic}_deterministic.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f'ClassifyFst.fst was restored from {far_file}.')
        else:
            logging.info(f"Creating ClassifyFst grammars.")
            cardinal = taggers_small.CardinalFst(deterministic=deterministic)
            cardinal_graph = cardinal.fst
            # import pdb; pdb.set_trace()

            # ordinal = taggers_small.OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            # ordinal_graph = ordinal.fst

            decimal = taggers_small.DecimalFst(cardinal=taggers.CardinalFst(), deterministic=deterministic)
            decimal_graph = decimal.fst
            # fraction = FractionFst(deterministic=deterministic, cardinal=cardinal)
            # fraction_graph = fraction.fst
            #
            # measure = MeasureFst(cardinal=cardinal, decimal=decimal, fraction=fraction, deterministic=deterministic)
            # measure_graph = measure.fst
            # date_graph = DateFst(cardinal=cardinal, deterministic=deterministic).fst
            # import pdb; pdb.set_trace()
            word_graph = taggers.WordFst(deterministic=deterministic).fst
            # time_graph = TimeFst(cardinal=cardinal, deterministic=deterministic).fst
            # telephone_graph = TelephoneFst(deterministic=deterministic).fst
            electonic_graph = taggers.ElectronicFst(deterministic=deterministic).fst
            # money_graph = MoneyFst(cardinal=cardinal, decimal=decimal, deterministic=deterministic).fst
            whitelist_graph = taggers.WhiteListFst(input_case=input_case, deterministic=deterministic).fst
            punct_graph = taggers.PunctuationFst(deterministic=deterministic).fst

            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                # | pynutil.add_weight(time_graph, 1.1)
                # | pynutil.add_weight(date_graph, 1.09)
                | pynutil.add_weight(decimal_graph, 1.1)
                # | pynutil.add_weight(measure_graph, 1.1)
                | pynutil.add_weight(cardinal_graph, 1.1)
                # | pynutil.add_weight(ordinal_graph, 1.1)
                # | pynutil.add_weight(money_graph, 1.1)
                # | pynutil.add_weight(telephone_graph, 1.1)
                | pynutil.add_weight(electonic_graph, 1.1)
                # | pynutil.add_weight(fraction_graph, 1.1)
                | pynutil.add_weight(word_graph, 100)
            )

            punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=1.1) + pynutil.insert(" }")
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(delete_extra_space + token_plus_punct)
            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
