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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.text_normalization.en.taggers.abbreviation import AbbreviationFst
from nemo_text_processing.text_normalization.en.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.en.taggers.date import DateFst
from nemo_text_processing.text_normalization.en.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.en.taggers.electronic import ElectronicFst
from nemo_text_processing.text_normalization.en.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.en.taggers.measure import MeasureFst
from nemo_text_processing.text_normalization.en.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.en.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.en.taggers.range import RangeFst as RangeFst
from nemo_text_processing.text_normalization.en.taggers.roman import RomanFst
from nemo_text_processing.text_normalization.en.taggers.telephone import TelephoneFst
from nemo_text_processing.text_normalization.en.taggers.time import TimeFst
from nemo_text_processing.text_normalization.en.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.en.taggers.word import WordFst
from nemo_text_processing.text_normalization.en.verbalizers.abbreviation import AbbreviationFst as vAbbreviation
from nemo_text_processing.text_normalization.en.verbalizers.cardinal import CardinalFst as vCardinal
from nemo_text_processing.text_normalization.en.verbalizers.date import DateFst as vDate
from nemo_text_processing.text_normalization.en.verbalizers.decimal import DecimalFst as vDecimal
from nemo_text_processing.text_normalization.en.verbalizers.electronic import ElectronicFst as vElectronic
from nemo_text_processing.text_normalization.en.verbalizers.fraction import FractionFst as vFraction
from nemo_text_processing.text_normalization.en.verbalizers.measure import MeasureFst as vMeasure
from nemo_text_processing.text_normalization.en.verbalizers.money import MoneyFst as vMoney
from nemo_text_processing.text_normalization.en.verbalizers.ordinal import OrdinalFst as vOrdinal
from nemo_text_processing.text_normalization.en.verbalizers.range import RangeFst as vRangeFst
from nemo_text_processing.text_normalization.en.verbalizers.roman import RomanFst as vRoman
from nemo_text_processing.text_normalization.en.verbalizers.telephone import TelephoneFst as vTelephone
from nemo_text_processing.text_normalization.en.verbalizers.time import TimeFst as vTime

from nemo.utils import logging

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class ClassifyFst(GraphFst):
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
        whitelist: path to a file with whitelist replacements
    """

    def __init__(
        self,
        input_case: str,
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = True,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != 'None':
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(
                cache_dir, f"_{input_case}_en_tn_{deterministic}_deterministic{whitelist_file}.far"
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode='r')['tokenize_and_classify']
            # self.whitelist_word = pynini.Far(far_file, mode='r')['whitelist_word']
            no_digits = pynini.closure(pynini.difference(NEMO_CHAR, NEMO_DIGIT))
            self.fst_no_digits = pynini.compose(self.fst, no_digits).optimize()
            logging.info(f'ClassifyFst.fst was restored from {far_file}.')
        else:
            logging.info(f'Creating ClassifyFst grammars. This might take some time...')
            # TAGGERS
            cardinal = CardinalFst(deterministic=True, baseline=True)
            cardinal_tagger = cardinal
            cardinal_graph = cardinal.fst

            ordinal = OrdinalFst(cardinal=cardinal, deterministic=True)
            ordinal_graph = ordinal.fst

            decimal = DecimalFst(cardinal=cardinal, deterministic=True)
            decimal_graph = decimal.fst
            fraction = FractionFst(deterministic=True, cardinal=cardinal)
            fraction_graph = fraction.fst

            # use False deterministic for measure to add range graph to cardinal options
            measure = MeasureFst(
                cardinal=cardinal, decimal=decimal, fraction=fraction, deterministic=True, baseline=True
            )
            measure_graph = measure.fst
            date = DateFst(cardinal=cardinal, deterministic=True)
            date_graph = date.fst
            word_graph = WordFst(deterministic=deterministic).fst
            time_graph = TimeFst(cardinal=cardinal, deterministic=True).fst
            telephone_graph = TelephoneFst(deterministic=True).fst
            electronic_graph = ElectronicFst(deterministic=True).fst
            money_graph = MoneyFst(cardinal=cardinal, decimal=decimal, deterministic=True).fst
            whitelist = WhiteListFst(input_case=input_case, deterministic=True, input_file=whitelist, baseline=True)
            whitelist_graph = whitelist.fst
            punct_graph = PunctuationFst(deterministic=True).fst

            v_time_graph = vTime(deterministic=True).fst
            time_final = pynini.compose(time_graph, v_time_graph)
            range_graph = RangeFst(time=time_final, deterministic=deterministic).fst

            classify = (
                pynutil.add_weight(time_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(measure_graph, 1.1)
                | pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(telephone_graph, 1.1)
                | pynutil.add_weight(electronic_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.1)
                | pynutil.add_weight(money_graph, 1.1)
                | pynutil.add_weight(date_graph, 1.09)
                | pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(range_graph, 1.1)
            ).optimize()

            roman_graph = RomanFst(deterministic=deterministic).fst
            classify |= pynutil.add_weight(roman_graph, 100)

            # range_graph = RangeFst(time=time_final, cardinal=cardinal_tagger, deterministic=deterministic).fst
            # v_range_graph = vRangeFst(deterministic=deterministic).fst
            # classify |= pynutil.insert("tokens { ") + "name: \"" + pynutil.add_weight(pynini.compose(range_graph, v_range_graph), 1.5) + pynutil.insert("\" }")
            classify |= pynutil.add_weight(word_graph, 100)

            punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=0) + pynutil.insert(" }")
            punct = pynini.closure(
                pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
                | (pynutil.insert(" ") + punct),
                1,
            )
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(
                (
                    pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
                    | (pynutil.insert(" ") + punct + pynutil.insert(" "))
                )
                + token_plus_punct
            )

            graph = delete_space + graph + delete_space
            graph |= punct

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                # generator_main(f"{far_file.replace('.far', '_whitelist.far')}", {"whitelist_word": self.whitelist_word})
                logging.info(f'ClassifyFst grammars are saved to {far_file}.')
