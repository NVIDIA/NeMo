# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.es.taggers.cardinal import CardinalFst
#from nemo_text_processing.text_normalization.de.taggers.date import DateFst
from nemo_text_processing.text_normalization.es.taggers.decimals import DecimalFst
#from nemo_text_processing.text_normalization.es.taggers.electronic import ElectronicFst
from nemo_text_processing.text_normalization.es.taggers.fraction import FractionFst
#from nemo_text_processing.text_normalization.es.taggers.measure import MeasureFst
#from nemo_text_processing.text_normalization.es.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.es.taggers.ordinal import OrdinalFst
#from nemo_text_processing.text_normalization.es.taggers.telephone import TelephoneFst
#from nemo_text_processing.text_normalization.es.taggers.time import TimeFst
#from nemo_text_processing.text_normalization.es.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.es.taggers.word import WordFst
from nemo_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
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


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence, that is lower cased.
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
        deterministic: bool = False,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)
        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(
                cache_dir, f"_{input_case}_es_tn_{deterministic}_deterministic{whitelist_file}.far"
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars. This might take some time...")

            self.cardinal = CardinalFst(deterministic=deterministic)
            cardinal_graph = self.cardinal.fst

            self.ordinal = OrdinalFst(cardinal=self.cardinal, deterministic=deterministic)
            ordinal_graph = self.ordinal.fst

            self.decimal = DecimalFst(cardinal=self.cardinal, deterministic=deterministic)
            decimal_graph = self.decimal.fst

            self.fraction = FractionFst(ordinal=self.ordinal, deterministic=deterministic)
            fraction_graph = self.fraction.fst
            # self.measure = MeasureFst(
            #     cardinal=self.cardinal, decimal=self.decimal, fraction=self.fraction, deterministic=deterministic
            # )
            # measure_graph = self.measure.fst
            # self.date = DateFst(cardinal=self.cardinal, deterministic=deterministic)
            # date_graph = self.date.fst
            word_graph = WordFst(deterministic=deterministic).fst
            # self.time = TimeFst(deterministic=deterministic)
            # time_graph = self.time.fst
            # self.telephone = TelephoneFst(cardinal=self.cardinal, deterministic=deterministic)
            # telephone_graph = self.telephone.fst
            # self.electronic = ElectronicFst(deterministic=deterministic)
            # electronic_graph = self.electronic.fst
            # self.money = MoneyFst(cardinal=self.cardinal, decimal=self.decimal, deterministic=deterministic)
            # money_graph = self.money.fst
            # self.whitelist = WhiteListFst(input_case=input_case, deterministic=deterministic, input_file=whitelist)
            # whitelist_graph = self.whitelist.fst
            punct_graph = PunctuationFst(deterministic=deterministic).fst

            classify = (
                # pynutil.add_weight(whitelist_graph, 1.01)
                # | pynutil.add_weight(time_graph, 1.1)
                # | pynutil.add_weight(measure_graph, 1.1)
                  pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.1)
                # | pynutil.add_weight(date_graph, 1.1)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                # | pynutil.add_weight(money_graph, 1.1)
                # | pynutil.add_weight(telephone_graph, 1.1)
                # | pynutil.add_weight(electronic_graph, 1.1)
                #| pynutil.add_weight(word_graph, 100)
            )

            punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=1.1) + pynutil.insert(" }")
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(pynutil.add_weight(delete_extra_space, 1.1) + token_plus_punct)
            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")


tagger = ClassifyFst(input_case="lower_cased",
        deterministic= False,
        cache_dir = ".",
        overwrite_cache= True,
        whitelist= None,)
from pynini.lib import rewrite
cardinals = [
"501",
    "201",
    "301",
    "401",
    "801",
    "1",
    "101",
    "2",
    "3",
    "4",
    "0",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "23",
    "24",
    "25",
    "26",
    "28",
    "59",
    "99",
    "-55",
    "100",
    "105",
    "115",
    "293",
    "200",
    "508",
    "1000",
    "2000",
    "1005",
    "-200",
    "-508",
    "-1000",
    "-1 000",
    "-1.000",
    "-2000",
    "-1.005",
    "-1105",
    "-2303",
    "-332303",
    "-25235303",
    "2.312.312.303",
    "23566303",
    "22322303",
    "555555222303",
    "111111111111",
    "100000000000000",
    "10 000 000 000 001 000 000",
    "10 000 001 110 001 010 101",
    "100100100100100100",
    "100",
    "123456789",
]

decimals = [
	"0,1",
	"0,01",
	"0,010",
	"1,0101",
	"0,0",
	"1,0",
	"1,00",
    "1,1",
	"233,32",
	"32,22 millones",
	"320 320,22 millones",
	"5.002,232",
	"3,2 trillones",
	"3 millones",
	"3 000 millones",
	"3000 millones",
	"3.000 millones",
	"1 millón",
	"1 000 millones",
	"1000 millones",
	"1.000 millones",
	"2,33302 millones", 
	"1 millón",
	"1,5332 millón",
	"1,53322 millón",
	"1,53321 millón",
	"101,010101 millones",
	"1,010100"]

ordinals = [
	"1000",
	"2000",
	"9000",
	"7000",
	"3000",
	"5000",
	"1.º",
	"1.ª",
	"3.ª",
	"4.ª",
	"5.ª",
	"6.ª",	
	"7.ª",
	"40.º",
	"21.º",
	"21.ᵉʳ", 
	"21.ª",
	"134.ª",
	"1248.º",
	"X",
	"I",
	"i", 
	"323.ᵉʳ",
	"8.ª",
	"9.º", 
	"100.ª",
	"11.º", 
	"12.º", 
	"13.º", 
	"14.º", 
	"15.º", 
	"16.º", 
	"17.º", 	
	"21.º",
	"28.º",
	"121.ª",
	"999.ª",
	"1000.ª",
	"1021.ª",
	"2000.ª",
	"500 000.º", 
	"1000.ª",
	"2050.ª",
	"2051.ª",
	"5432.ª",	
	"21001.ª",
	"21 001.ª",
]

fractions = [
	"1/2000",
	"1/5000",
	"1/501",
	"1/550",
	"1/1000000",
	"1/2000000",
	"1/2323422",
	"1/2",
	"1/3",
	"1/4",
	"1/5",
	"1/6",
	"1/7",
	"1/8",
	"1/9",
	"1/10",
	"1/11",
	"1/12",
	"1/20",
	"1/21",
	"1/100",
	"1/101",
	"1/1000",
	"1/2000",
	"2 3/5",
	"4 1/2",
	"2/3",	
	"4/6",
	"9/2",
	"4123123 4/5",	
	"4001 1/5",	
	"400100100 1/5"
]

for item in fractions:
	print(item, rewrite.rewrites(item, tagger.fst))
	print()