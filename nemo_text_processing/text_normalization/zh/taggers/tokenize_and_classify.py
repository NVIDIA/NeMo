import os
import time

import pynini
from pynini.lib import pynutil

# from nemo.utils import logging

from nemo_text_processing.text_normalization.zh.graph_utils import (GraphFst, NEMO_SIGMA)

from nemo_text_processing.text_normalization.zh.taggers.preprocessor import PreProcessor
from nemo_text_processing.text_normalization.zh.taggers.date import Date
from nemo_text_processing.text_normalization.zh.taggers.number import Number
from nemo_text_processing.text_normalization.zh.taggers.char import Char
from nemo_text_processing.text_normalization.zh.taggers.fraction import Fraction
from nemo_text_processing.text_normalization.zh.taggers.percent import Percent
from nemo_text_processing.text_normalization.zh.taggers.math_symbol import MathSymbol
from nemo_text_processing.text_normalization.zh.taggers.money import Money
from nemo_text_processing.text_normalization.zh.taggers.measure import Measure
from nemo_text_processing.text_normalization.zh.taggers.clock import Clock
from nemo_text_processing.text_normalization.zh.taggers.erhua import Erhua
from nemo_text_processing.text_normalization.zh.taggers.whitelist import Whitelist


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
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(
                cache_dir, f"zh_tn_{deterministic}_deterministic_{input_case}_{whitelist_file}_tokenize.far"
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            # logging.info(f'ClassifyFst.fst was restored from {far_file}.')
        else:
            # logging.info(f"Creating ClassifyFst grammars.")

            start_time = time.time()

            date = Date(deterministic=deterministic)
            number = Number(deterministic=deterministic)
            char = Char(deterministic=deterministic)
            fraction = Fraction(deterministic=deterministic)
            percent = Percent(deterministic=deterministic)
            math_symbol = MathSymbol(deterministic=deterministic)
            money = Money(deterministic=deterministic)
            measure = Measure(deterministic=deterministic)
            clock = Clock(deterministic=deterministic)
            erhua = Erhua(deterministic=deterministic)
            whitelist = Whitelist(deterministic=deterministic)

            # logging.debug(f"date: {time.time() - start_time: .2f}s -- {date_graph.num_states()} nodes")
            classify = pynini.union(
                pynutil.add_weight(date.fst,        0.4),
                pynutil.add_weight(fraction.fst,    0.5),
                pynutil.add_weight(percent.fst,     0.5),
                pynutil.add_weight(money.fst,       0.5),
                pynutil.add_weight(measure.fst,     0.5),
                pynutil.add_weight(clock.fst,       0.5),
                pynutil.add_weight(whitelist.fst,   0.3),
                pynutil.add_weight(number.fst,      1.2),
                pynutil.add_weight(math_symbol.fst, 1.5),
                pynutil.add_weight(erhua.fst,       2.0),
                pynutil.add_weight(char.fst,        200),
            )
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" } ")

            tagger = pynini.cdrewrite(token.optimize(),"","",NEMO_SIGMA).optimize()

            preprocessor = PreProcessor(
                remove_interjections = True,
                fullwidth_to_halfwidth = True, 
            )
            self.fst = preprocessor.fst @ tagger
