import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.zh.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)

from nemo_text_processing.text_normalization.zh.verbalizers.verbalize import VerbalizeFst
from nemo_text_processing.text_normalization.zh.verbalizers.postprocessor import PostProcessor

# from nemo.utils import logging


class VerbalizeFinalFst(GraphFst):
    """

    """
    def __init__(self, deterministic: bool = True, cache_dir: str = None, overwrite_cache: bool = False):
        super().__init__(name="verbalize_final", kind="verbalize", deterministic=deterministic)

        token_graph = VerbalizeFst(deterministic=deterministic)
        token_verbalizer = (
            pynutil.delete("tokens {") + 
            delete_space + token_graph.fst + delete_space +
            pynutil.delete(" }")
        )
        verbalizer = pynini.closure(delete_space + token_verbalizer + delete_space)

        postprocessor = PostProcessor(
            remove_puncts = False,
            to_upper = False,
            to_lower = False,
            tag_oov = False,
        )

        self.fst = (verbalizer @ postprocessor.fst).optimize()
