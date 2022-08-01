import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_CHAR, NEMO_SIGMA, GraphFst,NEMO_NOT_SPACE,NEMO_NOT_QUOTE
from pynini.lib import pynutil
from nemo_text_processing.text_normalization.zh.utils import get_abs_path

class Char(GraphFst):
    '''
        tokens { char: "你" } -> 你
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="char", kind="verbalize", deterministic=deterministic)

        graph = pynutil.delete("char: \"") + NEMO_NOT_QUOTE + pynutil.delete("\"")
        self.fst = graph.optimize()
