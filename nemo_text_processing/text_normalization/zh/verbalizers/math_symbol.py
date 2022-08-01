import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_SIGMA, NEMO_NOT_QUOTE ,GraphFst
from nemo_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil

class MathSymbol(GraphFst):
    '''
        tokens { sign: "加" }  -> 加
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="sign", kind="verbalize", deterministic=deterministic)

        graph = (
            pynutil.delete('sign: \"') 
            + pynini.closure(NEMO_NOT_QUOTE) 
            + pynutil.delete('\"') 
        )

        self.fst = graph.optimize()
