import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_SIGMA, GraphFst,NEMO_NOT_QUOTE
from pynini.lib import pynutil

class Whitelist(GraphFst):
    '''
        tokens { whitelist: "ATM" } -> A T M
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="whitelist", kind="verbalize", deterministic=deterministic)

        graph = (
            pynutil.delete("whitelist: \"") 
            + pynini.closure(NEMO_NOT_QUOTE) 
            + pynutil.delete("\"")
        )

        self.fst = graph.optimize()
