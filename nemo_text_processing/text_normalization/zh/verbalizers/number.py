import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst,NEMO_NOT_QUOTE
from pynini.lib import pynutil

class Number(GraphFst):
    '''
        tokens { number { number: "一二三" } } -> 一二三
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="number", kind="verbalize", deterministic=deterministic)

        graph = (
            pynutil.delete('number: \"') 
            + pynini.closure(NEMO_NOT_QUOTE) 
            + pynutil.delete('\"') 
        )

        self.fst = self.delete_tokens(graph).optimize()
