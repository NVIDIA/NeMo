import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst,insert_space,NEMO_DIGIT,NEMO_NOT_QUOTE
from nemo_text_processing.text_normalization.zh.taggers.number import Number
from pynini.lib import pynutil

class Percent(GraphFst):
    '''
        tokens { percent: "一点五" }  ->  百分之一点五
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="percent", kind="verbalize", deterministic=deterministic)   

        graph = (
            pynutil.delete("percent: \"") +
            pynutil.insert("百分之") + 
            pynini.closure(NEMO_NOT_QUOTE, 1) +
            pynutil.delete("\"")
        )

        self.fst = graph.optimize()
