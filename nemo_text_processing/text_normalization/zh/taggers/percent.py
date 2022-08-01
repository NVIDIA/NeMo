import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst,insert_space,NEMO_DIGIT,NEMO_NOT_QUOTE
from nemo_text_processing.text_normalization.zh.taggers.number import Number
from pynini.lib import pynutil

class Percent(GraphFst):
    '''
        1.5%  -> tokens { { percent: "一点五" } }
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="percent", kind="classify", deterministic=deterministic)        
        
        percent_graph = (
            pynutil.insert("percent: \"") 
            + Number().graph_number
            + pynutil.delete("%") 
            + pynutil.insert("\"") 
        )

        self.fst = percent_graph.optimize()
