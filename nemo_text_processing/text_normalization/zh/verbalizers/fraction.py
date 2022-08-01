import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst,NEMO_NOT_QUOTE
from nemo_text_processing.text_normalization.zh.taggers.number import Number
from pynini.lib import pynutil

class Fraction(GraphFst):
    '''
        tokens { fraction { denominator: "5" numerator: "1" } } -> 五分之一      
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)   

        denominator = pynutil.delete("denominator: \"") + Number().graph_number + pynutil.delete("\"")
        numerator = pynutil.delete("numerator: \"") + Number().graph_number + pynutil.delete("\"") 
        graph = denominator + pynutil.delete(" ") + pynutil.insert("分之") + numerator

        self.fst = self.delete_tokens(graph).optimize()
