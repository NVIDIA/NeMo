import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst,insert_space,NEMO_DIGIT,NEMO_CHAR
from pynini.lib import pynutil

class Fraction(GraphFst):
    '''
        1/5  -> tokens { fraction { numerator: "1" denominator: "5" } }
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        numerator = pynini.closure(NEMO_DIGIT,1) + pynutil.delete('/')
        denominator = pynini.closure(NEMO_DIGIT,1)
        graph = (
            pynutil.insert("numerator: \"") 
            + numerator 
            + pynutil.insert("\"") + insert_space 
            + pynutil.insert("denominator: \"") 
            + denominator 
            + pynutil.insert("\"") 
        )

        self.fst = self.add_tokens(graph).optimize()
