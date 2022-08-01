import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_CHAR, GraphFst,NEMO_NOT_SPACE,NEMO_DIGIT,NEMO_ALPHA,NEMO_PUNCT
from pynini.lib import pynutil, utf8
from nemo_text_processing.text_normalization.zh.utils import get_abs_path,load_labels

class Char(GraphFst):
    '''
        你 -> tokens { char: "你" }
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="char", kind="classify", deterministic=deterministic)

        graph = pynutil.insert("char: \"") + NEMO_CHAR + pynutil.insert("\"")
        self.fst = self.add_tokens(graph).optimize()
        self.fst = graph.optimize()
