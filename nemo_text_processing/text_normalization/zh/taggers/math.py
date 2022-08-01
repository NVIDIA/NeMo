import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst,NEMO_DIGIT
from nemo_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil
class MathSymbolFst(GraphFst):
    '''
        +  -> sign { sign: "加" }
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="sign", kind="classify", deterministic=deterministic)
        sign = pynini.string_file(get_abs_path("data/math/symbol.tsv"))
        #add your sign in data/math/symbol.tsv,this graph just convert sigh to character,you can add more 
        #cases with detailed cases 
        sign_graph = (
            pynutil.insert("sign: \"") 
            + sign 
            + pynutil.insert("\"")
        )
        sign_graph = self.add_tokens(sign_graph)
        self.fst = sign_graph.optimize()