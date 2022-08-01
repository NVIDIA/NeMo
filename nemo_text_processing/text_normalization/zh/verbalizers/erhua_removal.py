import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_SIGMA, GraphFst,NEMO_NOT_QUOTE
from pynini.lib import pynutil
class ErhuaRemovalFst(GraphFst):
    '''
        儿女
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="erhua", kind="verbalize", deterministic=deterministic)
        erhua = pynutil.delete("erhua: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        erhua = self.delete_tokens(erhua)
        self.fst = erhua.optimize()