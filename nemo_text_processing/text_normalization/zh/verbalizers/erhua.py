import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_SIGMA, GraphFst,NEMO_NOT_QUOTE
from pynini.lib import pynutil

class Erhua(GraphFst):
    '''
        tokens { erhua: "儿" } -> 儿
        tokens { erhua_whitelist: "儿女" } -> 儿女
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="erhua", kind="verbalize", deterministic=deterministic)

        remove_erhua = pynutil.delete("erhua: \"") + pynutil.delete("儿") + pynutil.delete("\"")
        retain_erhua_whitelist = pynutil.delete("erhua_whitelist: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        graph = remove_erhua | retain_erhua_whitelist

        self.fst = graph.optimize()
