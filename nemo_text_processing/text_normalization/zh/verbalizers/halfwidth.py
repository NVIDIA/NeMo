import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst,NEMO_NOT_SPACE
from pynini.lib import pynutil
class HalfwidthFst(GraphFst):
    '''
        halfwidth { halfwidth: "：" } ->  :
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="halfwidth", kind="verbalize", deterministic=deterministic)
        halfwidth = (
            pynutil.delete("halfwidth: \"") 
            + NEMO_NOT_SPACE 
            + pynutil.delete("\"")
        )
        halfwidth = self.delete_tokens(halfwidth)
        self.fst = halfwidth.optimize()