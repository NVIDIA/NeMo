import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst
from nemo_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil
class HalfwidthFst(GraphFst):
    '''
        ：  -> halfwidth { halfwidth: "：" }  used unless you want to process only once
        ：  ->  :   in common case
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="halfwidth", kind="classify", deterministic=deterministic)
        halfwidth = pynini.string_file(get_abs_path("data/char/fullwidth_to_halfwidth.tsv"))
        self.graph_halfwidth = halfwidth
        graph_halfwidth = (
            pynutil.insert("halfwidth: \"") 
            + halfwidth 
            + pynutil.insert("\"")
        )
        graph_halfwidth = self.add_tokens(graph_halfwidth)
        self.fst = graph_halfwidth.optimize()
        