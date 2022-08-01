import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_SIGMA, GraphFst,NEMO_NOT_SPACE
from pynini.lib import pynutil
from nemo_text_processing.text_normalization.zh.utils import get_abs_path,load_labels

class Erhua(GraphFst):
    '''
        这儿 -> tokens { char : "这" } tokens { erhua: "儿" }
        儿女 -> tokens { erhua_whitelist: "儿女" }
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="erhua", kind="classify", deterministic=deterministic)

        erhua = pynutil.insert("erhua: \"") + '儿' + pynutil.insert("\"")

        erhua_whitelist = (
            pynutil.insert("erhua_whitelist: \"") +
            pynini.string_file(get_abs_path("data/erhua/whitelist.tsv")) +
            pynutil.insert("\"")
        )
        graph = pynutil.add_weight(erhua, 0.1) | erhua_whitelist

        self.fst = graph.optimize()
