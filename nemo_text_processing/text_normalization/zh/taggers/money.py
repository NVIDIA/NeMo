import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_SIGMA, GraphFst, insert_space,NEMO_DIGIT
from nemo_text_processing.text_normalization.zh.utils import get_abs_path
from nemo_text_processing.text_normalization.zh.taggers.number import Number
from pynini.lib import pynutil

class Money(GraphFst):
    '''
        ￥1.25 -> tokens { money { cur: "元" num: "一点五" } }
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        currency = pynini.string_file(get_abs_path("data/money/currency.tsv"))  
        graph = (
            pynutil.insert("cur: \"") + currency + pynutil.insert("\"") +
            insert_space +
            pynutil.insert("num: \"") + Number().graph_number + pynutil.insert("\"")
        )

        self.fst = self.add_tokens(graph).optimize()
        