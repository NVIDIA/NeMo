import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst,insert_space,NEMO_DIGIT,NEMO_NOT_QUOTE
from nemo_text_processing.text_normalization.zh.taggers.number import NumberFst
from nemo_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil
class TimeFst(GraphFst):
    '''
        1:02          -> time { h: "1" m: "02" }
        1:02:36       -> time { h: "1" m: "02" s: "36" }
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="time", kind="classify", deterministic=deterministic)
        h = pynini.string_file(get_abs_path("data/time/hour.tsv"))
        time_tens = pynini.string_file(get_abs_path("data/time/tens.tsv"))
        time_digit = pynini.string_file(get_abs_path("data/time/digit.tsv"))
        m = time_tens + time_digit
        s = (time_tens + time_digit)|time_digit

        delete_colon = pynini.cross(':', ' ')
        # 5:05, 14:30
        h_m = \
            pynutil.insert('h: \"') + h + pynutil.insert('\"') + \
            delete_colon + \
            pynutil.insert('m: \"') + m + pynutil.insert('\"')

        # 1:30:15
        h_m_s = \
            pynutil.insert('h: \"') + h + pynutil.insert('\"') + \
            delete_colon + \
            pynutil.insert('m: \"') + m + pynutil.insert('\"') + \
            delete_colon + \
            pynutil.insert('s: \"') + s + pynutil.insert('\"')

        patterns = h_m | h_m_s
        self.fst = self.add_tokens(patterns).optimize()
