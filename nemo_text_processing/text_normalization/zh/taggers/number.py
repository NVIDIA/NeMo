import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst
from nemo_text_processing.text_normalization.zh.utils import (
    get_abs_path,
    UNIT_1e01,
    UNIT_1e02,
    UNIT_1e03,
    UNIT_1e04,
    UNIT_1e08,
    UNIT_1e12, 
)
from pynini.lib import pynutil

class Number(GraphFst):
    '''
        self.graph_number:
        5       -> number { number: "五"}
        12      -> number { number: "十二"}
        213     -> number { number: "二百一十三"}
        3123    -> number { number: "三千一百二十三"}
        3,123   -> number { number: "三千一百二十三"}
        51234   -> number { number: "五万一千二百三十四"}
        51,234  -> number { number: "五万一千二百三十四"}
        0.125   -> number { number: "零点一二五"}
        self.fst:
        123     -> number { number: "一二三" }
    '''
    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="number", kind="classify", deterministic=deterministic)
        #base number from digit to chinese
        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/number/digit_teen.tsv"))

        graph_digit_with_zero = graph_digit|graph_zero# from 0(read out) to 9
        graph_no_zero = pynini.cross("0","") #number zero,in some case we don't read it out
        graph_digit_no_zero = graph_digit|graph_no_zero# from 0(not read out) to 9
        insert_zero = pynutil.insert('零')
        delete_punct = pynutil.delete(",") # to deal with 5,000,number in english form

        # 15 in 215 
        graph_ten_u = (
            (graph_digit + pynutil.insert(UNIT_1e01) + graph_digit_no_zero)|
            (graph_zero + graph_digit)
        )
        # 15 only 
        graph_ten = (
            (graph_teen + pynutil.insert(UNIT_1e01) + graph_digit_no_zero)|
            (graph_zero + graph_digit)
        )
        # 215 or 200
        graph_hundred = (
            (graph_digit + pynutil.insert(UNIT_1e02) + graph_ten_u)|
            (graph_digit + pynutil.insert(UNIT_1e02) + graph_no_zero**2)
        )
        # 3000 or 3002 or 3012 or 3123
        graph_thousand = (
            (graph_digit + pynutil.insert(UNIT_1e03) + graph_hundred)|
            (graph_digit + pynutil.insert(UNIT_1e03) + graph_zero + graph_digit + pynutil.insert(UNIT_1e01) + graph_digit_no_zero)|
            (graph_digit + pynutil.insert(UNIT_1e03) + graph_zero + graph_no_zero + graph_digit)|
            (graph_digit + pynutil.insert(UNIT_1e03) + graph_no_zero**3)
        )
        # just 3,000 or 3,002 or 3,012 or 3,123
        graph_thousand_sign =(
            (graph_digit + pynutil.insert(UNIT_1e03) + delete_punct + graph_hundred)|
            (graph_digit + pynutil.insert(UNIT_1e03) + delete_punct + graph_zero + graph_digit + pynutil.insert(UNIT_1e01) + graph_digit_no_zero)|
            (graph_digit + pynutil.insert(UNIT_1e03) +delete_punct + graph_zero + graph_no_zero + graph_digit)| 
            (graph_digit + pynutil.insert(UNIT_1e03) + delete_punct + graph_no_zero**3)
        )
        # 20001234 or 2001234 or 201234 or 21234 or 20234 or 20023 or 20002 or 20000
        # 8 digits max supported,for 9 digits number,often write with unit instead of read it in only number form 
        graph_ten_thousand = (
            (graph_thousand|graph_hundred|graph_ten|graph_digit_no_zero) + pynutil.insert(UNIT_1e04) + \
            (
                graph_thousand|
                (graph_no_zero + insert_zero + graph_hundred)|
                (graph_no_zero**2 + insert_zero + (graph_digit + pynutil.insert(UNIT_1e01) + graph_digit_no_zero))|
                (graph_no_zero**3 + insert_zero + graph_digit)|
                (graph_no_zero**4)
            )
        )
        #just like thousand,to deal with case like 23,111
        graph_ten_thousand_sign = (
            (graph_thousand_sign|graph_hundred|graph_ten|graph_digit_no_zero) + pynutil.insert(UNIT_1e04) + \
            (
                graph_thousand_sign|
                (graph_no_zero + delete_punct + insert_zero + graph_hundred)|
                (graph_no_zero + delete_punct + graph_no_zero + insert_zero + (graph_digit + pynutil.insert(UNIT_1e01) + graph_digit_no_zero))|
                (graph_no_zero + delete_punct + graph_no_zero**2 + insert_zero + graph_digit)|
                (graph_no_zero**4)
            )
        )

        #number string like 123 or 123.456,used in phone number,ID number,etc.
        graph_numstring = (
            pynini.closure(graph_digit_with_zero,1)|
            (pynini.closure(graph_digit_with_zero,1) + pynini.cross(".","点") + pynini.closure(graph_digit_with_zero,1))
        )
        
        graph = (
            graph_hundred           | 
            graph_thousand          | 
            graph_ten               | 
            graph_digit_with_zero   | 
            graph_ten_thousand      |
            graph_thousand_sign     |
            graph_ten_thousand_sign
        )
        # 456.123
        graph_decimal = (
            graph + pynini.cross('.','点') + pynini.closure(graph_digit_with_zero,1) 
        )
        graph_number = graph | graph_decimal
        self.graph_number = graph_number.optimize()

        graph_numstring = self.add_tokens(
            pynutil.insert("number: \"") + graph_numstring + pynutil.insert("\"")
        )
        self.fst = graph_numstring.optimize()
