# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemo_tools.text_denormalization.graph_utils import GraphFst
from nemo_tools.text_denormalization.taggers.cardinal import CardinalFst
from nemo_tools.text_denormalization.taggers.date import DateFst
from nemo_tools.text_denormalization.taggers.decimal import DecimalFst
from nemo_tools.text_denormalization.taggers.measure import MeasureFst
from nemo_tools.text_denormalization.taggers.money import MoneyFst
from nemo_tools.text_denormalization.taggers.ordinal import OrdinalFst
from nemo_tools.text_denormalization.taggers.time import TimeFst
from nemo_tools.text_denormalization.taggers.whitelist import WhiteListFst
from nemo_tools.text_denormalization.taggers.word import WordFst
from pynini.lib import pynutil


class ClassifyFst(GraphFst):
    """
    Composes other classfier grammars. This class will be compiled and exported to thrax FAR. 
    """

    def __init__(self):
        super().__init__(name="tokenize_and_classify", kind="classify")

        cardinal = CardinalFst()
        cardinal_no_exception = cardinal.graph_no_exception
        cardinal_graph_hundred_component_at_least_one_none_zero_digit = cardinal.graph_hundred_component_at_least_one_none_zero_digit

        cardinal = cardinal.fst

        ordinal = OrdinalFst(cardinal_no_exception)
        # ordinal_graph = cardinal_no_exception# ordinal.graph
        ordinal_graph = ordinal.graph
        ordinal = ordinal.fst

        decimal = DecimalFst(cardinal_no_exception, cardinal_graph_hundred_component_at_least_one_none_zero_digit)
        decimal_graph = decimal.final_graph_wo_negative
        decimal = decimal.fst

        measure = MeasureFst(cardinal_no_exception, decimal_graph).fst
        date = DateFst(ordinal_graph).fst
        word = WordFst().fst
        time = TimeFst().fst
        money = MoneyFst(cardinal_no_exception, decimal_graph).fst
        whitelist = WhiteListFst().fst

        all_graphs = {'cardinal':cardinal, 'ordinal': ordinal, 'time': time, 'date': date,
                      'decimal':decimal, 'whitelist': whitelist, 'measure': measure, 'money': money,
                      'word': word}
        total_st = 0
        for gr_name, gr_cat in all_graphs.items():
            num_st_cat = gr_cat.optimize().num_states()
            print(f'{gr_name}: {num_st_cat}')
            total_st += num_st_cat

        graph = (
            pynutil.add_weight(whitelist, 1.01)
            # | pynutil.add_weight(time, 1.1)
            # | pynutil.add_weight(date, 1.09)
            # | pynutil.add_weight(decimal, 1.1)
            | pynutil.add_weight(measure, 1.1)
            | pynutil.add_weight(cardinal, 1.1)
            | pynutil.add_weight(ordinal, 1.1)
            | pynutil.add_weight(money, 1.1)
            | pynutil.add_weight(word, 100)
        )
        print(f'SUM: {total_st}')
        print(f'before optim: {graph.num_states()}')
        self.fst = graph.optimize()
        print(f' after optim: {self.fst.num_states()}')
