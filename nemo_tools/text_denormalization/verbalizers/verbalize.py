# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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
from nemo_tools.text_denormalization.verbalizers.cardinal import CardinalFst
from nemo_tools.text_denormalization.verbalizers.date import DateFst
from nemo_tools.text_denormalization.verbalizers.decimal import DecimalFst
from nemo_tools.text_denormalization.verbalizers.measure import MeasureFst
from nemo_tools.text_denormalization.verbalizers.money import MoneyFst
from nemo_tools.text_denormalization.verbalizers.ordinal import OrdinalFst
from nemo_tools.text_denormalization.verbalizers.time import TimeFst
from nemo_tools.text_denormalization.verbalizers.whitelist import WhiteListFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars. This class will be compiled and exported to thrax FAR.
    """

    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")
        cardinal = CardinalFst().fst
        ordinal = OrdinalFst().fst
        decimal = DecimalFst().fst
        measure = MeasureFst().fst
        time = TimeFst().fst
        date = DateFst().fst
        money = MoneyFst().fst
        whitelist = WhiteListFst().fst
        graph = time | date | money | measure | ordinal | decimal | cardinal | whitelist
        self.fst = graph
