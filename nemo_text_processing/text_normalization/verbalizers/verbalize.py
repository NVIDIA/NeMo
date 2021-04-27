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

from nemo_text_processing.text_normalization.graph_utils import GraphFst
from nemo_text_processing.text_normalization.verbalizers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.verbalizers.date import DateFst
from nemo_text_processing.text_normalization.verbalizers.decimal import DecimalFst
from nemo_text_processing.text_normalization.verbalizers.electronic import ElectronicFst
from nemo_text_processing.text_normalization.verbalizers.measure import MeasureFst
from nemo_text_processing.text_normalization.verbalizers.money import MoneyFst
from nemo_text_processing.text_normalization.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.verbalizers.telephone import TelephoneFst
from nemo_text_processing.text_normalization.verbalizers.time import TimeFst
from nemo_text_processing.text_normalization.verbalizers.whitelist import WhiteListFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File. 
    More details to deployment at NeMo/tools/text_processing_deployment.
    """

    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")
        decimal = DecimalFst()
        decimal_graph = decimal.fst
        cardinal = CardinalFst()
        cardinal_graph = cardinal.fst
        ordinal = OrdinalFst()
        ordinal_graph = ordinal.fst
        telephone_graph = TelephoneFst().fst
        electronic_graph = ElectronicFst().fst
        measure_graph = MeasureFst(decimal=decimal, cardinal=cardinal).fst
        time_graph = TimeFst().fst
        date_graph = DateFst(ordinal=ordinal).fst
        money_graph = MoneyFst(decimal=decimal).fst
        whitelist_graph = WhiteListFst().fst
        graph = (
            time_graph
            | date_graph
            | money_graph
            | measure_graph
            | ordinal_graph
            | decimal_graph
            | cardinal_graph
            | telephone_graph
            | electronic_graph
            | whitelist_graph
        )
        self.fst = graph
