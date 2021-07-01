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
from nemo_text_processing.text_normalization.ru.verbalizers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.ru.verbalizers.decimal import DecimalFst
from nemo_text_processing.text_normalization.ru.verbalizers.measure import MeasureFst
from nemo_text_processing.text_normalization.ru.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.verbalizers.whitelist import WhiteListFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File. 
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)
        cardinal = CardinalFst()
        cardinal_graph = cardinal.fst
        ordinal_graph = OrdinalFst().fst
        decimal = DecimalFst()
        decimal_graph = decimal.fst
        measure = MeasureFst(decimal=decimal, cardinal=cardinal, fraction=None, deterministic=deterministic)
        measure_graph = measure.fst
        whitelist_graph = WhiteListFst().fst

        graph = measure_graph | cardinal_graph | decimal_graph | ordinal_graph | whitelist_graph
        self.fst = graph
