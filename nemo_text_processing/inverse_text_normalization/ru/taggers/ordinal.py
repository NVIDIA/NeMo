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


from nemo_text_processing.text_normalization.graph_utils import NEMO_CHAR, NEMO_SIGMA, GraphFst

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
        -23 -> cardinal { negative: "true"  integer: "twenty three" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        from nemo_text_processing.text_normalization.ru.taggers.ordinal import OrdinalFst

        ordinal_tn = OrdinalFst(deterministic=False)
        ordinal_tn = ordinal_tn.ordinal_numbers

        graph = ordinal_tn.invert().optimize()
        self.graph = graph
        graph = pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()


if __name__ == '__main__':
    from pynini.lib import rewrite

    fst = OrdinalFst()
    print(rewrite.rewrites("двадцатый", fst.graph))

    import pdb

    pdb.set_trace()
    print(rewrite.rewrites("20", fst.ordinal_tn))
    reformat = pynini.cdrewrite(pynini.cross('ый', '-ый'), "", "[EOS]", NEMO_SIGMA)
    print(rewrite.rewrites("20", fst.ordinal_tn @ reformat))
