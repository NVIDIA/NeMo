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

# Copyright (c) 2020, Xiaomi CORPORATION.  All rights reserved.
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

from functools import lru_cache
from typing import Iterable

import torch

import k2

from nemo.collections.asr.parts.k2.utils import build_ctc_topo
from nemo.collections.asr.parts.k2.utils import get_phone_symbols


class CtcTrainingTopologyCompiler(object):

    def __init__(self, num_classes, blank):
        assert blank < num_classes
        self.blank = blank
        phone_ids_with_blank = list(range(num_classes))
        self.ctc_topo = k2.arc_sort(build_ctc_topo(phone_ids_with_blank, self.blank))

    def compile(self, targets: torch.Tensor, target_lengths: torch.Tensor) -> k2.Fsa:
        decoding_graphs = k2.create_fsa_vec(
            [self.compile_one_and_cache(t[:l]) for t, l in zip(targets, target_lengths)])

        # make sure the gradient is not accumulated
        decoding_graphs.requires_grad_(False)
        return decoding_graphs

    @lru_cache(maxsize=1000000)
    def compile_one_and_cache(self, target: torch.Tensor) -> k2.Fsa:
        # shift targets to emulate blank = 0
        target = (target + int(self.blank != 0)).tolist()
        label_graph = k2.arc_sort(k2.linear_fsa(target))
        decoding_graph = k2.connect(k2.compose(self.ctc_topo, label_graph))
        return decoding_graph


class CtcTrainingGraphCompiler(object):

    def __init__(self,
                 L_inv: k2.Fsa,
                 phones: k2.SymbolTable,
                 words: k2.SymbolTable,
                 oov: str = '<UNK>'):
        '''
        Args:
          L_inv:
            Its labels are words, while its aux_labels are phones.
        phones:
          The phone symbol table.
        words:
          The word symbol table.
        oov:
          Out of vocabulary word.
        '''
        if L_inv.properties & k2.fsa_properties.ARC_SORTED != 0:
            L_inv = k2.arc_sort(L_inv)

        assert oov in words

        self.L_inv = L_inv
        self.phones = phones
        self.words = words
        self.oov = oov
        phone_ids = get_phone_symbols(phones)
        phone_ids_with_blank = [0] + phone_ids
        self.ctc_topo = k2.arc_sort(build_ctc_topo(phone_ids_with_blank))

    def compile(self, texts: Iterable[str]) -> k2.Fsa:
        decoding_graphs = k2.create_fsa_vec(
            [self.compile_one_and_cache(text) for text in texts])

        # make sure the gradient is not accumulated
        decoding_graphs.requires_grad_(False)
        return decoding_graphs

    @lru_cache(maxsize=100000)
    def compile_one_and_cache(self, text: str) -> k2.Fsa:
        tokens = (token if token in self.words else self.oov
                  for token in text.split(' '))
        word_ids = [self.words[token] for token in tokens]
        label_graph = k2.linear_fsa(word_ids)
        decoding_graph = k2.connect(k2.intersect(label_graph,
                                                 self.L_inv)).invert_()
        decoding_graph = k2.arc_sort(decoding_graph)
        decoding_graph = k2.compose(self.ctc_topo, decoding_graph)
        decoding_graph = k2.connect(decoding_graph)
        return decoding_graph
