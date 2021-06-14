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

from typing import List, Tuple

import k2
import torch


def build_topo(name: str, tokens: List[int]) -> k2.Fsa:
    if name == "ctc_default":
        return build_ctc_topo(tokens)
    elif name == "ctc_no_selfloops":
        return build_ctc_topo_no_selfloops(tokens)
    elif name == "identity":
        return build_identity_topo(tokens)
    else:
        raise ValueError(f"Unknown topo name: {name}")

def build_ctc_topo(tokens: List[int]) -> k2.Fsa:
    """Build CTC topology.
    A token which appears once on the right side (i.e. olabels) may
    appear multiple times on the left side (ilabels), possibly with
    epsilons in between.
    When 0 appears on the left side, it represents the blank symbol;
    when it appears on the right side, it indicates an epsilon. That
    is, 0 has two meanings here.
    Args:
      tokens:
        A list of tokens, e.g., phones, characters, etc.
    Returns:
      Returns an FST that converts repeated tokens to a single token.
    """
    assert 0 in tokens, "We assume 0 is ID of the blank symbol"

    num_states = len(tokens)
    final_state = num_states
    arcs = ""
    for i in range(num_states):
        for j in range(num_states):
            if i == j:
                arcs += f"{i} {i} {tokens[i]} 0 0.0\n"
            else:
                arcs += f"{i} {j} {tokens[j]} {tokens[j]} 0.0\n"
        arcs += f"{i} {final_state} -1 -1 0.0\n"
    arcs += f"{final_state}"
    ans = k2.Fsa.from_str(arcs, num_aux_labels=1)
    ans = k2.arc_sort(ans)
    return ans

def build_ctc_topo_no_selfloops(tokens: List[int]) -> k2.Fsa:
    assert 0 in tokens, "We assume 0 is ID of the blank symbol"

    num_states = len(tokens)
    final_state = num_states
    arcs = "0 0 0 0 0.0\n"
    for i in range(num_states):
        for j in range(num_states):
            if i != j:
                arcs += f"{i} {j} {tokens[j]} {tokens[j]} 0.0\n"
        arcs += f"{i} {final_state} -1 -1 0.0\n"
    arcs += f"{final_state}"
    ans = k2.Fsa.from_str(arcs, num_aux_labels=1)
    ans = k2.arc_sort(ans)
    return ans

def build_identity_topo(tokens: List[int]) -> k2.Fsa:
    assert 0 in tokens, "We assume 0 is ID of the blank symbol"

    num_tokens = len(tokens)
    final_state = 1
    arcs = ""
    for i in range(num_tokens):
        arcs += f"0 0 {tokens[i]} {tokens[i]} 0.0\n"
    arcs += f"0 {final_state} -1 -1 0.0\n"
    arcs += f"{final_state}"
    ans = k2.Fsa.from_str(arcs, num_aux_labels=1)
    ans = k2.arc_sort(ans)
    return ans
