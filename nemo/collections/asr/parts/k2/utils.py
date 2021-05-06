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

import math
import re
from typing import List, Tuple

import k2
import torch


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


def get_phone_symbols(symbol_table: k2.SymbolTable,
                      pattern: str = r'^#\d+$') -> List[int]:
    '''Return a list of phone IDs containing no disambiguation symbols.
    Caution:
      0 is not a phone ID so it is excluded from the return value.
    Args:
      symbol_table:
        A symbol table in k2.
      pattern:
        Symbols containing this pattern are disambiguation symbols.
    Returns:
      Return a list of symbol IDs excluding those from disambiguation symbols.
    '''
    regex = re.compile(pattern)
    symbols = symbol_table.symbols
    ans = []
    for s in symbols:
        if not regex.match(s):
            ans.append(symbol_table[s])
    if 0 in ans:
        ans.remove(0)
    ans.sort()
    return ans

def get_tot_objf_and_num_frames(
        tot_scores: torch.Tensor,
        frames_per_seq: torch.Tensor,
        reduction: str
    ) -> Tuple[torch.Tensor, int, int]:
    """Figures out the total score(log-prob) over all successful supervision segments
    (i.e. those for which the total score wasn't -infinity), and the corresponding
    number of frames of neural net output
         Args:
            tot_scores: a Torch tensor of shape (num_segments,) containing total scores
                       from forward-backward
            frames_per_seq: a Torch tensor of shape (num_segments,) containing the number of
                           frames for each segment
        Returns:
             Returns a tuple of 3 scalar tensors:  (tot_score, ok_frames, all_frames)
        where ok_frames is the frames for successful (finite) segments, and
       all_frames is the frames for all segments (finite or not).
    """
    mask = torch.ne(tot_scores, -math.inf)
    # finite_indexes is a tensor containing successful segment indexes, e.g.
    # [ 0 1 3 4 5 ]
    finite_indexes = torch.nonzero(mask).squeeze(1)
    ok_frames = frames_per_seq[finite_indexes].sum()
    all_frames = frames_per_seq.sum()
    if reduction == 'mean':
        tot_scores = tot_scores[finite_indexes].mean()
    elif reduction == 'sum':
        tot_scores = tot_scores[finite_indexes].sum()
    else:
        tot_scores = tot_scores[finite_indexes]
    return tot_scores, ok_frames, all_frames
