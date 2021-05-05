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

from typing import List, Optional, Tuple

import torch

import k2
from nemo.collections.asr.parts.k2.utils import get_tot_objf_and_num_frames


class CTCLoss(torch.nn.Module):
    """
    Connectionist Temporal Classification (CTC) loss.
    Ported from https://github.com/k2-fsa/snowfall/blob/master/snowfall/objectives/ctc.py
    """
    def __init__(
            self,
            num_classes: int,
            blank: int,
            reduction: str = 'mean_batch',
            graph_type: str = 'topo',
            L_inv: Optional[k2.Fsa],
            phones: Optional[k2.SymbolTable],
            words: Optional[k2.SymbolTable],
            oov: str = '<UNK>'
    ):
        super().__init__()
        self.reduction = reduction
        if graph_type == 'topo':
            from nemo.collections.asr.parts.k2.graph_compilers import CtcTrainingTopologyCompiler as compiler
            self.graph_compiler = compiler(num_classes, blank)
        elif graph_type == 'graph':
            from nemo.collections.asr.parts.k2.graph_compilers import CtcTrainingGraphCompiler as compiler
            # self.graph_compiler = compiler(L_inv, phones, words, oov)
            raise NotImplementedError("Not adapted yet")
        else:
            raise ValueError(f"Invalid value of `graph_type`: {graph_type}.")

    def create_supervision(self, input_lengths):
        supervisions = [[i, 0, input_lengths[i]] for i in range(len(input_lengths))]
        supervisions = torch.IntTensor(supervisions)
        # the duration column has to be sorted in decreasing order
        supervisions = supervisions[supervisions[:, -1].sort()[1][::-1]]
        return supervisions

    def forward(
            self,
            log_probs: torch.Tensor,
            targets: torch.Tensor,
            input_lengths: torch.Tensor,
            target_lengths: torch.Tensor
    ) -> torch.Tensor:
        supervision_segments = self.create_supervision(input_lengths)
        num_graphs = self.graph_compiler.compile(targets).to(nnet_output.device)
        dense_fsa_vec = k2.DenseFsaVec(log_probs, supervision_segments)

        num_lats = k2.intersect_dense(num_graphs, dense_fsa_vec, 10.0)

        num_tot_scores = num_lats.get_tot_scores(
            log_semiring=True,
            use_double_scores=True
        )
        tot_scores = num_tot_scores
        tot_scores, _, _ = get_tot_objf_and_num_frames(
            tot_scores,
            supervision_segments[:, 2],
            self.reduction
        )
        return tot_scores
