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
            reduction: str = 'mean',
            graph_type: str = 'topo',
            L_inv: Optional[k2.Fsa] = None,
            phones: Optional[k2.SymbolTable] = None,
            words: Optional[k2.SymbolTable] = None,
            oov: str = '<UNK>'
    ):
        super().__init__()
        self.blank = blank
        self.num_classes = num_classes
        self.reduction = reduction
        if graph_type == 'topo':
            from nemo.collections.asr.parts.k2.graph_compilers import CtcTrainingTopologyCompiler as compiler
            self.graph_compiler = compiler(self.num_classes)
        elif graph_type == 'graph':
            from nemo.collections.asr.parts.k2.graph_compilers import CtcTrainingGraphCompiler as compiler
            # self.graph_compiler = compiler(L_inv, phones, words, oov)
            raise NotImplementedError("Not adapted yet")
        else:
            raise ValueError(f"Invalid value of `graph_type`: {graph_type}.")

    def create_supervision(self, input_lengths):
        supervisions = torch.stack(
            (
                torch.tensor(range(input_lengths.shape[0])),
                torch.zeros(input_lengths.shape[0]),
                input_lengths.cpu(),
            ),
            1,
        ).to(torch.int32)
        # the duration column has to be sorted in decreasing order
        order = torch.argsort(supervisions[:, -1], descending=True)
        return supervisions, order

    def forward(
            self,
            log_probs: torch.Tensor,
            targets: torch.Tensor,
            input_lengths: torch.Tensor,
            target_lengths: torch.Tensor
    ) -> torch.Tensor:
        if self.blank != 0:
            # rearrange log_probs to put blank at the first place
            index = list(range(self.num_classes))
            del index[self.blank]
            index = torch.tensor([self.blank] + index).to(log_probs.device)
            log_probs = torch.index_select(log_probs, -1, index)
            # shift targets to emulate blank = 0
            targets += 1
        supervisions, order = self.create_supervision(input_lengths)
        supervisions = supervisions[order]
        targets = targets[order]
        target_lengths = target_lengths[order]
        num_graphs = self.graph_compiler.compile(targets, target_lengths).to(log_probs.device)

        dense_fsa_vec = k2.DenseFsaVec(log_probs, supervisions)

        num_lats = k2.intersect_dense(num_graphs, dense_fsa_vec, torch.finfo(torch.float32).max)

        num_tot_scores = num_lats.get_tot_scores(
            log_semiring=True,
            use_double_scores=False
        )
        tot_scores = num_tot_scores
        tot_scores, _, _ = get_tot_objf_and_num_frames(
            tot_scores,
            supervisions[:, 2],
            self.reduction
        )
        return -tot_scores
