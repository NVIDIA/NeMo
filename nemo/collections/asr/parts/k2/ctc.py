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

from typing import Optional, Tuple, Union

import torch
import k2

from nemo.collections.asr.parts.k2.utils import create_supervision
from nemo.collections.asr.parts.k2.utils import get_tot_objf_and_num_frames
from nemo.collections.asr.parts.k2.utils import load_graph
from nemo.collections.asr.parts.k2.utils import make_blank_first
from nemo.collections.asr.parts.k2.utils import GradExpNormalize


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
            aux_graph: Optional[Union[k2.Fsa, str]] = None,
            **kwargs
    ):
        super().__init__()
        self.blank = blank
        self.num_classes = num_classes
        self.reduction = reduction
        if graph_type == 'topo':
            from nemo.collections.asr.parts.k2.graph_compilers import CtcTrainingTopologyCompiler as compiler
            self.graph_compiler = compiler(self.num_classes)
        elif graph_type == 'graph':
            from nemo.collections.asr.parts.k2.graph_compilers import CtcTrainingNumGraphCompiler as compiler
            raise NotImplementedError("Not tested yet")
            if isinstance(aux_graph, str):
                aux_graph = load_graph(aux_graph)
            self.graph_compiler = compiler(self.num_classes, aux_graph=aux_graph)
        else:
            raise ValueError(f"Invalid value of `graph_type`: {graph_type}.")

    def forward(
            self,
            log_probs: torch.Tensor,
            targets: torch.Tensor,
            input_lengths: torch.Tensor,
            target_lengths: torch.Tensor
    ) -> torch.Tensor:
        if self.blank != 0:
            # rearrange log_probs to put blank at the first place
            # and shift targets to emulate blank = 0
            log_probs, targets = make_blank_first(self.blank, log_probs, targets)
        supervisions, order = create_supervision(input_lengths)
        targets = targets[order]
        target_lengths = target_lengths[order]
        # PyTorch is doing the log-softmax normalization as part of the CTC computation.
        # More: https://github.com/k2-fsa/k2/issues/575
        # It would be nice to have dense_fsa_vec.get_tot_scores() instead.
        log_probs = GradExpNormalize.apply(log_probs, input_lengths, "mean" if self.reduction != "sum" else "none")

        if log_probs.device != self.graph_compiler.device:
            self.graph_compiler.to(log_probs.device)
        num_graphs = self.graph_compiler.compile(targets, target_lengths)

        dense_fsa_vec = k2.DenseFsaVec(log_probs, supervisions)

        num_lats = k2.intersect_dense(num_graphs, dense_fsa_vec, torch.finfo(torch.float32).max)

        num_tot_scores = num_lats.get_tot_scores(
            log_semiring=True,
            use_double_scores=False
        )
        tot_scores = num_tot_scores
        tot_scores, _ = get_tot_objf_and_num_frames(
            tot_scores,
            supervisions[:, 2],
            self.reduction
        )
        return -tot_scores
