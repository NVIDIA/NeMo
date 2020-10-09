# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from nemo.core.classes import NeuralModule


@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms."""

    score: float
    y_sequence: Union[List[int], torch.Tensor]
    dec_state: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor]]] = None
    y: List[torch.tensor] = None
    lm_state: Union[Dict[str, Any], List[Any]] = None
    lm_scores: torch.Tensor = None


class AbstractRNNTDecoder(NeuralModule, ABC):
    def __init__(self, vocab_size):
        super().__init__()

        # self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.blank_idx = vocab_size  # last index of vocabulary

    @abstractmethod
    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        add_sos: bool = False,
        batch_size: Optional[int] = None,
    ) -> (torch.Tensor, List[torch.Tensor]):
        raise NotImplementedError()

    @abstractmethod
    def initialize_state(self, y: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def score_hypothesis(
        self, hypothesis: Hypothesis, cache: Dict[Tuple[int], Any]
    ) -> (torch.Tensor, List[torch.Tensor], torch.Tensor):
        raise NotImplementedError()

    def batch_score_hypothesis(
        self,
        hypotheses: List[Hypothesis],
        cache: Dict[Tuple[int], Any],
        batch_states: List[torch.Tensor]
    ) -> (torch.Tensor, List[torch.Tensor], torch.Tensor):
        """

        Args:
            hypotheses:
            cache:
            batch_states:

        Returns:

        """
        raise NotImplementedError()


class AbstractRNNTJoint(NeuralModule, ABC):
    @abstractmethod
    def joint(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def num_classes_with_blank(self):
        raise NotImplementedError()
