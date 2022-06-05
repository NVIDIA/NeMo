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

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import ElementType, HypothesisType, LengthsType, LogprobsType, NeuralType
from nemo.utils import logging


def pack_hypotheses(hypotheses: List[rnnt_utils.Hypothesis], logitlen: torch.Tensor,) -> List[rnnt_utils.Hypothesis]:

    if logitlen is not None:
        if hasattr(logitlen, 'cpu'):
            logitlen_cpu = logitlen.to('cpu')
        else:
            logitlen_cpu = logitlen

    for idx, hyp in enumerate(hypotheses):  # type: rnnt_utils.Hypothesis
        hyp.y_sequence = torch.tensor(hyp.y_sequence, dtype=torch.long)

        if logitlen is not None:
            hyp.length = logitlen_cpu[idx]

        if hyp.dec_state is not None:
            hyp.dec_state = _states_to_device(hyp.dec_state)

    return hypotheses


def _states_to_device(dec_state, device='cpu'):
    if torch.is_tensor(dec_state):
        dec_state = dec_state.to(device)

    elif isinstance(dec_state, (list, tuple)):
        dec_state = tuple(_states_to_device(dec_i, device) for dec_i in dec_state)

    return dec_state


class GreedyCTCInfer(Typing):
    """A greedy transducer decoder.

    Provides a common abstraction for sample level and batch level greedy decoding.

    Args:
        decoder_model: rnnt_utils.AbstractRNNTDecoder implementation.
        joint_model: rnnt_utils.AbstractRNNTJoint implementation.
        blank_index: int index of the blank token. Can be 0 or len(vocabulary).
        max_symbols_per_step: Optional int. The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        preserve_alignments: Bool flag which preserves the history of alignments generated during
            greedy decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `alignments` in it. Here, `alignments` is a List of List of ints.

            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
            U is the number of target tokens for the current timestep Ti.
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "decoder_output": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "decoder_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"predictions": [NeuralType(elements_type=HypothesisType())]}

    def __init__(
        self, blank_id: int, preserve_alignments: bool = False, compute_timestamps: bool = False,
    ):
        super().__init__()

        self.blank_id = blank_id
        self.preserve_alignments = preserve_alignments
        self.compute_timestamps = compute_timestamps

    @typecheck()
    def forward(
        self, decoder_output: torch.Tensor, decoder_lengths: torch.Tensor,
    ):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            decoder_output: A tensor of size (batch, timesteps, features).
            decoder_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        with torch.inference_mode():
            hypotheses = []
            # Process each sequence independently
            prediction_cpu_tensor = decoder_output.long().cpu()
            for ind in range(prediction_cpu_tensor.shape[0]):
                out_len = decoder_lengths[ind] if decoder_lengths is not None else None
                hypothesis = self._greedy_decode(prediction_cpu_tensor[ind], out_len)
                hypotheses.append(hypothesis)

            # Pack results into Hypotheses
            packed_result = pack_hypotheses(hypotheses, decoder_lengths)

        return (packed_result,)

    @torch.no_grad()
    def _greedy_decode(self, x: torch.Tensor, out_len: torch.Tensor):
        # x: [T, 1, D]
        # out_len: [seq_len]

        # Initialize blank state and empty label set in Hypothesis
        hypothesis = rnnt_utils.Hypothesis(score=0.0, y_sequence=[], dec_state=None, timestep=[], last_token=None)
        prediction = x.detach().cpu()

        if out_len is not None:
            prediction = prediction[:out_len]

        prediction_logprobs, prediction_labels = prediction.max(-1)

        non_blank_ids = prediction_labels != self.blank_id
        hypothesis.y_sequence = prediction_labels.numpy().tolist()
        hypothesis.score = sum(prediction_logprobs[non_blank_ids])

        if self.preserve_alignments:
            hypothesis.alignments = prediction.clone()

        if self.compute_timestamps:
            hypothesis.timestep = torch.nonzero(non_blank_ids, as_tuple=False)[:, 0].numpy().tolist()

        return hypothesis

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


@dataclass
class GreedyCTCInferConfig:
    preserve_alignments: bool = False
    compute_timestamps: bool = False
