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

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from nemo.collections.asr.parts import rnnt_utils
from nemo.collections.asr.parts.rnn import label_collate
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import *
from nemo.utils import logging


class _GreedyRNNTInfer(Typing):
    """A greedy transducer decoder.
        Args:
            blank_symbol: See `Decoder`.
            model: Model to use for prediction.
            max_symbols_per_step: The maximum number of symbols that can be added
                to a sequence in a single time step; if set to None then there is
                no limit.
            cutoff_prob: Skip to next step in search if current highest character
                probability is less than this.
        """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"predictions": NeuralType(elements_type=HypothesisType())}

    def __init__(
        self,
        decoder_model: rnnt_utils.AbstractRNNTDecoder,
        joint_model: rnnt_utils.AbstractRNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
    ):
        super().__init__()
        self.decoder = decoder_model
        self.joint = joint_model

        self._blank_index = blank_index
        self._SOS = blank_index  # Start of single index
        self.max_symbols = max_symbols_per_step

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _pred_step(
        self,
        label: Union[torch.Tensor, int],
        hidden: Optional[torch.Tensor],
        add_sos: bool = False,
        batch_size: Optional[int] = None,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            label (int/torch.Tensor): Label or "Start-of-Signal" token.
            hidden (Optional torch.Tensor): RNN State vector
            batch_size (Optional torch.Tensor): Batch size of output
        Returns:
            g: (B, U + 1, H)
            hid: (h, c) where h is the final sequence hidden state and c is
                the final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)
        """
        if isinstance(label, torch.Tensor):
            # label: [batch, 1]
            if label.dtype != torch.long:
                label = label.long()

        else:
            # Label is an integer
            if label == self._SOS:
                return self.decoder.predict(None, hidden, add_sos=add_sos, batch_size=batch_size)

            label = label_collate([[label]])

        # output: [B, 1, K]
        return self.decoder.predict(label, hidden, add_sos=add_sos, batch_size=batch_size)

    def _joint_step(self, enc, pred, log_normalize: Optional[bool] = None):
        """
        Args:
            enc:
            pred:
            log_normalize:
        Returns:
             logits of shape (B, T=1, U=1, K + 1)
        """
        logits = self.joint.joint(enc, pred)

        if log_normalize is None:
            if not logits.is_cuda:  # Use log softmax only if on CPU
                logits = logits.log_softmax(dim=len(logits.shape) - 1)
        else:
            if log_normalize:
                logits = logits.log_softmax(dim=len(logits.shape) - 1)

        return logits


class GreedyRNNTInfer(_GreedyRNNTInfer):
    """A greedy transducer decoder.
    Args:
        blank_symbol: See `Decoder`.
        model: Model to use for prediction.
        max_symbols_per_step: The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        cutoff_prob: Skip to next step in search if current highest character
            probability is less than this.
    """

    def __init__(
        self,
        decoder_model: rnnt_utils.AbstractRNNTDecoder,
        joint_model: rnnt_utils.AbstractRNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
    ):
        super().__init__(
            decoder_model=decoder_model,
            joint_model=joint_model,
            blank_index=blank_index,
            max_symbols_per_step=max_symbols_per_step,
        )

    @typecheck()
    def forward(self, encoder_output: torch.Tensor, encoded_lengths: torch.Tensor):
        """Returns a list of sentences given an input batch.
        Args:
            x: A tensor of size (batch, features, timesteps).
            out_lens: list of int representing the length of each sequence
                output sequence.
        Returns:
            list containing batch number of sentences (strings).
        """
        # Preserve decoder and joint training state
        decoder_training_state = self.decoder.training
        joint_training_state = self.joint.training

        with torch.no_grad():
            # Apply optional preprocessing
            encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)

            self.decoder.eval()
            self.joint.eval()

            hypotheses = []
            for batch_idx in range(encoder_output.size(0)):
                inseq = encoder_output[batch_idx, :, :].unsqueeze(1)  # [T, 1, D]
                logitlen = encoded_lengths[batch_idx]
                sentence = self._greedy_decode(inseq, logitlen)
                hypotheses.append(sentence)

            packed_result = [
                rnnt_utils.Hypothesis(y_sequence=torch.tensor(sent, dtype=torch.long), score=-1.0)
                for sent in hypotheses
            ]

        self.decoder.train(decoder_training_state)
        self.joint.train(joint_training_state)

        return (packed_result,)

    @torch.no_grad()
    def _greedy_decode(self, x: torch.Tensor, out_len: torch.Tensor):
        hidden = None
        label = []

        for time_idx in range(out_len):
            f = x[time_idx, :, :].unsqueeze(0)  # [1, 1, D]

            not_blank = True
            symbols_added = 0

            while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                last_label = self._SOS if label == [] else label[-1]
                g, hidden_prime = self._pred_step(last_label, hidden)
                logp = self._joint_step(f, g, log_normalize=None)[0, 0, 0, :]

                if logp.dtype != torch.float32:
                    logp = logp.float()

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()

                if k == self._blank_index:
                    not_blank = False
                else:
                    label.append(k)
                    hidden = hidden_prime
                symbols_added += 1

        return label


class GreedyBatchedRNNTInfer(_GreedyRNNTInfer):
    """A greedy transducer decoder.
    Args:
        blank_symbol: See `Decoder`.
        model: Model to use for prediction.
        max_symbols_per_step: The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        cutoff_prob: Skip to next step in search if current highest character
            probability is less than this.
    """

    def __init__(
        self,
        decoder_model: rnnt_utils.AbstractRNNTDecoder,
        joint_model: rnnt_utils.AbstractRNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
    ):
        super().__init__(
            decoder_model=decoder_model,
            joint_model=joint_model,
            blank_index=blank_index,
            max_symbols_per_step=max_symbols_per_step,
        )

        if self.decoder.blank_as_pad:
            self._greedy_decode = self._greedy_decode_blank_as_pad
        else:
            self._greedy_decode = self._greedy_decode_masked

    @typecheck()
    def forward(self, encoder_output: torch.Tensor, encoded_lengths: torch.Tensor):
        """Returns a list of sentences given an input batch.
        Args:
            x: A tensor of size (batch, features, timesteps).
            out_lens: list of int representing the length of each sequence
                output sequence.
        Returns:
            list containing batch number of sentences (strings).
        """
        # Preserve decoder and joint training state
        decoder_training_state = self.decoder.training
        joint_training_state = self.joint.training

        with torch.no_grad():
            # Apply optional preprocessing
            encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)
            logitlen = encoded_lengths  # .to('cpu').numpy()

            self.decoder.eval()
            self.joint.eval()

            inseq = encoder_output  # [B, T, D]
            hypotheses = self._greedy_decode(inseq, logitlen, device=inseq.device)

            packed_result = [
                rnnt_utils.Hypothesis(y_sequence=torch.tensor(sent, dtype=torch.long), score=-1.0)
                for sent in hypotheses
            ]

        self.decoder.train(decoder_training_state)
        self.joint.train(joint_training_state)

        return (packed_result,)

    @torch.no_grad()
    def _greedy_decode_blank_as_pad(self, x: torch.Tensor, out_len: torch.Tensor, device: torch.device):
        # Initialize state
        hidden = None
        batchsize = x.shape[0]

        # Output string buffer
        label = [[] for _ in range(batchsize)]

        # Last Label buffer + Last Label without blank buffer
        # batch level equivalent of the last_label
        last_label = torch.full([batchsize, 1], fill_value=self._blank_index, dtype=torch.long, device=device)

        # Mask buffers
        blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)
        reset = torch.tensor(0, dtype=torch.bool, device=device)

        # Get max sequence length
        max_out_len = out_len.max()

        for time_idx in range(max_out_len):
            f = x[:, time_idx: time_idx + 1, :]  # [B, 1, D]

            # Prepare t timestamp batch variables
            not_blank = True
            symbols_added = 0

            # Reset blank mask
            blank_mask *= reset

            # Update blank mask with time mask
            # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
            # Forcibly mask with "blank" tokens, for all sample where current time step T > seq_len
            time_mask = time_idx >= out_len
            blank_mask.bitwise_or_(time_mask)

            # Start inner loop
            while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                # Batch prediction and joint network steps
                # If very first prediction step, submit SOS tag (blank) to pred_step.
                # This feeds a zero tensor as input to AbstractRNNTDecoder to prime the state
                if time_idx == 0 and symbols_added == 0:
                    g, hidden_prime = self._pred_step(self._SOS, hidden, batch_size=batchsize)
                else:
                    # Perform batch step prediction of decoder, getting new states and scores ("g")
                    g, hidden_prime = self._pred_step(last_label, hidden, batch_size=batchsize)

                # Batched joint step - Output = [B, V + 1]
                logp = self._joint_step(f, g, log_normalize=None)[:, 0, 0, :]

                if logp.dtype != torch.float32:
                    logp = logp.float()

                # Get index k, of max prob for batch
                v, k = logp.max(1)

                # Update blank mask with current predicted blanks
                # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
                k_is_blank = k == self._blank_index
                blank_mask.bitwise_or_(k_is_blank)

                # If all samples predict / have predicted prior blanks, exit loop early
                # This is equivalent to if single sample predicted k
                if blank_mask.all():
                    not_blank = False
                else:
                    # Collect batch indices where blanks occurred now/past
                    blank_indices = []
                    if hidden is not None:
                        blank_indices = (blank_mask == 1).nonzero(as_tuple=False)

                    # Recover prior state for all samples which predicted blank now/past
                    if hidden is not None:
                        # LSTM has 2 states
                        for state_id in range(len(hidden)):
                            hidden_prime[state_id][:, blank_indices, :] = hidden[state_id][:, blank_indices, :]

                    # Recover prior predicted label for all samples which predicted blank now/past
                    k[blank_indices] = last_label[blank_indices, 0]

                    # Update new label and hidden state for next iteration
                    last_label = k.view(-1, 1)
                    hidden = hidden_prime

                    # Update predicted labels, accounting for time mask
                    # If blank was predicted even once, now or in the past,
                    # Force the current predicted label to also be blank
                    # This ensures that blanks propogate across all timesteps
                    # once they have occured (normally stopping condition of sample level loop).
                    k = k.masked_fill(blank_mask, self._blank_index)
                    for kidx, ki in enumerate(k):
                        if time_mask[kidx] == 0 and ki != self._blank_index:
                            label[kidx].append(ki)

                symbols_added += 1

        return label

    @torch.no_grad()
    def _greedy_decode_masked(self, x: torch.Tensor, out_len: torch.Tensor, device: torch.device):
        # Initialize state
        hidden = None
        batchsize = x.shape[0]

        # Output string buffer
        label = [[] for _ in range(batchsize)]

        # Last Label buffer + Last Label without blank buffer
        # batch level equivalent of the last_label
        last_label = torch.full([batchsize, 1], fill_value=self._blank_index, dtype=torch.long, device=device)
        last_label_without_blank = last_label.clone()

        # Mask buffers
        blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)
        reset = torch.tensor(0, dtype=torch.bool, device=device)

        # Get max sequence length
        max_out_len = out_len.max()
        for time_idx in range(max_out_len):
            f = x[:, time_idx : time_idx + 1, :]  # [B, 1, D]

            # Prepare t timestamp batch variables
            not_blank = True
            symbols_added = 0

            # Reset blank mask
            blank_mask *= reset

            # Update blank mask with time mask
            # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
            # Forcibly mask with "blank" tokens, for all sample where current time step T > seq_len
            time_mask = time_idx >= out_len
            blank_mask.bitwise_or_(time_mask)

            # Start inner loop
            while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                # Batch prediction and joint network steps
                # If very first prediction step, submit SOS tag (blank) to pred_step.
                # This feeds a zero tensor as input to AbstractRNNTDecoder to prime the state
                if time_idx == 0 and symbols_added == 0:
                    g, hidden_prime = self._pred_step(self._SOS, hidden, batch_size=batchsize)
                else:
                    # Set a dummy label for the blank value
                    # This value will be overwritten by "blank" again the last label update below
                    # This is done as vocabulary of prediction network does not contain "blank" token of RNNT
                    last_label_without_blank_mask = last_label == self._blank_index
                    last_label_without_blank[last_label_without_blank_mask] = 0  # temp change of label
                    last_label_without_blank[~last_label_without_blank_mask] = last_label[
                        ~last_label_without_blank_mask
                    ]

                    # Perform batch step prediction of decoder, getting new states and scores ("g")
                    g, hidden_prime = self._pred_step(last_label_without_blank, hidden, batch_size=batchsize)

                # Batched joint step - Output = [B, V + 1]
                logp = self._joint_step(f, g, log_normalize=None)[:, 0, 0, :]

                if logp.dtype != torch.float32:
                    logp = logp.float()

                # Get index k, of max prob for batch
                v, k = logp.max(1)

                # Update blank mask with current predicted blanks
                # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
                k_is_blank = k == self._blank_index
                blank_mask.bitwise_or_(k_is_blank)

                # If all samples predict / have predicted prior blanks, exit loop early
                # This is equivalent to if single sample predicted k
                if blank_mask.all():
                    not_blank = False
                else:
                    # Collect batch indices where blanks occurred now/past
                    blank_indices = []
                    if hidden is not None:
                        blank_indices = (blank_mask == 1).nonzero(as_tuple=False)

                    # Recover prior state for all samples which predicted blank now/past
                    if hidden is not None:
                        # LSTM has 2 states
                        for state_id in range(len(hidden)):
                            hidden_prime[state_id][:, blank_indices, :] = hidden[state_id][:, blank_indices, :]

                    # Recover prior predicted label for all samples which predicted blank now/past
                    k[blank_indices] = last_label[blank_indices, 0]

                    # Update new label and hidden state for next iteration
                    last_label = k.view(-1, 1)
                    hidden = hidden_prime

                    # Update predicted labels, accounting for time mask
                    # If blank was predicted even once, now or in the past,
                    # Force the current predicted label to also be blank
                    # This ensures that blanks propogate across all timesteps
                    # once they have occured (normally stopping condition of sample level loop).
                    k = k.masked_fill(blank_mask, self._blank_index)
                    for kidx, ki in enumerate(k):
                        if time_mask[kidx] == 0 and ki != self._blank_index:
                            label[kidx].append(ki)

                symbols_added += 1

        return label
