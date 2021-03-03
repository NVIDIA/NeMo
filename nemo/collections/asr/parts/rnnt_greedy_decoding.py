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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from nemo.collections.asr.modules import rnnt_abstract
from nemo.collections.asr.parts import rnnt_utils
from nemo.collections.common.parts.rnn import label_collate
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import AcousticEncodedRepresentation, HypothesisType, LengthsType, NeuralType


def pack_hypotheses(
    hypotheses: List[List[int]],
    timesteps: List[List[int]],
    logitlen: torch.Tensor,
    alignments: Optional[List[List[int]]] = None,
) -> List[rnnt_utils.Hypothesis]:
    logitlen_cpu = logitlen.to("cpu")
    return [
        rnnt_utils.Hypothesis(
            y_sequence=torch.tensor(sent, dtype=torch.long),
            score=-1.0,
            timestep=timestep,
            length=length,
            alignments=alignments[idx] if alignments is not None else None,
        )
        for idx, (sent, timestep, length) in enumerate(zip(hypotheses, timesteps, logitlen_cpu))
    ]


class _GreedyRNNTInfer(Typing):
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
            "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"predictions": [NeuralType(elements_type=HypothesisType())]}

    def __init__(
        self,
        decoder_model: rnnt_abstract.AbstractRNNTDecoder,
        joint_model: rnnt_abstract.AbstractRNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments: bool = False,
    ):
        super().__init__()
        self.decoder = decoder_model
        self.joint = joint_model

        self._blank_index = blank_index
        self._SOS = blank_index  # Start of single index
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def _pred_step(
        self,
        label: Union[torch.Tensor, int],
        hidden: Optional[torch.Tensor],
        add_sos: bool = False,
        batch_size: Optional[int] = None,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Common prediction step based on the AbstractRNNTDecoder implementation.

        Args:
            label: (int/torch.Tensor): Label or "Start-of-Signal" token.
            hidden: (Optional torch.Tensor): RNN State vector
            add_sos (bool): Whether to add a zero vector at the begging as "start of sentence" token.
            batch_size: Batch size of the output tensor.

        Returns:
            g: (B, U, H) if add_sos is false, else (B, U + 1, H)
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
        Common joint step based on AbstractRNNTJoint implementation.

        Args:
            enc: Output of the Encoder model. A torch.Tensor of shape [B, 1, H1]
            pred: Output of the Decoder model. A torch.Tensor of shape [B, 1, H2]
            log_normalize: Whether to log normalize or not. None will log normalize only for CPU.

        Returns:
             logits of shape (B, T=1, U=1, V + 1)
        """
        with torch.no_grad():
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

    Sequence level greedy decoding, performed auto-repressively.

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

    def __init__(
        self,
        decoder_model: rnnt_abstract.AbstractRNNTDecoder,
        joint_model: rnnt_abstract.AbstractRNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments: bool = False,
    ):
        super().__init__(
            decoder_model=decoder_model,
            joint_model=joint_model,
            blank_index=blank_index,
            max_symbols_per_step=max_symbols_per_step,
            preserve_alignments=preserve_alignments,
        )

    @typecheck()
    def forward(self, encoder_output: torch.Tensor, encoded_lengths: torch.Tensor):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            encoder_output: A tensor of size (batch, features, timesteps).
            encoded_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
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
            timesteps = []
            alignments = [] if self.preserve_alignments else None
            # Process each sequence independently
            with self.decoder.as_frozen(), self.joint.as_frozen():
                for batch_idx in range(encoder_output.size(0)):
                    inseq = encoder_output[batch_idx, :, :].unsqueeze(1)  # [T, 1, D]
                    logitlen = encoded_lengths[batch_idx]
                    sentence, timestep, alignment = self._greedy_decode(inseq, logitlen)
                    hypotheses.append(sentence)
                    timesteps.append(timestep)

                    if self.preserve_alignments:
                        alignments.append(alignment)

            # Pack results into Hypotheses
            packed_result = pack_hypotheses(hypotheses, timesteps, encoded_lengths, alignments=alignments)

        self.decoder.train(decoder_training_state)
        self.joint.train(joint_training_state)

        return (packed_result,)

    @torch.no_grad()
    def _greedy_decode(self, x: torch.Tensor, out_len: torch.Tensor):
        # x: [T, 1, D]
        # out_len: [seq_len]

        # Initialize blank state and empty label set
        hidden = None
        label = []
        timesteps = []

        if self.preserve_alignments:
            # Alignments is a 2-dimensional dangling list representing T x U
            alignments = [[]]
        else:
            alignments = None

        # For timestep t in X_t
        for time_idx in range(out_len):
            # Extract encoder embedding at timestep t
            # f = x[time_idx, :, :].unsqueeze(0)  # [1, 1, D]
            f = x.narrow(dim=0, start=time_idx, length=1)

            # Setup exit flags and counter
            not_blank = True
            symbols_added = 0

            # While blank is not predicted, or we dont run out of max symbols per timestep
            while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                # In the first timestep, we initialize the network with RNNT Blank
                # In later timesteps, we provide previous predicted label as input.
                last_label = self._SOS if label == [] else label[-1]

                # Perform prediction network and joint network steps.
                g, hidden_prime = self._pred_step(last_label, hidden)
                logp = self._joint_step(f, g, log_normalize=None)[0, 0, 0, :]

                del g

                # torch.max(0) op doesnt exist for FP 16.
                if logp.dtype != torch.float32:
                    logp = logp.float()

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()  # K is the label at timestep t_s in inner loop, s >= 0.

                if self.preserve_alignments:
                    # insert logits into last timestep
                    alignments[-1].append(k)

                del logp

                # If blank token is predicted, exit inner loop, move onto next timestep t
                if k == self._blank_index:
                    not_blank = False

                    if self.preserve_alignments:
                        # convert Ti-th logits into a torch array
                        alignments.append([])  # blank buffer for next timestep
                else:
                    # Append token to label set, update RNN state.
                    label.append(k)
                    timesteps.append(time_idx)
                    hidden = hidden_prime

                # Increment token counter.
                symbols_added += 1

        # Remove trailing empty list of Alignments
        if self.preserve_alignments:
            if len(alignments[-1]) == 0:
                del alignments[-1]

        return label, timesteps, alignments


class GreedyBatchedRNNTInfer(_GreedyRNNTInfer):
    """A batch level greedy transducer decoder.

    Batch level greedy decoding, performed auto-repressively.

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

    def __init__(
        self,
        decoder_model: rnnt_abstract.AbstractRNNTDecoder,
        joint_model: rnnt_abstract.AbstractRNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments: bool = False,
    ):
        super().__init__(
            decoder_model=decoder_model,
            joint_model=joint_model,
            blank_index=blank_index,
            max_symbols_per_step=max_symbols_per_step,
            preserve_alignments=preserve_alignments,
        )

        # Depending on availability of `blank_as_pad` support
        # switch between more efficient batch decoding technique
        if self.decoder.blank_as_pad:
            self._greedy_decode = self._greedy_decode_blank_as_pad
        else:
            self._greedy_decode = self._greedy_decode_masked

    @typecheck()
    def forward(self, encoder_output: torch.Tensor, encoded_lengths: torch.Tensor):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            encoder_output: A tensor of size (batch, features, timesteps).
            encoded_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        # Preserve decoder and joint training state
        decoder_training_state = self.decoder.training
        joint_training_state = self.joint.training

        with torch.no_grad():
            # Apply optional preprocessing
            encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)
            logitlen = encoded_lengths

            self.decoder.eval()
            self.joint.eval()

            with self.decoder.as_frozen(), self.joint.as_frozen():
                inseq = encoder_output  # [B, T, D]
                hypotheses, timesteps, alignments = self._greedy_decode(inseq, logitlen, device=inseq.device)

            # Pack the hypotheses results
            packed_result = pack_hypotheses(hypotheses, timesteps, logitlen, alignments=alignments)

            del hypotheses, timesteps

        self.decoder.train(decoder_training_state)
        self.joint.train(joint_training_state)

        return (packed_result,)

    def _greedy_decode_blank_as_pad(self, x: torch.Tensor, out_len: torch.Tensor, device: torch.device):
        with torch.no_grad():
            # x: [B, T, D]
            # out_len: [B]
            # device: torch.device

            # Initialize state
            hidden = None
            batchsize = x.shape[0]

            # Output string buffer
            label = [[] for _ in range(batchsize)]
            timesteps = [[] for _ in range(batchsize)]

            # If alignments need to be preserved, register a danling list to hold the values
            if self.preserve_alignments:
                # alignments is a 3-dimensional dangling list representing B x T x U
                alignments = []
                for _ in range(batchsize):
                    alignments.append([[]])
            else:
                alignments = None

            # Last Label buffer + Last Label without blank buffer
            # batch level equivalent of the last_label
            last_label = torch.full([batchsize, 1], fill_value=self._blank_index, dtype=torch.long, device=device)

            # Mask buffers
            blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)

            # Get max sequence length
            max_out_len = out_len.max()
            for time_idx in range(max_out_len):
                f = x.narrow(dim=1, start=time_idx, length=1)  # [B, 1, D]

                # Prepare t timestamp batch variables
                not_blank = True
                symbols_added = 0

                # Reset blank mask
                blank_mask.mul_(False)

                # Update blank mask with time mask
                # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
                # Forcibly mask with "blank" tokens, for all sample where current time step T > seq_len
                blank_mask = time_idx >= out_len
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
                    del v, g

                    # Update blank mask with current predicted blanks
                    # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
                    k_is_blank = k == self._blank_index
                    blank_mask.bitwise_or_(k_is_blank)

                    del k_is_blank

                    # If preserving alignments, check if sequence length of sample has been reached
                    # before adding alignment
                    if self.preserve_alignments:
                        # Insert ids into last timestep per sample
                        logp_vals = logp.to('cpu').max(1)[1]
                        for batch_idx in range(batchsize):
                            if time_idx < out_len[batch_idx]:
                                alignments[batch_idx][-1].append(logp_vals[batch_idx])
                        del logp_vals
                    del logp

                    # If all samples predict / have predicted prior blanks, exit loop early
                    # This is equivalent to if single sample predicted k
                    if blank_mask.all():
                        not_blank = False

                        # If preserving alignments, convert the current Uj alignments into a torch.Tensor
                        # Then preserve U at current timestep Ti
                        # Finally, forward the timestep history to Ti+1 for that sample
                        # All of this should only be done iff the current time index <= sample-level AM length.
                        # Otherwise ignore and move to next sample / next timestep.
                        if self.preserve_alignments:

                            # convert Ti-th logits into a torch array
                            for batch_idx in range(batchsize):

                                # this checks if current timestep <= sample-level AM length
                                # If current timestep > sample-level AM length, no alignments will be added
                                # Therefore the list of Uj alignments is empty here.
                                if len(alignments[batch_idx][-1]) > 0:
                                    alignments[batch_idx].append([])  # blank buffer for next timestep
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
                        last_label = k.clone().view(-1, 1)
                        hidden = hidden_prime

                        # Update predicted labels, accounting for time mask
                        # If blank was predicted even once, now or in the past,
                        # Force the current predicted label to also be blank
                        # This ensures that blanks propogate across all timesteps
                        # once they have occured (normally stopping condition of sample level loop).
                        for kidx, ki in enumerate(k):
                            if blank_mask[kidx] == 0:
                                label[kidx].append(ki)
                                timesteps[kidx].append(time_idx)

                        symbols_added += 1

            # Remove trailing empty list of alignments at T_{am-len} x Uj
            if self.preserve_alignments:
                for batch_idx in range(batchsize):
                    if len(alignments[batch_idx][-1]) == 0:
                        del alignments[batch_idx][-1]

        return label, timesteps, alignments

    @torch.no_grad()
    def _greedy_decode_masked(self, x: torch.Tensor, out_len: torch.Tensor, device: torch.device):
        # x: [B, T, D]
        # out_len: [B]
        # device: torch.device

        # Initialize state
        hidden = None
        batchsize = x.shape[0]

        # Output string buffer
        label = [[] for _ in range(batchsize)]
        timesteps = [[] for _ in range(batchsize)]

        # If alignments need to be preserved, register a danling list to hold the values
        if self.preserve_alignments:
            # alignments is a 3-dimensional dangling list representing B x T x U
            alignments = []
            for _ in range(batchsize):
                alignments.append([[]])
        else:
            alignments = None

        # Last Label buffer + Last Label without blank buffer
        # batch level equivalent of the last_label
        last_label = torch.full([batchsize, 1], fill_value=self._blank_index, dtype=torch.long, device=device)
        last_label_without_blank = last_label.clone()

        # Mask buffers
        blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)

        # Get max sequence length
        max_out_len = out_len.max()
        for time_idx in range(max_out_len):
            f = x.narrow(dim=1, start=time_idx, length=1)  # [B, 1, D]

            # Prepare t timestamp batch variables
            not_blank = True
            symbols_added = 0

            # Reset blank mask
            blank_mask.mul_(False)

            # Update blank mask with time mask
            # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
            # Forcibly mask with "blank" tokens, for all sample where current time step T > seq_len
            blank_mask = time_idx >= out_len

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
                del v, g

                # Update blank mask with current predicted blanks
                # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
                k_is_blank = k == self._blank_index
                blank_mask.bitwise_or_(k_is_blank)

                # If preserving alignments, check if sequence length of sample has been reached
                # before adding alignment
                if self.preserve_alignments:
                    # Insert ids into last timestep per sample
                    logp_vals = logp.to('cpu').max(1)[1]
                    for batch_idx in range(batchsize):
                        if time_idx < out_len[batch_idx]:
                            alignments[batch_idx][-1].append(logp_vals[batch_idx])
                    del logp_vals
                del logp

                # If all samples predict / have predicted prior blanks, exit loop early
                # This is equivalent to if single sample predicted k
                if blank_mask.all():
                    not_blank = False

                    # If preserving alignments, convert the current Uj alignments into a torch.Tensor
                    # Then preserve U at current timestep Ti
                    # Finally, forward the timestep history to Ti+1 for that sample
                    # All of this should only be done iff the current time index <= sample-level AM length.
                    # Otherwise ignore and move to next sample / next timestep.
                    if self.preserve_alignments:

                        # convert Ti-th logits into a torch array
                        for batch_idx in range(batchsize):

                            # this checks if current timestep <= sample-level AM length
                            # If current timestep > sample-level AM length, no alignments will be added
                            # Therefore the list of Uj alignments is empty here.
                            if len(alignments[batch_idx][-1]) > 0:
                                alignments[batch_idx].append([])  # blank buffer for next timestep
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
                    for kidx, ki in enumerate(k):
                        if blank_mask[kidx] == 0:
                            label[kidx].append(ki)
                            timesteps[kidx].append(time_idx)

                symbols_added += 1

        # Remove trailing empty list of alignments at T_{am-len} x Uj
        if self.preserve_alignments:
            for batch_idx in range(batchsize):
                if len(alignments[batch_idx][-1]) == 0:
                    del alignments[batch_idx][-1]

        return label, timesteps, alignments


@dataclass
class GreedyRNNTInferConfig:
    max_symbols_per_step: Optional[int] = None
    preserve_alignments: bool = False


@dataclass
class GreedyBatchedRNNTInferConfig:
    max_symbols_per_step: Optional[int] = None
    preserve_alignments: bool = False
