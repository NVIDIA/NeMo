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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms.

    score: A float score obtained from an AbstractRNNTDecoder module's score_hypothesis method.

    y_sequence: Either a sequence of integer ids pointing to some vocabulary, or a packed torch.Tensor
        behaving in the same manner. dtype must be torch.Long in the latter case.

    dec_state: A list (or list of list) of LSTM-RNN decoder states. Can be None.

    text: (Optional) A decoded string after processing via CTC / RNN-T decoding (removing the CTC/RNNT
        `blank` tokens, and optionally merging word-pieces). Should be used as decoded string for
        Word Error Rate calculation.

    timestep: (Optional) A list of integer indices representing at which index in the decoding
        process did the token appear. Should be of same length as the number of non-blank tokens.

    alignments: (Optional) Represents the CTC / RNNT token alignments as integer tokens along an axis of
        time T (for CTC) or Time x Target (TxU).
        For CTC, represented as a single list of integer indices.
        For RNNT, represented as a dangling list of list of integer indices.
        Outer list represents Time dimension (T), inner list represents Target dimension (U).
        The set of valid indices **includes** the CTC / RNNT blank token in order to represent alignments.

    length: Represents the length of the sequence (the original length without padding), otherwise
        defaults to 0.

    y: (Unused) A list of torch.Tensors representing the list of hypotheses.

    lm_state: (Unused) A dictionary state cache used by an external Language Model.

    lm_scores: (Unused) Score of the external Language Model.

    tokens: (Optional) A list of decoded tokens (can be characters or word-pieces.

    last_token (Optional): A token or batch of tokens which was predicted in the last step.
    """

    score: float
    y_sequence: Union[List[int], torch.Tensor]
    text: Optional[str] = None
    dec_out: Optional[List[torch.Tensor]] = None
    dec_state: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor]]] = None
    timestep: Union[List[int], torch.Tensor] = field(default_factory=list)
    alignments: Optional[Union[List[int], List[List[int]]]] = None
    length: Union[int, torch.Tensor] = 0
    y: List[torch.tensor] = None
    lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
    lm_scores: Optional[torch.Tensor] = None
    tokens: Optional[Union[List[int], torch.Tensor]] = None
    last_token: Optional[torch.Tensor] = None


@dataclass
class NBestHypotheses:
    """List of N best hypotheses"""

    n_best_hypotheses: Optional[List[Hypothesis]]


def is_prefix(x: List[int], pref: List[int]) -> bool:
    """
    Obtained from https://github.com/espnet/espnet.

    Check if pref is a prefix of x.

    Args:
        x: Label ID sequence.
        pref: Prefix label ID sequence.

    Returns:
        : Whether pref is a prefix of x.
    """
    if len(pref) >= len(x):
        return False

    for i in range(len(pref)):
        if pref[i] != x[i]:
            return False

    return True


def select_k_expansions(
    hyps: List[Hypothesis], topk_idxs: torch.Tensor, topk_logps: torch.Tensor, gamma: float, beta: int,
) -> List[Tuple[int, Hypothesis]]:
    """
    Obtained from https://github.com/espnet/espnet

    Return K hypotheses candidates for expansion from a list of hypothesis.
    K candidates are selected according to the extended hypotheses probabilities
    and a prune-by-value method. Where K is equal to beam_size + beta.

    Args:
        hyps: Hypotheses.
        topk_idxs: Indices of candidates hypothesis. Shape = [B, num_candidates]
        topk_logps: Log-probabilities for hypotheses expansions. Shape = [B, V + 1]
        gamma: Allowed logp difference for prune-by-value method.
        beta: Number of additional candidates to store.

    Return:
        k_expansions: Best K expansion hypotheses candidates.
    """
    k_expansions = []

    for i, hyp in enumerate(hyps):
        hyp_i = [(int(k), hyp.score + float(v)) for k, v in zip(topk_idxs[i], topk_logps[i])]
        k_best_exp_val = max(hyp_i, key=lambda x: x[1])

        k_best_exp_idx = k_best_exp_val[0]
        k_best_exp = k_best_exp_val[1]

        expansions = sorted(filter(lambda x: (k_best_exp - gamma) <= x[1], hyp_i), key=lambda x: x[1],)

        if len(expansions) > 0:
            k_expansions.append(expansions)
        else:
            k_expansions.append([(k_best_exp_idx, k_best_exp)])

    return k_expansions


class StatelessNet(torch.nn.Module):
    """
    Helper class used in transducer models with stateless decoders. This stateless
    simply outputs embedding or concatenated embeddings for the input label[s],
    depending on the configured context size.

    Args:
        context_size: history context size for the stateless decoder network. Could be any positive integer. We recommend setting this as 2.
        vocab_size: total vocabulary size.
        emb_dim: total embedding size of the stateless net output.
        blank_idx: index for the blank symbol for the transducer model.
        normalization_mode: normalization run on the output embeddings. Could be either 'layer' or None. We recommend using 'layer' to stabilize training.
        dropout: dropout rate on the embedding outputs.
    """

    def __init__(self, context_size, vocab_size, emb_dim, blank_idx, normalization_mode, dropout):
        super().__init__()
        assert context_size > 0
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.Identity()
        if normalization_mode == 'layer':
            self.norm = torch.nn.LayerNorm(emb_dim, elementwise_affine=False)

        embeds = []
        for i in range(self.context_size):
            # We use different embedding matrices for different context positions.
            # In this list, a smaller index means more recent history word.
            # We assign more dimensions for the most recent word in the history.
            # The detailed method is, we first allocate half the embedding-size
            # to the most recent history word, and then allocate the remaining
            # dimensions evenly among all history contexts. E.g. if total embedding
            # size is 200, and context_size is 2, then we allocate 150 dimensions
            # to the last word, and 50 dimensions to the second-to-last word.
            if i != 0:
                embed_size = emb_dim // 2 // self.context_size
            else:
                embed_size = emb_dim - (emb_dim // 2 // self.context_size) * (self.context_size - 1)

            embed = torch.nn.Embedding(vocab_size + 1, embed_size, padding_idx=blank_idx)
            embeds.append(embed)

        self.embeds = torch.nn.ModuleList(embeds)
        self.blank_idx = blank_idx

    def forward(
        self, y: Optional[torch.Tensor] = None, state: Optional[List[torch.Tensor]] = None,
    ):
        """
        Although this is a *stateless* net, we use the "state" parameter to
        pass in the previous labels, unlike LSTMs where state would represent
        hidden activations of the network.

        Args:
            y: a Integer tensor of shape B x U.
            state: a list of 1 tensor in order to be consistent with the stateful
                   decoder interface, and the element is a tensor of shape [B x context-length].

        Returns:
            The return dimension of this function's output is B x U x D, with D being the total embedding dim.
        """
        outs = []

        [B, U] = y.shape
        appended_y = y
        if state != None:
            appended_y = torch.concat([state[0], y], axis=1)
            context_size = appended_y.shape[1]

            if context_size < self.context_size:
                # This is the case at the beginning of an utterance where we have
                # seen less words than context_size. In this case, we need to pad
                # it to the right length.
                padded_state = torch.ones([B, self.context_size], dtype=torch.long, device=y.device) * self.blank_idx
                padded_state[:, self.context_size - context_size :] = appended_y
            elif context_size == self.context_size + 1:
                padded_state = appended_y[:, 1:]
                # This is the case where the previous state already has reached context_size.
                # We need to truncate the history by omitting the 0'th token.
            else:
                # Context has just the right size. Copy directly.
                padded_state = appended_y

            for i in range(self.context_size):
                out = self.embeds[i](padded_state[:, self.context_size - 1 - i : self.context_size - i])
                outs.append(out)
        else:
            for i in range(self.context_size):
                out = self.embeds[i](y)

                if i != 0:
                    out[:, i:, :] = out[
                        :, :-i, :
                    ].clone()  # needs clone() here or it might complain about src and dst mem location have overlaps.
                    out[:, :i, :] *= 0.0
                outs.append(out)

        out = self.dropout(torch.concat(outs, axis=-1))
        out = self.norm(out)

        state = None
        if y is not None:
            state = [appended_y[:, appended_y.shape[1] - self.context_size + 1 :]]
        return out, state
