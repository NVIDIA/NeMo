# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from nemo.utils import logging


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


class StatelessNetMultiHead(torch.nn.Module):
    def __init__(
        self,
        context_size: int,
        vocab_size: int,
        emb_dim: int,
        blank_idx: int,
        n_heads: int,
        learnable_pos: bool,
        dropout: float = 0.1,
        normalization_mode: str = "layer",
        **kwargs,
    ) -> None:
        """
        Implementation of the stateless net with explicit learneable or non-learneable positional 
        encoding with multi-head processing. 

        It is extremly important to observe correct dimensions on each step, so
        this letters will be used for correctness check and comments:
            B - batch size
            C - context size
            U - labels num in padded labels tensor
            H - head nums
            D - emdeddings dimensional
            hd - head emdeddings dimensional

        Args:
            context_size: history context size for the stateless decoder network. Could be any positive integer. We recommend setting this as 2.
            vocab_size: total vocabulary size.
            emb_dim: total embedding size of the stateless net output.
            blank_idx: index for the blank symbol for the transducer model.
            normalization_mode: normalization run on the output embeddings. Could be either 'layer' or None. We recommend using 'layer' to stabilize training.
            dropout: dropout rate on the embedding outputs.
        """
        super().__init__()

        assert context_size > 0
        assert emb_dim % n_heads == 0
        assert n_heads > 0

        self.context_size = context_size
        self.emb_dim = emb_dim
        self.blank_idx = blank_idx
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.head_emb_dim = emb_dim // n_heads

        self.shared_embed = torch.nn.Embedding(vocab_size + 1, self.emb_dim, padding_idx=blank_idx,)

        # Prepeare possitional encoder matrix, for each head
        # it will be corresponding possitional encoders; may be
        # learnable or not;
        self.pos_embs = torch.nn.Parameter(
            torch.Tensor(size=(n_heads, self.head_emb_dim, context_size)), requires_grad=learnable_pos,
        )
        torch.nn.init.xavier_uniform_(self.pos_embs)

        self.norm = torch.nn.Identity()
        if normalization_mode == "layer":
            self.norm = torch.nn.LayerNorm(self.emb_dim, eps=1e-5, elementwise_affine=False)
        else:
            self.norm = torch.nn.Identity()

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, y: torch.Tensor, state: Optional[List[torch.Tensor]] = None):
        """
            y: torch.tensor of size [B, U], where B - batch size, U - padded labels
                indixes;
            state: equals to None if it is training or second step of rnnt decoding,
                otherwise equals to list with only one element - torch.tensor of 
                size [B, C - 1], where C is context size;
        """
        B, U = y.shape

        if U == 0:
            logging.warning("Zero label dimension in stateless prediction network forward.")
            outs = torch.tensor(data=[], dtype=torch.float32, device=y.device).reshape(B, 0, self.emb_dim)
            state = torch.ones([B, self.context_size - 1], dtype=torch.long, device=y.device) * self.blank_idx
            return outs, [state]

        if state is not None:
            appended_y = torch.concat([state[0], y], axis=1)  # (В, U + C - 1)
        else:
            padded_state = (
                torch.ones([B, self.context_size - 1], dtype=torch.long, device=y.device) * self.blank_idx
            )  # (B, C - 1)
            appended_y = torch.concat([padded_state, y], axis=1)  # (В, U + C - 1), U + C - 1 >= C

        # get embeddings from one shared embedder
        appended_y_embs = self.shared_embed(appended_y)  # (В, U + C - 1, D)

        # split embeddings into folds for windowed operations
        appended_y_embs = appended_y_embs.unfold(dimension=1, size=self.context_size, step=1)  # B x U x D x C
        appended_y_embs = appended_y_embs.permute(0, 2, 1, 3)  # B x D x U x C
        appended_y_embs = appended_y_embs.reshape(
            B, self.n_heads, self.head_emb_dim, U, self.context_size
        )  # (B, n_heads, hd, U, C)

        weights = torch.einsum('bnhuc,nhc->bnuc', appended_y_embs, self.pos_embs)
        weighted_y_embs = torch.einsum('bnhuc,bnuc->bnhu', appended_y_embs, weights)
        weighted_y_embs = weighted_y_embs.reshape(B, -1, U).permute(0, 2, 1)  # (B, U, D)

        weighted_y_embs = self.dropout(weighted_y_embs)  # B x U x D
        weighted_y_embs = self.norm(weighted_y_embs)  # B x U x D

        state = [appended_y[:, appended_y.shape[1] - self.context_size + 1 :]]  # B x (C - 1)

        return weighted_y_embs, state  # (B, U, D), B x (C - 1)
