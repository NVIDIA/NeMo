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

from contextlib import contextmanager

import torch

from nemo.collections.common.parts import NEG_INF, mask_padded_tokens

__all__ = [
    "GreedySequenceGenerator",
    "TopKSequenceGenerator",
    "BeamSearchSequenceGenerator",
]


class GreedySequenceGenerator:
    """
    Greedy sequence generator based on the decoder followed by log_softmax.

    Args:
        embedding: nn.Module, transforms input_ids into vector embeddings
        decoder: nn.Module, takes embeddings and produces hidden_states
        log_softmax: nn.Module, takes hidden_states and produces log_probs
            which correspond to probability distribution of tokens (ids)
        pad: index of padding token in the vocabulary
        bos: index of beginning of sequence token in the vocabulary
        eos: index of end of sequence token in the vocabulary
        max_sequence_length: maximum allowed length for generated sequences
        max_delta_length: in case of encoder-decoder generation (e.g. NMT),
            forbids generated sequences to be longer than the length of
            source sequences plus max_delta_length
        batch_size: size of the batch of generated sequences if neither
            source nor target starting sequences are provided
    """

    def __init__(
        self,
        embedding,
        decoder,
        log_softmax,
        pad=0,
        bos=1,
        eos=2,
        max_sequence_length=512,
        max_delta_length=20,
        batch_size=1,
    ):
        super().__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.log_softmax = log_softmax
        self.pad, self.bos, self.eos = pad, bos, eos
        self.max_seq_length = max_sequence_length
        self.max_delta_len = max_delta_length
        self.batch_size = batch_size

    def _one_step_forward(
        self,
        decoder_input_ids=None,
        encoder_hidden_states=None,
        encoder_input_mask=None,
        decoder_mems_list=None,
        pos=0,
    ):
        """
        One step of autoregressive output generation.

        Args:
            decoder_input_ids: starting sequence of tokens to generate from;
                if None, generation will start from a batch of <bos> tokens
            encoder_hidden_states: output of the encoder for conditional
                sequence generation; if None, generator will use unconditional
                mode (e.g., language modeling)
            encoder_input_mask: input mask used in the encoder
            decoder_mems_list: list of size num_layers with cached activations
                of sequence (x[1], ..., x[k-1]) for fast generation of x[k]
            pos: starting position in positional encoding
        """

        decoder_hidden_states = self.embedding.forward(decoder_input_ids, start_pos=pos)
        decoder_input_mask = mask_padded_tokens(decoder_input_ids, self.pad).float()

        if encoder_hidden_states is not None:
            decoder_mems_list = self.decoder.forward(
                decoder_hidden_states,
                decoder_input_mask,
                encoder_hidden_states,
                encoder_input_mask,
                decoder_mems_list,
                return_mems=True,
            )
        else:
            decoder_mems_list = self.decoder.forward(
                decoder_hidden_states, decoder_input_mask, decoder_mems_list, return_mems=True
            )
        log_probs = self.log_softmax.forward(hidden_states=decoder_mems_list[-1][:, -1:])
        return log_probs, decoder_mems_list

    def _prepare_for_search(self, decoder_input_ids=None, encoder_hidden_states=None):
        """
        Helper function which defines starting sequence to begin generating
        with and maximum allowed number of tokens to be generated.
        """

        decoder_parameter = next(self.decoder.parameters())
        batch_size = self.batch_size

        # for encoder-decoder generation, maximum length of generated sequence
        # is min(max_sequence_length, src_len + max_delta_length)
        if encoder_hidden_states is not None:
            batch_size, src_len, _ = encoder_hidden_states.size()
            max_seq_length = min(self.max_seq_length, src_len + self.max_delta_len)
        else:
            max_seq_length = self.max_seq_length

        # if no input is provided, start with the batch of <bos> tokens
        if decoder_input_ids is not None:
            tgt = decoder_input_ids
            batch_size, tgt_len = decoder_input_ids.size()
        else:
            tgt = torch.zeros(batch_size, 1).long().fill_(self.bos).to(decoder_parameter.device)
            tgt_len = 1
        max_generation_length = max_seq_length - tgt_len

        return tgt, batch_size, max_generation_length

    def _forward(self, decoder_input_ids=None, encoder_hidden_states=None, encoder_input_mask=None):

        tgt, batch_size, max_generation_length = self._prepare_for_search(decoder_input_ids, encoder_hidden_states)

        # pad profile tracks sequences ending with <eos> token to replace
        # everything after <eos> with <pad> token
        decoder_parameter = next(self.decoder.parameters())
        pad_profile = torch.zeros(batch_size, 1).long().to(decoder_parameter.device)

        decoder_mems_list = None
        for i in range(max_generation_length):

            log_probs, decoder_mems_list = self._one_step_forward(
                tgt[:, -1:], encoder_hidden_states, encoder_input_mask, decoder_mems_list, i
            )

            next_tokens = torch.argmax(log_probs[:, -1], dim=-1, keepdim=True)
            next_tokens = self.pad * pad_profile + next_tokens * (1 - pad_profile)
            pad_profile = torch.max(pad_profile, (next_tokens == self.eos).long())
            tgt = torch.cat((tgt, next_tokens), dim=-1)

            # abort generation if all sequences end with <eos>
            if pad_profile.sum() == batch_size:
                break

        return tgt

    def __call__(self, decoder_input_ids=None, encoder_hidden_states=None, encoder_input_mask=None):
        with self.as_frozen():
            return self._forward(decoder_input_ids, encoder_hidden_states, encoder_input_mask)

    def freeze(self) -> None:
        """Freeze weights of embedding, decoder, and classification layers to prevent memory leak.
        """
        for param in self.embedding.parameters():
            param.requires_grad = False
        self.embedding.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.eval()
        for param in self.log_softmax.parameters():
            param.require_grad = False
        self.log_softmax.eval()

    def unfreeze(self) -> None:
        """Unfreeze weights of embedding, decoder, and classification layers.
        """
        for param in self.embedding.parameters():
            param.requires_grad = True
        self.embedding.train()
        for param in self.decoder.parameters():
            param.requires_grad = True
        self.decoder.train()
        for param in self.log_softmax.parameters():
            param.require_grad = True
        self.log_softmax.train()

    @contextmanager
    def as_frozen(self):
        """
        Context manager which temporarily freezes embedding, decoder, and log_softmax modules,
        yields control and finally unfreezes the modules.
        """
        self.freeze()

        try:
            yield
        finally:
            self.unfreeze()


class TopKSequenceGenerator(GreedySequenceGenerator):
    """
    Top-k sequence generator based on the decoder followed by log_softmax.

    Args:
        *all args of GreedySequenceGenerator class
        beam_size: size of the beam (parameter k in top-k)
        temperature: temperature of top-k sampling, all logits are divided
            by temperature before rescaling. High temperature leads to
            uniform distribution, low leads to delta-like distribution.
    Kwargs:
        all remaining parameters of GreedySequenceGenerator class
    """

    def __init__(self, embedding, decoder, log_softmax, beam_size=1, temperature=1.0, **kwargs):
        super().__init__(embedding, decoder, log_softmax, **kwargs)
        self.beam_size = beam_size
        self.temp = temperature

    # @torch.no_grad()
    def _one_step_forward(
        self,
        decoder_input_ids=None,
        encoder_hidden_states=None,
        encoder_input_mask=None,
        decoder_mems_list=None,
        pos=0,
    ):
        log_probs, decoder_mems_list = super()._one_step_forward(
            decoder_input_ids, encoder_hidden_states, encoder_input_mask, decoder_mems_list, pos
        )

        batch_size, seq_len, vocab_size = log_probs.size()
        scores, indices = torch.topk(log_probs, self.beam_size, dim=-1)

        rescaled_logexp = torch.zeros_like(log_probs).scatter(-1, indices, scores.div(self.temp).exp())
        probs = rescaled_logexp / rescaled_logexp.norm(1, -1, keepdim=True)

        # We randomly sample next tokens from rescaled probability distribution
        # over top-k candidates and return a binary tensor which indicates
        # candidates that have been selected. We call this object
        # `pseudo_log_probs` as genuine log_probs should have -infs instead of
        # 0s and 0s instead of 1s.
        ids = torch.multinomial(probs.view(-1, vocab_size), 1).view(-1, seq_len, 1)
        pseudo_log_probs = torch.zeros_like(log_probs).scatter(-1, ids, 1.0)

        return pseudo_log_probs, decoder_mems_list


class BeamSearchSequenceGenerator(GreedySequenceGenerator):
    def __init__(self, embedding, decoder, log_softmax, beam_size=1, len_pen=0, **kwargs):
        """
        Beam Search sequence generator based on the decoder followed by
        log_softmax.

        Args:
            *all args of GreedySequenceGenerator class
            beam_size: size of the beam
            len_pen: length penalty parameter
        Kwargs:
            all remaining parameters of GreedySequenceGenerator class
        """

        super().__init__(embedding, decoder, log_softmax, **kwargs)
        self.beam_size = beam_size
        self.len_pen = len_pen

    @staticmethod
    def compute_len_penalty(lengths, alpha):
        """Returns length penalty according to https://arxiv.org/pdf/1609.08144.pdf"""
        return ((5 + lengths) / 6).pow(alpha)

    def _forward(self, decoder_input_ids=None, encoder_hidden_states=None, encoder_input_mask=None):
        tgt, batch_size, max_generation_length = self._prepare_for_search(decoder_input_ids, encoder_hidden_states)

        # generate initial buffer of beam_size prefixes-hypotheses
        log_probs, decoder_mems_list = self._one_step_forward(tgt, encoder_hidden_states, encoder_input_mask, None, 0)
        scores, prefixes = torch.topk(log_probs.permute(0, 2, 1), self.beam_size, dim=1)
        scores, prefixes = scores.view(-1, 1), prefixes.view(-1, 1)

        # repeat init target prefixes and cached memory states beam_size times
        prefixes = torch.cat((tgt.repeat(1, self.beam_size).view(-1, 1), prefixes), dim=1)
        for j in range(len(decoder_mems_list)):
            decoder_mems_list[j] = decoder_mems_list[j].repeat(self.beam_size, 1, 1)

        # repeat source sequence beam_size times for beam search
        if encoder_hidden_states is not None:
            _, src_length, hidden_size = encoder_hidden_states.size()
            encoder_input_mask = encoder_input_mask.repeat(1, self.beam_size).view(-1, src_length)
            encoder_hidden_states = encoder_hidden_states.repeat(1, self.beam_size, 1).view(
                -1, src_length, hidden_size
            )
        else:
            hidden_size = decoder_mems_list[0].size(2)

        # pad_profile tracks finished hypotheses to generate only <pad> tokens
        # if <eos> or <pad> has been generated
        pad_profile = torch.zeros_like(scores).long()

        # prefixes_len tracks lengths of generated hypotheses to perform
        # length penalty correction
        prefixes_len = torch.zeros_like(scores).fill_(prefixes.size(1) + 1)

        for i in range(max_generation_length):

            # mask all finished hypotheses to exclude them from beam
            pad_mask = pad_profile.repeat(1, self.beam_size)

            # generate and score candidates for prefixes continuation
            log_probs, decoder_mems_list = self._one_step_forward(
                prefixes[:, -1:], encoder_hidden_states, encoder_input_mask, decoder_mems_list, i + 1
            )
            scores_i, prefixes_i = torch.topk(log_probs[:, -1, :], self.beam_size, dim=-1)

            # for all prefixes ending with <eos> or <pad> replace generated
            # continuations with <pad>
            prefixes_i = self.pad * pad_mask + prefixes_i * (1 - pad_mask)

            # force all hypotheses but one generated from already finished
            # hypotheses to have extremely low score, so they will not be
            # considered during beam re-ranking
            pad_mask[:, 1:] = pad_mask[:, 1:] * NEG_INF
            scores = scores + scores_i * (1 - pad_mask).to(scores.dtype)

            # choose top-k hypotheses with length penalty applied
            len_penalties = self.compute_len_penalty(prefixes_len, self.len_pen)
            scores = scores / len_penalties
            scores, indices_i = torch.topk(scores.view(-1, self.beam_size ** 2), self.beam_size, dim=1)
            scores = scores.view(-1, 1) * len_penalties

            # select prefixes which correspond to the chosen hypotheses
            prefixes = prefixes.unsqueeze(1).repeat(1, self.beam_size, 1)
            prefixes = torch.cat((prefixes, prefixes_i.unsqueeze(2)), dim=2)
            prefixes = prefixes.view(batch_size, self.beam_size ** 2, -1)
            p_len = prefixes.size(2)
            prefixes_ids = indices_i.unsqueeze(2).repeat(1, 1, p_len)
            prefixes = prefixes.gather(1, prefixes_ids).view(-1, p_len)

            # reshuffle cached decoder memory states to restore the order
            # of hypotheses broken after top-k selection
            mems_ids = indices_i.unsqueeze(2).unsqueeze(3).repeat(1, 1, p_len - 1, hidden_size) // self.beam_size
            for j in range(len(decoder_mems_list)):
                decoder_mems_list[j] = (
                    decoder_mems_list[j]
                    .view(-1, self.beam_size, p_len - 1, hidden_size)
                    .gather(1, mems_ids)
                    .view(-1, p_len - 1, hidden_size)
                )

            # update prefixes_len and pad_profile
            not_eos_pad = prefixes.ne(self.eos) & prefixes.ne(self.pad)
            prefixes_len = 1 + not_eos_pad.sum(dim=1, keepdim=True).to(scores.dtype)
            pad_profile = (~not_eos_pad[:, -1:]).long()

            # if all hypotheses end with <eos> or <pad>, interrupt search
            if pad_profile.sum() == batch_size * self.beam_size:
                break

        # select best performing hypotheses in each element of the batch
        len_penalties = self.compute_len_penalty(prefixes_len, self.len_pen)
        scores = scores / len_penalties
        best_guesses = (
            torch.argmax(scores.view(-1, self.beam_size), dim=1, keepdim=True).repeat(1, prefixes.size(1)).unsqueeze(1)
        )
        tgt = prefixes.view(batch_size, self.beam_size, -1).gather(1, best_guesses)

        return tgt.squeeze(1)
