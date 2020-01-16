__all__ = ['DSTGenerator',
           'DSTMaskedCrossEntropy']

import random

import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from nemo.backends.pytorch.nm import TrainableNM, LossNM
from nemo.core.neural_types import (NeuralType,
                                    AxisType,
                                    BatchTag,
                                    TimeTag,
                                    ChannelTag)


class DSTGenerator(TrainableNM):
    @staticmethod
    def create_ports():
        input_ports = {
            'encoder_hidden': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            'encoder_outputs': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            'input_lens': NeuralType({
                0: AxisType(BatchTag),
            }),
            'src_ids': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            'targets': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(ChannelTag),
                2: AxisType(TimeTag)
            }),

        }
        output_ports = {
            'point_outputs': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag),
                3: AxisType(ChannelTag)
            }),
            'gate_outputs': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(ChannelTag),
                2: AxisType(ChannelTag)
            })
        }
        return input_ports, output_ports

    def __init__(self,
                 vocab,
                 embeddings,
                 hid_size,
                 dropout,
                 slots,
                 nb_gate,
                 teacher_forcing=0.5):
        super().__init__()
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.embedding = embeddings
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(hid_size,
                          hid_size,
                          dropout=dropout,
                          batch_first=True)
        self.nb_gate = nb_gate
        self.hidden_size = hid_size
        self.w_ratio = nn.Linear(3 * hid_size, 1)
        self.w_gate = nn.Linear(hid_size, nb_gate)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots
        self.teacher_forcing = teacher_forcing

        # Create independent slot embeddings
        self._slots_split_to_index()
        self.slot_emb = nn.Embedding(len(self.slot_w2i), hid_size)
        self.slot_emb.weight.data.normal_(0, 0.1)
        self.to(self._device)

    def _slots_split_to_index(self):
        split_slots = [slot.split('-') for slot in self.slots]
        domains = [split_slot[0] for split_slot in split_slots]
        slots = [split_slot[1] for split_slot in split_slots]
        split_slots = list({s: 0 for s in sum(split_slots, [])})
        self.slot_w2i = {split_slots[i]: i for i in range(len(split_slots))}
        self.domain_idx = torch.tensor(
            [self.slot_w2i[domain] for domain in domains], device=self._device)
        self.subslot_idx = torch.tensor(
            [self.slot_w2i[slot] for slot in slots], device=self._device)

    def forward(self,
                encoder_hidden,
                encoder_outputs,
                input_lens,
                src_ids,
                targets=None):

        if (not self.training) \
                or (random.random() > self.teacher_forcing):
            use_teacher_forcing = False
        else:
            use_teacher_forcing = True

        #TODO: set max_res_len to 10 in evaluation mode
        max_res_len = targets.shape[2] # if targets is not None else 10
        batch_size = encoder_hidden.shape[0]

        targets = targets.transpose(0, 1)

        all_point_outputs = torch.zeros(len(self.slots),
                                        batch_size,
                                        max_res_len,
                                        self.vocab_size, device=self._device)
        all_gate_outputs = torch.zeros(len(self.slots),
                                       batch_size,
                                       self.nb_gate, device=self._device)

        domain_emb = self.slot_emb(self.domain_idx).to(self._device)
        subslot_emb = self.slot_emb(self.subslot_idx).to(self._device)
        slot_emb = domain_emb + subslot_emb
        slot_emb = slot_emb.unsqueeze(1)
        slot_emb = slot_emb.repeat(1, batch_size, 1)
        decoder_input = self.dropout(
            slot_emb).view(-1, self.hidden_size)
        hidden = encoder_hidden.transpose(0, 1).repeat(len(self.slots), 1, 1)

        hidden = hidden.view(-1, self.hidden_size).unsqueeze(0)

        enc_len = input_lens.repeat(len(self.slots))
        for wi in range(max_res_len):
            dec_state, hidden = self.rnn(decoder_input.unsqueeze(1),
                                         hidden)

            enc_out = encoder_outputs.repeat(len(self.slots), 1, 1)
            context_vec, logits, prob = self.attend(enc_out,
                                                    hidden.squeeze(0),
                                                    # 480 x 400
                                                    enc_len)

            if wi == 0:
                all_gate_outputs = torch.reshape(self.w_gate(context_vec),
                                                 all_gate_outputs.size())

            p_vocab = self.attend_vocab(self.embedding.weight,
                                        hidden.squeeze(0))
            p_gen_vec = torch.cat(
                [dec_state.squeeze(1), context_vec, decoder_input], -1)
            vocab_pointer_switches = self.sigmoid(self.w_ratio(p_gen_vec))
            p_context_ptr = torch.zeros(p_vocab.size(), device=self._device)

            p_context_ptr.scatter_add_(1,
                                       src_ids.repeat(len(self.slots), 1),
                                       prob)

            final_p_vocab = \
                (1 - vocab_pointer_switches).expand_as(p_context_ptr) \
                * p_context_ptr + \
                vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
            pred_word = torch.argmax(final_p_vocab, dim=1)
            # words = [self.vocab.idx2word[w_idx.item()]
            #          for w_idx in pred_word]

            all_point_outputs[:, :, wi, :] = torch.reshape(
                final_p_vocab,
                (len(self.slots), batch_size, self.vocab_size))

            if use_teacher_forcing:
                decoder_input = self.embedding(torch.flatten(targets[:, :, wi]))
            else:
                decoder_input = self.embedding(pred_word)

            decoder_input = decoder_input.to(self._device)
        all_point_outputs = all_point_outputs.transpose(0, 1).contiguous()
        all_gate_outputs = all_gate_outputs.transpose(0, 1).contiguous()
        return all_point_outputs, all_gate_outputs

    def attend(self, seq, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        scores = F.softmax(scores_, dim=1)
        return scores


class DSTMaskedCrossEntropy(LossNM):
    """
    Neural module which implements Masked Language Modeling (MLM) loss.

    Args:
        label_smoothing (float): label smoothing regularization coefficient
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "logits": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag),
                3: AxisType(ChannelTag)
            }),
            "targets": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(ChannelTag),
                2: AxisType(TimeTag)
            }),
            "mask": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(ChannelTag)
            }),
        }

        output_ports = {"loss": NeuralType(None)}
        return input_ports, output_ports

    def __init__(self):
        LossNM.__init__(self)

    def _loss_function(self, logits, targets, mask):
        logits_flat = logits.view(-1, logits.size(-1))
        eps = 1e-10
        log_probs_flat = torch.log(torch.clamp(logits_flat, min=eps))
        target_flat = targets.view(-1, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        losses = losses_flat.view(*targets.size())  # b * |s| * m
        loss = self.masking(losses, mask)
        return loss

    def masking(self, losses, mask):
        mask_ = []
        batch_size = mask.size(0)
        max_len = losses.size(2)
        for si in range(mask.size(1)):
            seq_range = torch.arange(0, max_len).long()
            seq_range_expand = \
                seq_range.unsqueeze(0).expand(batch_size, max_len)
            if mask[:, si].is_cuda:
                seq_range_expand = seq_range_expand.cuda()
            seq_length_expand = mask[:, si].unsqueeze(
                1).expand_as(seq_range_expand)
            mask_.append((seq_range_expand < seq_length_expand))
        mask_ = torch.stack(mask_)
        mask_ = mask_.transpose(0, 1)
        if losses.is_cuda:
            mask_ = mask_.cuda()
        losses = losses * mask_.float()
        loss = losses.sum() / (mask_.sum().float())
        return loss
