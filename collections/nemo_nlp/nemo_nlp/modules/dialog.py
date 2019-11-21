__all__ = ['DSTGenerator',
           'DSTMaskedCrossEntropy']

import numpy as np
import torch
from torch import nn as nn

import nemo
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
            'input_lengths': NeuralType({
                0: AxisType(BatchTag),
            }),
            'story': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            'max_res_len': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            'targets': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag),

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
                1: AxisType(TimeTag),
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
                 batch_size=16,
                 teacher_forcing=0.5):
        super().__init__()
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.embedding = embeddings
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hid_size, hid_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hid_size
        self.w_ratio = nn.Linear(3 * hid_size, 1)
        self.w_gate = nn.Linear(hid_size, nb_gate)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots

        # Create independent slot embeddings
        self.slot_w2i = {}
        for slot in self.slots:
            if slot.split("-")[0] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[0]] = len(self.slot_w2i)
            if slot.split("-")[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[1]] = len(self.slot_w2i)
        self.slot_emb = nn.Embedding(len(self.slot_w2i), hid_size)
        self.slot_emb.weight.data.normal_(0, 0.1)
        self.batch_size = batch_size

    def forward(self,
                encoder_hidden,
                encoder_outputs,
                input_lens,
                story,
                max_res_len,
                targets):
        if (not self.training) \
                or (random.random() <= self.teacher_forcing):
            use_teacher_forcing = False
        else:
            use_teacher_forcing = True

        batch_size = self.batch_size
        all_point_outputs = torch.zeros(len(self.slots),
                                        batch_size,
                                        max_res_len,
                                        self.vocab_size)
        all_gate_outputs = torch.zeros(len(self.slots),
                                       batch_size,
                                       self.nb_gate)
        all_point_outputs = all_point_outputs.to(self._device)
        all_gate_outputs = all_gate_outputs.to(self._device)

        # Get the slot embedding
        slot_emb_dict = {}
        for i, slot in enumerate(self.slots):
            # Domain embbeding
            if slot.split("-")[0] in self.slot_w2i.keys():
                domain_w2idx = [self.slot_w2i[slot.split("-")[0]]]
                domain_w2idx = torch.tensor(domain_w2idx)
                domain_w2idx = domain_w2idx.to(self._device)
                domain_emb = self.slot_emb(domain_w2idx)
            # Slot embbeding
            if slot.split("-")[1] in self.slot_w2i.keys():
                slot_w2idx = [self.slot_w2i[slot.split("-")[1]]]
                slot_w2idx = torch.tensor(slot_w2idx)
                slot_w2idx = slot_w2idx.to(self._device)
                slot_emb = self.slot_emb(slot_w2idx)

            # Combine two embeddings as one query
            combined_emb = domain_emb + slot_emb
            slot_emb_dict[slot] = combined_emb
            slot_emb_exp = combined_emb.expand_as(encoder_hidden)
            if i == 0:
                slot_emb_arr = slot_emb_exp.clone()
            else:
                slot_emb_arr = torch.cat((slot_emb_arr, slot_emb_exp), dim=0)

        # Compute pointer-generator output,
        # puting all (domain, slot) in one batch
        decoder_input = self.dropout(slot_emb_arr).view(-1, self.hidden_size)
        # (batch*|slot|) * emb
        hidden = encoder_hidden.repeat(1, len(self.slots), 1)
        # 1 * (batch*|slot|) * emb

        for wi in range(max_res_len):
            dec_state, hidden = self.gru(
                decoder_input.expand_as(hidden), hidden)

            enc_out = encoder_outputs.repeat(len(self.slots), 1, 1)
            enc_len = input_lens * len(self.slots)
            context_vec, logits, prob = self.attend(
                enc_out, hidden.squeeze(0), enc_len)

            if wi == 0:
                all_gate_outputs = torch.reshape(
                    self.w_gate(context_vec), all_gate_outputs.size())

            p_vocab = self.attend_vocab(
                self.embedding.weight, hidden.squeeze(0))
            p_gen_vec = torch.cat(
                [dec_state.squeeze(0), context_vec, decoder_input], -1)
            vocab_pointer_switches = self.sigmoid(self.w_ratio(p_gen_vec))
            p_context_ptr = torch.zeros(p_vocab.size())
            p_context_ptr = p_context_ptr.to(self._device)

            p_context_ptr.scatter_add_(
                1, story.repeat(len(self.slots), 1), prob)

            final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
            pred_word = torch.argmax(final_p_vocab, dim=1)
            words = [self.lang.index2word[w_idx.item()]
                     for w_idx in pred_word]

            all_point_outputs[:, :, wi, :] = torch.reshape(
                final_p_vocab, (len(self.slots), batch_size, self.vocab_size))

            if use_teacher_forcing:
                decoder_input = self.embedding(torch.flatten(
                    targets[:, :, wi].transpose(1, 0)))
            else:
                decoder_input = self.embedding(pred_word)

            decoder_input = decoder_input.to(self._device)

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
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
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
        # -1 means infered from other dimentions
        logits_flat = logits.view(-1, logits.size(-1))
        # print(logits_flat.size())
        log_probs_flat = torch.log(logits_flat)
        # print("log_probs_flat", log_probs_flat)
        target_flat = targets.view(-1, 1)
        # print("target_flat", target_flat)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        losses = losses_flat.view(*targets.size())  # b * |s| * m
        loss = masking(losses, mask)
        return loss


def masking(losses, mask):
    mask_ = []
    batch_size = mask.size(0)
    max_len = losses.size(2)
    for si in range(mask.size(1)):
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
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
