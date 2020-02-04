import numpy as np
import torch
from torch.utils import data as pt_data

import nemo
from nemo.collections.nlp.data.datasets import MultiWOZDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core.neural_types import *

__all__ = ['MultiWOZDataLayer']


class MultiWOZDataLayer(TextDataLayer):
    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            "src_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "src_lens": NeuralType({0: AxisType(BatchTag)}),
            "tgt_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag), 2: AxisType(TimeTag)}),
            "tgt_lens": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
            "gating_labels": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
            'turn_domain': NeuralType(None),
        }

    def __init__(
        self,
        data_dir,
        domains,
        all_domains,
        vocab,
        slots,
        gating_dict,
        num_samples=-1,
        batch_size=16,
        mode='train',
        dataset_type=MultiWOZDataset,
        shuffle=False,
        num_workers=0,
        input_dropout=0,
        is_training=False,
    ):

        dataset_params = {
            'data_dir': data_dir,
            'domains': domains,
            'num_samples': num_samples,
            'mode': mode,
            'shuffle': shuffle,
            'all_domains': all_domains,
            'vocab': vocab,
            'slots': slots,
            'gating_dict': gating_dict,
        }
        super().__init__(dataset_type, dataset_params, batch_size=batch_size)

        if self._placement == nemo.core.DeviceType.AllGpu:
            sampler = pt_data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None

        self._dataloader = pt_data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            sampler=sampler,
        )
        self.pad_id = self._dataset.vocab.pad_id
        self.gating_dict = self._dataset.gating_dict
        self.input_dropout = input_dropout
        self.is_training = is_training
        self.vocab = self._dataset.vocab
        self.slots = self._dataset.slots

    def _collate_fn(self, data):
        """ data is a list of batch_size sample
        each sample is a dictionary of features
        """

        def pad_batch_context(sequences):
            '''
            merge from batch * sent_len to batch * max_len
            '''
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            for i, seq in enumerate(sequences):
                sequences[i] = seq + [1] * (max_len - len(seq))
            return torch.tensor(sequences), torch.tensor(lengths)

        def pad_batch_response(sequences, pad_id):
            '''
            merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
            '''
            lengths = []
            for bsz_seq in sequences:
                length = [len(v) for v in bsz_seq]
                lengths.append(length)
            max_len = max([max(l) for l in lengths])
            padded_seqs = []
            for bsz_seq in sequences:
                pad_seq = []
                for v in bsz_seq:
                    v = v + [pad_id] * (max_len - len(v))
                    pad_seq.append(v)
                padded_seqs.append(pad_seq)
            padded_seqs = torch.tensor(padded_seqs)
            lengths = torch.tensor(lengths)
            return padded_seqs, lengths

        data.sort(key=lambda x: len(x['context_ids']), reverse=True)
        item_info = {}
        for key in data[0]:
            item_info[key] = [item[key] for item in data]

        src_ids, src_lens = pad_batch_context(item_info['context_ids'])
        tgt_ids, tgt_lens = pad_batch_response(item_info['responses_ids'], self._dataset.vocab.pad_id)
        gating_label = torch.tensor(item_info['gating_label'])
        turn_domain = torch.tensor(item_info['turn_domain'])

        if self.input_dropout > 0 and self.is_training:
            bi_mask = np.random.binomial([np.ones(src_ids.size())], 1.0 - self.input_dropout)[0]
            rand_mask = torch.Tensor(bi_mask).long().to(src_ids.device)
            src_ids = src_ids * rand_mask

        return (
            src_ids.to(self._device),
            src_lens.to(self._device),
            tgt_ids.to(self._device),
            tgt_lens.to(self._device),
            gating_label.to(self._device),
            turn_domain.to(self._device),
        )

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader
