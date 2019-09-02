from functools import partial

import torch
from torch.utils.data import DataLoader, DistributedSampler

from nemo.backends.pytorch.common.parts import TextDataset
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core import DeviceType
from nemo.core.neural_types import *
from nemo.utils.misc import pad_to


class TextDataLayer(DataLayerNM):
    """A simple Neural Module for loading textual data

    Args:
        path: (str) Path to file with newline separate strings of text
        labels (list): List of string labels to use when to str2int translation
        eos_id (int): Label position of end of string symbol
        pad_id (int): Label position of padding symbol
        batch_size (int): Size of batches to generate in data loader
        drop_last (bool): Whether we drop last (possibly) incomplete batch.
            Defaults to False.
        num_workers (int): Number of processes to work on data loading (0 for
            just main process).
            Defaults to 0.

    """

    @staticmethod
    def create_ports():
        input_ports = {}
        output_ports = {
            'texts': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }
        return input_ports, output_ports

    def __init__(self, path, labels, eos_id, pad_id,
                 batch_size, drop_last=False, num_workers=0,
                 **kwargs):
        super().__init__(**kwargs)

        self._dataset = TextDataset(path, labels, eos_id)

        if self._placement == DeviceType.AllGpu:
            sampler = DistributedSampler(self._dataset)
        else:
            sampler = None

        # noinspection PyTypeChecker
        self._dataloader = DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=partial(self._collate_fn, pad_id=pad_id, pad8=True),
            drop_last=drop_last,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=num_workers
        )

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        # return self._dataset
        return None

    @property
    def data_iterator(self):
        return self._dataloader

    @staticmethod
    def _collate_fn(batch_list, pad_id, pad8=False):
        max_len = max(len(s) for s in batch_list)
        if pad8:
            max_len = pad_to(max_len, 8)

        texts = torch.empty(len(batch_list), max_len,
                            dtype=torch.long)
        texts.fill_(pad_id)

        for i, s in enumerate(batch_list):
            texts[i].narrow(0, 0, s.size(0)).copy_(s)

        assert len(texts.shape) == 2

        return texts
