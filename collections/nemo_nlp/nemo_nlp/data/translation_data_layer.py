# Copyright (c) 2019 NVIDIA Corporation

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import *
from nemo.core import DeviceType
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from ..data.datasets.translation import TranslationDataset


class TranslationDataLayer(DataLayerNM):
    @staticmethod
    def create_ports():
        input_ports = {}
        output_ports = {
            "src_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "src_mask":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "tgt_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "tgt_mask":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "labels":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "sent_ids": NeuralType({
                0: AxisType(BatchTag)
            })
        }

        return input_ports, output_ports

    def __init__(
            self, *,
            tokenizer_src,
            tokenizer_tgt,
            dataset_src,
            dataset_tgt,
            tokens_in_batch,
            clean=False,
            **kwargs
    ):
        DataLayerNM.__init__(self, **kwargs)

        self._device = torch.device(
            "cuda" if self.placement in [DeviceType.GPU, DeviceType.AllGpu]
            else "cpu"
        )

        self.translation_dataset = TranslationDataset(
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
            dataset_src=dataset_src,
            dataset_tgt=dataset_tgt,
            tokens_in_batch=tokens_in_batch,
            clean=clean)

        if self._placement == DeviceType.AllGpu:
            sampler = DistributedSampler(self.translation_dataset)
        else:
            sampler = None

        self._dataloader = DataLoader(
            dataset=self.translation_dataset,
            batch_size=1,
            collate_fn=lambda x: self._collate_fn(x),
            shuffle=True if sampler is None else False,
            sampler=sampler)

    def _collate_fn(self, x):
        src_ids, src_mask, tgt_ids, tgt_mask, labels, sent_ids = x[0]
        src_ids = torch.Tensor(src_ids).long().to(self._device)
        src_mask = torch.Tensor(src_mask).float().to(self._device)
        tgt_ids = torch.Tensor(tgt_ids).long().to(self._device)
        tgt_mask = torch.Tensor(tgt_mask).float().to(self._device)
        labels = torch.Tensor(labels).long().to(self._device)
        sent_ids = torch.Tensor(sent_ids).long().to(self._device)
        return src_ids, src_mask, tgt_ids, tgt_mask, labels, sent_ids

    def __len__(self):
        return len(self.translation_dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader
