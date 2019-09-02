# Copyright (c) 2019 NVIDIA Corporation

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import *
from nemo.core import DeviceType
import torch
from .datasets import BertPretrainingDataset


class BertPretrainingDataLayer(DataLayerNM):
    @staticmethod
    def create_ports():
        input_ports = {}
        output_ports = {
            "input_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_type_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_mask":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "output_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "output_mask":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "labels":
            NeuralType({0: AxisType(BatchTag)}),
        }

        return input_ports, output_ports

    def __init__(self, *, tokenizer, dataset, name, max_seq_length,
                 sentence_indices_filename=None, mask_probability=0.15,
                 **kwargs):
        DataLayerNM.__init__(self, **kwargs)

        self._device = torch.device(
            "cuda" if self.placement in [DeviceType.GPU, DeviceType.AllGpu]
            else "cpu"
        )

        self._dataset = BertPretrainingDataset(
            tokenizer=tokenizer,
            dataset=dataset,
            name=name,
            sentence_indices_filename=sentence_indices_filename,
            max_length=max_seq_length,
            mask_probability=mask_probability)

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None
