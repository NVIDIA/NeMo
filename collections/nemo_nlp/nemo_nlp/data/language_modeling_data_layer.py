# Copyright (c) 2019 NVIDIA Corporation

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import *
from nemo.core import DeviceType
import torch
from .datasets import LanguageModelingDataset


class LanguageModelingDataLayer(DataLayerNM):
    @staticmethod
    def create_ports():
        input_ports = {}
        output_ports = {
            "input_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_mask":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "labels":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }

        return input_ports, output_ports

    def __init__(self, *, tokenizer, dataset, max_seq_length, **kwargs):
        DataLayerNM.__init__(self, **kwargs)

        self._device = torch.device(
            "cuda" if self.placement in [DeviceType.GPU, DeviceType.AllGpu]
            else "cpu"
        )

        self._dataset = LanguageModelingDataset(
            tokenizer=tokenizer,
            dataset=dataset,
            max_sequence_length=max_seq_length,
            batch_step=self.local_parameters.get("batch_step", None)
        )

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None
