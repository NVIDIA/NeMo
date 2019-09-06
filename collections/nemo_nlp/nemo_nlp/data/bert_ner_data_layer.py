# Copyright (c) 2019 NVIDIA Corporation
# pylint: disable=E0401, E0602, E0611, E1101

import torch

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import *
from nemo.core import DeviceType
from .datasets import BertNERDataset


class BertNERDataLayer(DataLayerNM):
    @staticmethod
    def create_ports():
        input_ports = {}
        output_ports = {
            "input_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_type_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_mask": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "labels": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "seq_ids": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def __init__(self, *, tokenizer, path_to_data, max_seq_length, **kwargs):
        DataLayerNM.__init__(self, **kwargs)

        self._dataset = BertNERDataset(
            tokenizer=tokenizer,
            input_file=path_to_data,
            max_seq_length=max_seq_length)

    def eval_preds(self, logits, seq_ids, tag_ids):
        return self._dataset.eval_preds(logits, seq_ids, tag_ids)

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None
