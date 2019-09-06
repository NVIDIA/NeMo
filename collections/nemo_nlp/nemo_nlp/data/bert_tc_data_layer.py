# Copyright (c) 2019 NVIDIA Corporation

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import *
from nemo.core import DeviceType
import torch
from .datasets import BertTokenClassificationDataset


class BertTokenClassificationDataLayer(DataLayerNM):
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
                1: AxisType(TimeTag),
                2: AxisType(TimeTag)
            }),
            "labels":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "seq_ids":
            NeuralType({0: AxisType(BatchTag)})
        }

        return input_ports, output_ports

    def __init__(self, *, tokenizer, path_to_data, max_seq_length, **kwargs):
        DataLayerNM.__init__(self, **kwargs)
        self._dataset = BertTokenClassificationDataset(
            tokenizer=tokenizer,
            input_file=path_to_data,
            max_seq_length=max_seq_length)

    def eval_preds(self, logits, seq_ids):
        correct_labels, incorrect_labels, correct_preds, total_preds, \
            total_correct = self._dataset.eval_preds(logits, seq_ids)
        return correct_labels, incorrect_labels, correct_preds, total_preds, \
            total_correct

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None
