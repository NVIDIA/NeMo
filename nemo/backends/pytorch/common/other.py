# Copyright (c) 2019 NVIDIA Corporation
"""Core PyTorch-base Neural Modules"""
__all__ = [
    'SequenceEmbedding',
    'ZerosLikeNM',
]

from typing import Dict, Iterable, Mapping, Optional, Set

import torch
import torch.nn as nn

from nemo.backends.pytorch.nm import NonTrainableNM, TrainableNM
from nemo.core.neural_types import *


class SequenceEmbedding(TrainableNM):
    @property
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        # return {"input_seq": NeuralType({0: AxisType(TimeTag), 1: AxisType(BatchTag)})}
        return {"input_seq": NeuralType(('B', 'T'))}

    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        # return {"outputs": NeuralType({0: AxisType(TimeTag), 1: AxisType(BatchTag), 2: AxisType(ChannelTag),})}
        return {"outputs": NeuralType(('B', 'T', 'C'), ChannelType())}

    def __init__(self, voc_size, hidden_size, dropout=0.0):
        super().__init__()

        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.embedding = nn.Embedding(self.voc_size, self.hidden_size)
        if self.dropout != 0.0:
            self.embedding_dropout = nn.Dropout(self.dropout)
        self.to(self._device)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        if self.dropout != 0.0:
            embedded = self.embedding_dropout(embedded)
        return embedded


class ZerosLikeNM(NonTrainableNM):
    @property
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        # return {"input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag),})}
        return {"input_type_ids": NeuralType(('B', 'T'), VoidType())}

    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        # return {"input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag),})}
        return {"input_type_ids": NeuralType(('B', 'T'), ChannelType())}

    def __init__(self):
        super().__init__()

    def forward(self, input_type_ids):
        return torch.zeros_like(input_type_ids).long()
