# Copyright (c) 2019 NVIDIA Corporation
"""Core PyTorch-base Neural Modules"""
from typing import Iterable, Optional, Mapping, Set, Dict

import torch
import torch.nn as nn

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core import NeuralModule
from nemo.core.neural_types import *


class SimpleCombiner(TrainableNM):
    """Performs simple combination of two NmTensors. For example, it can
    perform x1 + x2.

    Args:
        mode (str): Can be ['add', 'sum', 'max'].
            Defaults to 'add'.

    """

    @staticmethod
    def create_ports():
        input_ports = {"x1": NeuralType({}), "x2": NeuralType({})}

        output_ports = {"combined": None}

        return input_ports, output_ports

    def __init__(self, mode="add", **kwargs):
        TrainableNM.__init__(self, **kwargs)
        self._mode = mode

    def forward(self, x1, x2):
        if self._mode == "add" or self._mode == "sum":
            return x1 + x2
        elif self._mode == "max":
            return torch.max(x1, x2, out=None)
        else:
            raise NotImplementedError(
                "SimpleCombiner does not have {0} mode".format(self._mode)
            )


class ArgMaxSimple(TrainableNM):  # Notice TWO base classes
    """
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "x": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)})
        }
        output_ports = {
            "values": NeuralType({0: AxisType(BatchTag)}),
            "indices": NeuralType({0: AxisType(BatchTag)}),
        }

        return input_ports, output_ports

    def __init__(self, **kwargs):
        TrainableNM.__init__(self, **kwargs)

    # this method is key method you need to overwrite from PyTorch
    # nn.Module's API
    def forward(self, x):
        values, indices = torch.max(x, 1)
        return values, indices


class TableLookUp(NeuralModule):
    """Performs a table lookup. For example, convert class ids to names"""

    def set_weights(self, name2weight: Dict[(str, bool)],
                    name2name_and_transform):
        pass

    def tie_weights_with(self, module, weight_names):
        pass

    def save_to(self, path):
        pass

    def restore_from(self, path):
        pass

    def freeze(self, weights: Set[str] = None):
        pass

    def unfreeze(self, weights: Set[str] = None):
        pass

    @staticmethod
    def create_ports():
        input_ports = {
            "indices": NeuralType(
                {0: AxisType(TimeTag), 1: AxisType(BatchTag)})
        }
        output_ports = {
            "indices": NeuralType(
                {0: AxisType(BatchTag), 1: AxisType(TimeTag)})
        }

        return input_ports, output_ports

    def __init__(self, ids2classes=None, **kwargs):
        NeuralModule.__init__(self, **kwargs)

        if ids2classes is None:
            ids2classes = {}
        self._ids2classes = ids2classes
        # self._input_ports = {"indices": NeuralType({0: AxisType(BatchTag)})}

    def __call__(self, force_pt=False, *input, **kwargs):
        pt_call = len(input) > 0 or force_pt
        if pt_call:
            # [inds] = kwargs.values()
            # np_inds = inds.detach().cpu().numpy().reshape(-1)
            # result = [self._ids2classes[i] for i in np_inds]
            # #result = list(map(lambda x: self._ids2classes[x], np_inds))
            # return result
            inds = kwargs["indices"]
            np_inds = inds.detach().transpose_(1, 0).cpu().numpy().tolist()
            result = []
            for lst in np_inds:
                sublst = []
                for tid in lst:
                    if tid != 1:
                        sublst.append(tid)
                    else:
                        break
                result.append(
                    list(map(lambda x: self._ids2classes[x], sublst)))
            return [result]
        else:
            return NeuralModule.__call__(self, **kwargs)

    def parameters(self):
        return None

    def get_weights(self) -> Iterable[Optional[Mapping]]:
        return None


class TableLookUp2(NeuralModule):
    """Performs a table lookup. For example, convert class ids to names"""

    def set_weights(self, name2weight: Dict[(str, bool)],
                    name2name_and_transform):
        pass

    def tie_weights_with(self, module, weight_names):
        pass

    def save_to(self, path):
        pass

    def restore_from(self, path):
        pass

    def freeze(self, weights: Set[str] = None):
        pass

    def unfreeze(self, weights: Set[str] = None):
        pass

    @staticmethod
    def create_ports():
        input_ports = {
            "indices": NeuralType(
                {0: AxisType(BatchTag), 1: AxisType(TimeTag)})
        }
        output_ports = {"classes": None}

        return input_ports, output_ports

    def __init__(self, detokenizer=None, **kwargs):
        NeuralModule.__init__(self, **kwargs)
        # self._sp_decoder = self.local_parameters.get("sp_decoder", {})
        self._detokenizer = detokenizer

    def __call__(self, force_pt=False, *input, **kwargs):
        pt_call = len(input) > 0 or force_pt
        if pt_call:
            # [inds] = kwargs.values()
            inds = kwargs["indices"]
            np_inds = inds.detach().cpu().numpy().tolist()
            result = []
            for lst in np_inds:
                sublst = []
                for tid in lst:
                    if tid != 1:
                        sublst.append(tid)
                    else:
                        break
                result.append(self._detokenizer(sublst))
            return result
        else:
            return NeuralModule.__call__(self, **kwargs)

    def parameters(self):
        return None

    def get_weights(self) -> Iterable[Optional[Mapping]]:
        return None


class SequenceEmbedding(TrainableNM):
    @staticmethod
    def create_ports():
        input_ports = {
            "input_seq": NeuralType(
                {0: AxisType(TimeTag), 1: AxisType(BatchTag)})
        }
        output_ports = {
            "outputs": NeuralType(
                {0: AxisType(TimeTag), 1: AxisType(BatchTag),
                 2: AxisType(ChannelTag)}
            )
        }

        return input_ports, output_ports

    def __init__(self, *, voc_size, hidden_size, dropout=0.0, **kwargs):
        TrainableNM.__init__(self, **kwargs)

        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.embedding = nn.Embedding(self.voc_size, self.hidden_size)
        if self.dropout != 0.0:
            self.embedding_dropout = nn.Dropout(self.dropout)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        if self.dropout != 0.0:
            embedded = self.embedding_dropout(embedded)
        return embedded


class SequenceProjection(TrainableNM):
    @staticmethod
    def create_ports():
        input_ports = {"input_seq": NeuralType({})}
        output_ports = {"outputs": None}

        return input_ports, output_ports

    def __init__(self, *, from_dim, to_dim, dropout=0.0, **kwargs):
        TrainableNM.__init__(self, **kwargs)

        self.from_dim = from_dim
        self.to_dim = to_dim
        self.dropout = dropout
        self.projection = nn.Linear(self.from_dim, self.to_dim, bias=False)
        if self.dropout != 0.0:
            self.embedding_dropout = nn.Dropout(self.dropout)

    def forward(self, input_seq):
        p = self.projection(input_seq)
        if self.dropout != 0.0:
            p = self.dropout(p)
        return p
