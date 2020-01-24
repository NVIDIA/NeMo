# Copyright (c) 2019 NVIDIA Corporation
"""Core PyTorch-base Neural Modules"""
__all__ = [
    'SimpleCombiner',
    'ArgMaxSimple',
    'TableLookUp',
    'TableLookUp2',
    'SequenceEmbedding',
    'SequenceProjection',
    'ZerosLikeNM',
]

from typing import Dict, Iterable, Mapping, Optional, Set

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

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        x1:
            Empty?!?

        x2:
            Empty?!?
        """
        return {"x1": NeuralType({}), "x2": NeuralType({})}

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        combined:
            None
        """
        return {"combined": None}

    def __init__(self, mode="add", **kwargs):
        TrainableNM.__init__(self, **kwargs)
        self._mode = mode

    def forward(self, x1, x2):
        if self._mode == "add" or self._mode == "sum":
            return x1 + x2
        elif self._mode == "max":
            return torch.max(x1, x2, out=None)
        else:
            raise NotImplementedError("SimpleCombiner does not have {0} mode".format(self._mode))


class ArgMaxSimple(TrainableNM):  # Notice TWO base classes
    """
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        x:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag)
        """
        return {"x": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)})}

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        values:
            0: AxisType(BatchTag)

        indices:
            0: AxisType(BatchTag)
        """
        return {
            "values": NeuralType({0: AxisType(BatchTag)}),
            "indices": NeuralType({0: AxisType(BatchTag)}),
        }

    def __init__(self, **kwargs):
        TrainableNM.__init__(self, **kwargs)

    # this method is key method you need to overwrite from PyTorch
    # nn.Module's API
    def forward(self, x):
        values, indices = torch.max(x, 1)
        return values, indices


class TableLookUp(NeuralModule):
    """Performs a table lookup. For example, convert class ids to names"""

    def __init__(self, ids2classes=None, **kwargs):
        NeuralModule.__init__(self, **kwargs)

        if ids2classes is None:
            ids2classes = {}
        self._ids2classes = ids2classes

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        indices:
            0: AxisType(TimeTag)

            1: AxisType(BatchTag)
        """
        return {"indices": NeuralType({0: AxisType(TimeTag), 1: AxisType(BatchTag)})}

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

            indices:
                0: AxisType(BatchTag)
                1: AxisType(TimeTag)
        """
        return {"indices": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)})}

    def set_weights(self, name2weight: Dict[(str, bool)], name2name_and_transform):
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
                result.append(list(map(lambda x: self._ids2classes[x], sublst)))
            return [result]
        else:
            return NeuralModule.__call__(self, **kwargs)

    def parameters(self):
        return None

    def get_weights(self) -> Iterable[Optional[Mapping]]:
        return None


class TableLookUp2(NeuralModule):
    """Performs a table lookup. For example, convert class ids to names"""

    def set_weights(self, name2weight: Dict[(str, bool)], name2name_and_transform):
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

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        """
        return {}

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        classes:
            None
        """
        return {"classes": None}

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
    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        input_seq:
            0: AxisType(TimeTag)

            1: AxisType(BatchTag)
        """
        return {"input_seq": NeuralType({0: AxisType(TimeTag), 1: AxisType(BatchTag)})}

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        outputs:
            0: AxisType(TimeTag)

            1: AxisType(BatchTag)

            2: AxisType(ChannelTag)
        """
        return {"outputs": NeuralType({0: AxisType(TimeTag), 1: AxisType(BatchTag), 2: AxisType(ChannelTag),})}

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
    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        input_seq:
            Empty Type?!?
        """
        return {"input_seq": NeuralType({})}

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        outputs:
            None
        """
        return {"outputs": None}

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


class ZerosLikeNM(TrainableNM):
    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        input_type_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {"input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag),})}

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_type_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {"input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag),})}

    def __init__(self, **kwargs):
        TrainableNM.__init__(self, **kwargs)

    def forward(self, input_type_ids):
        return torch.zeros_like(input_type_ids).long()
