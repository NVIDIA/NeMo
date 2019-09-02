# Copyright (c) 2019 NVIDIA Corporation
"""This file contains a collection of overly simplistic NeuralModules"""
import numpy as np
import torch as t
import torch.nn as nn
import torch.utils.data as t_utils

from ..nm import TrainableNM, DataLayerNM, LossNM
from ....core import NeuralModule, DeviceType
from ....core.neural_types import *


class TaylorNet(TrainableNM):  # Note inheritance from TrainableNM
    """Module which learns Taylor's coefficients."""
    @staticmethod
    def create_ports():
        input_ports = {
            "x": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)})
        }
        output_ports = {
            "y_pred": NeuralType(
                {0: AxisType(BatchTag), 1: AxisType(ChannelTag)})
        }

        return input_ports, output_ports

    def __init__(self, *, dim, **kwargs):
        # Part specific for Neural Modules API:
        #   (1) call base constructor
        #   (2) define input and output ports
        TrainableNM.__init__(self, **kwargs)

        # And of Neural Modules specific part. Rest is Pytorch code
        self._dim = dim
        self.fc1 = nn.Linear(self._dim, 1)
        t.nn.init.xavier_uniform_(self.fc1.weight)
        self._device = t.device(
            "cuda" if self.placement == DeviceType.GPU else "cpu")
        self.to(self._device)

    # IMPORTANT: input arguments to forward must match input input ports' names
    def forward(self, x):
        lst = []
        for pw in range(self._dim):
            lst.append(x ** pw)
        nx = t.cat(lst, dim=-1)
        return self.fc1(nx)


class TaylorNetO(TrainableNM):  # Note inheritance from TrainableNM
    """Module which learns Taylor's coefficients."""
    @staticmethod
    def create_ports():
        input_ports = {
            "x": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
            "o": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
        }

        output_ports = {
            "y_pred": NeuralType(
                {0: AxisType(BatchTag), 1: AxisType(ChannelTag)}, optional=True
            )
        }

        return input_ports, output_ports

    def __init__(self, *, dim, **kwargs):
        # Part specific for Neural Modules API:
        #   (1) call base constructor
        #   (2) define input and output ports
        TrainableNM.__init__(self, **kwargs)

        # And of Neural Modules specific part. Rest is Pytorch code
        self._dim = dim
        self.fc1 = nn.Linear(self._dim, 1)
        t.nn.init.xavier_uniform_(self.fc1.weight)
        self._device = t.device(
            "cuda" if self.placement == DeviceType.GPU else "cpu")
        self.to(self._device)

    # IMPORTANT: input arguments to forward must match input input ports' names
    # If port is Optional, the default value should be None
    def forward(self, x, o=None):
        lst = []
        if o is None:
            print("O is None")
        else:
            print("O is not None")
        for pw in range(self._dim):
            lst.append(x ** pw)
        nx = t.cat(lst, dim=-1)
        return self.fc1(nx)


class RealFunctionDataLayer(DataLayerNM):
    def __len__(self):
        return self._n

    @staticmethod
    def create_ports():
        input_ports = {}

        output_ports = {
            "x": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
            "y": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
        }

        return input_ports, output_ports

    def __init__(self, *, n, batch_size, **kwargs):
        DataLayerNM.__init__(self, **kwargs)

        self._n = n
        self._batch_size = batch_size
        self._device = t.device(
            "cuda" if self.placement == DeviceType.GPU else "cpu")

        x_data = (
            t.tensor(np.random.uniform(low=-4, high=4, size=self._n))
            .unsqueeze(-1).to(self._device)
        )
        y_data = t.sin(x_data)

        self._data_iterator = t_utils.DataLoader(
            t_utils.TensorDataset(x_data.float(), y_data.float()),
            batch_size=self._batch_size,
        )

    @property
    def data_iterator(self):
        return self._data_iterator

    @property
    def dataset(self):
        return None


class MSELoss(LossNM):
    @staticmethod
    def create_ports():
        input_ports = {
            "predictions": NeuralType(
                {0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
            "target": NeuralType(
                {0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
        }
        output_ports = {"loss": NeuralType(None)}
        return input_ports, output_ports

    def __init__(self, **kwargs):
        LossNM.__init__(self, **kwargs)
        self._criterion = nn.MSELoss()

    def _loss_function(self, **kwargs):
        return self._criterion(*(kwargs.values()))


class L1Loss(LossNM):
    @staticmethod
    def create_ports():
        input_ports = {
            "predictions": NeuralType(
                {0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
            "target": NeuralType(
                {0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
        }
        output_ports = {"loss": NeuralType(None)}
        return input_ports, output_ports

    def __init__(self, **kwargs):
        LossNM.__init__(self, **kwargs)
        self._criterion = nn.L1Loss()

    def _loss_function(self, **kwargs):
        return self._criterion(*(kwargs.values()))


class CrossEntropyLoss(LossNM):
    @staticmethod
    def create_ports():
        input_ports = {
            "predictions": NeuralType(
                {0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
            "labels": NeuralType({0: AxisType(BatchTag)}),
        }
        output_ports = {"loss": NeuralType(None)}
        return input_ports, output_ports

    def __init__(self, **kwargs):
        # Neural Module API specific
        NeuralModule.__init__(self, **kwargs)
        # End of Neural Module API specific
        self._criterion = nn.CrossEntropyLoss()

    # You need to implement this function
    def _loss_function(self, **kwargs):
        return self._criterion(*(kwargs.values()))


class DopeDualLoss(LossNM):
    """
    The dual loss function that DOPE uses
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "belief_predictions": NeuralType(
                {0: AxisType(BatchTag), 1: AxisType(ChannelTag)}
            ),
            "belief_labels": NeuralType({0: AxisType(BatchTag)}),
            "affinity_predictions": NeuralType(
                {0: AxisType(BatchTag), 1: AxisType(ChannelTag)}
            ),
            "affinity_labels": NeuralType({0: AxisType(BatchTag)}),
        }

        output_ports = {"loss": NeuralType(None)}
        return input_ports, output_ports

    def __init__(self, **kwargs):
        # Neural Module API specific
        NeuralModule.__init__(self, **kwargs)

    # You need to implement this function
    def _loss_function(self, **kwargs):
        loss = 0.0

        # Belief maps loss
        # output, each belief map layers.
        for l in kwargs["belief_predictions"]:
            loss_tmp = (
                    (l - kwargs["belief_labels"]) * (
                     l - kwargs["belief_labels"])
            ).mean()
            loss += loss_tmp

        # Affinities loss
        # output, each belief map layers.
        for l in kwargs["affinity_predictions"]:
            loss_tmp = (
                    (l - kwargs["affinity_labels"]) * (
                     l - kwargs["affinity_labels"])
            ).mean()
            loss += loss_tmp

        return loss
