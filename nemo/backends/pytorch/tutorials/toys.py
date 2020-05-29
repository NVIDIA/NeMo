# Copyright (c) 2019 NVIDIA Corporation
"""This file contains a collection of overly simplistic NeuralModules"""
import numpy as np
import torch as t
import torch.nn as nn
import torch.utils.data as t_utils

from nemo.backends.pytorch.nm import DataLayerNM, LossNM, TrainableNM
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs


class TaylorNet(TrainableNM):  # Note inheritance from TrainableNM
    """Module which learns Taylor's coefficients."""

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.

        Returns:
          A (dict) of module's input ports names to NeuralTypes mapping
        """
        return {"x": NeuralType(('B', 'D'), ChannelType())}

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        Returns:
          A (dict) of module's output ports names to NeuralTypes mapping
        """
        return {"y_pred": NeuralType(('B', 'D'), ChannelType())}

    def __init__(self, dim, name=None):
        """
            Creates TaylorNet object.

            Args:
                dim: Number of dimensions (number of terms in Taylor series).
                name: Name of the module instance
        """
        super().__init__(name=name)

        # And of Neural Modules specific part. Rest is Pytorch code
        self._dim = dim
        self.fc1 = nn.Linear(self._dim, 1)
        t.nn.init.xavier_uniform_(self.fc1.weight)
        self.to(self._device)

    # IMPORTANT: input arguments to forward must match input input ports' names
    def forward(self, x):
        lst = []
        for pw in range(self._dim):
            lst.append(x ** pw)
        nx = t.cat(lst, dim=-1)
        return self.fc1(nx)


class RealFunctionDataLayer(DataLayerNM):
    """
    Data layer that yields (x, f(x)) data and label pairs.

    Args:
        n: Total number of samples
        batch_size: Size of each batch per iteration
        f_name: Name of the function that will be applied to each x value to get labels.
           Must take a torch tensor as input, and output a torch tensor of
           the same shape. Defaults to torch.sin().
           [Options: sin | cos]
        x_lo: Lower bound of domain to sample
        x_hi: Upper bound of domain to sample
    """

    def __len__(self):
        return self._n

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports
        """
        return {
            "x": NeuralType(('B', 'D'), ChannelType()),
            "y": NeuralType(('B', 'D'), LabelsType()),
        }

    def __init__(self, batch_size, f_name="sin", n=1000, x_lo=-4, x_hi=4, name=None):
        """
            Creates a datalayer returning (x-y) pairs, with n points from a given range.

            Args:
                batch_size: size of batch
                f_name: name of function ["sin" | "cos"]
                n: number of points
                x_lo: lower boundary along x axis
                x_hi: higher boundary along x axis
                name: Name of the module instance
        """
        super().__init__(name=name)

        # Dicionary with handled functions.
        handled_funcs = {"sin": t.sin, "cos": t.cos}

        # Get function - raises an exception if function is not handled
        func = handled_funcs[f_name]

        self._n = n
        self._batch_size = batch_size

        x_data = t.tensor(np.random.uniform(low=x_lo, high=x_hi, size=self._n)).unsqueeze(-1)
        y_data = func(x_data)
        self._dataset = t_utils.TensorDataset(x_data.float(), y_data.float())
        self._data_iterator = t_utils.DataLoader(self._dataset, batch_size=self._batch_size,)

    @property
    def data_iterator(self):
        return self._data_iterator

    @property
    def dataset(self):
        return self._dataset


class MSELoss(LossNM):
    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.

        predictions:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag)

        target:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag)
        """
        return {
            "predictions": NeuralType(('B', 'D'), ChannelType()),
            "target": NeuralType(('B', 'D'), LabelsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, name=None):
        super().__init__(name=name)
        self._criterion = nn.MSELoss()

    def _loss_function(self, **kwargs):
        return self._criterion(*(kwargs.values()))


class L1Loss(LossNM):
    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "predictions": NeuralType(('B', 'D'), ChannelType()),
            "target": NeuralType(('B', 'D'), LabelsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, name=None):
        super().__init__(name=name)
        self._criterion = nn.L1Loss()

    def _loss_function(self, **kwargs):
        return self._criterion(*(kwargs.values()))


class CrossEntropyLoss(LossNM):
    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "predictions": NeuralType(('B', 'D'), ChannelType()),
            "labels": NeuralType(tuple('B'), LabelsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, name=None):
        super().__init__(name=name)
        self._criterion = nn.CrossEntropyLoss()

    # You need to implement this function
    def _loss_function(self, **kwargs):
        return self._criterion(*(kwargs.values()))
