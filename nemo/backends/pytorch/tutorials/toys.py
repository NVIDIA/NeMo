# Copyright (c) 2019 NVIDIA Corporation
"""This file contains a collection of overly simplistic NeuralModules"""
import numpy as np
import torch as t
import torch.nn as nn
import torch.utils.data as t_utils

from nemo import logging
from nemo.backends.pytorch.nm import DataLayerNM, LossNM, TrainableNM
from nemo.core import DeviceType, NeuralModule
from nemo.core.neural_types import *


class TaylorNet(TrainableNM):  # Note inheritance from TrainableNM
    """Module which learns Taylor's coefficients."""

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        Returns:
          A (dict) of module's input ports names to NeuralTypes mapping
        """
        return {"x": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)})}

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        Returns:
          A (dict) of module's output ports names to NeuralTypes mapping
        """
        return {"y_pred": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)})}

    def __init__(self, *, dim, **kwargs):
        # Part specific for Neural Modules API:
        #   (1) call base constructor
        #   (2) define input and output ports
        TrainableNM.__init__(self, **kwargs)

        # And of Neural Modules specific part. Rest is Pytorch code
        self._dim = dim
        self.fc1 = nn.Linear(self._dim, 1)
        t.nn.init.xavier_uniform_(self.fc1.weight)
        self._device = t.device("cuda" if self.placement == DeviceType.GPU else "cpu")
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

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        x:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag)

        o:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag)
        """
        return {
            "x": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
            "o": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        y_pred:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag)
        """
        return {"y_pred": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}, optional=True)}

    def __init__(self, *, dim, **kwargs):
        # Part specific for Neural Modules API:
        #   (1) call base constructor
        #   (2) define input and output ports
        TrainableNM.__init__(self, **kwargs)

        # And of Neural Modules specific part. Rest is Pytorch code
        self._dim = dim
        self.fc1 = nn.Linear(self._dim, 1)
        t.nn.init.xavier_uniform_(self.fc1.weight)
        self._device = t.device("cuda" if self.placement == DeviceType.GPU else "cpu")
        self.to(self._device)

    # IMPORTANT: input arguments to forward must match input input ports' names
    # If port is Optional, the default value should be None
    def forward(self, x, o=None):
        lst = []
        if o is None:
            logging.debug("O is None")
        else:
            logging.debug("O is not None")
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
    def output_ports(self):
        """Returns definitions of module output ports

        x:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag)

        y:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag)
        """
        return {
            "x": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
            "y": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
        }

    def __init__(self, *, n, batch_size, f_name="sin", x_lo=-4, x_hi=4):
        DataLayerNM.__init__(self)

        # Dicionary with handled functions.
        handled_funcs = {"sin": t.sin, "cos": t.cos}

        # Get function - raises an exception if function is not handled
        func = handled_funcs[f_name]

        self._n = n
        self._batch_size = batch_size
        self._device = t.device("cuda" if self.placement == DeviceType.GPU else "cpu")

        x_data = t.tensor(np.random.uniform(low=x_lo, high=x_hi, size=self._n)).unsqueeze(-1).to(self._device)
        y_data = func(x_data)

        self._data_iterator = t_utils.DataLoader(
            t_utils.TensorDataset(x_data.float(), y_data.float()), batch_size=self._batch_size,
        )

    @property
    def data_iterator(self):
        return self._data_iterator

    @property
    def dataset(self):
        return None


class MSELoss(LossNM):
    @property
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
            "predictions": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
            "target": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(None)}

    def __init__(self, **kwargs):
        LossNM.__init__(self, **kwargs)
        self._criterion = nn.MSELoss()

    def _loss_function(self, **kwargs):
        return self._criterion(*(kwargs.values()))


class L1Loss(LossNM):
    @property
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
            "predictions": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
            "target": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(None)}

    def __init__(self, **kwargs):
        LossNM.__init__(self, **kwargs)
        self._criterion = nn.L1Loss()

    def _loss_function(self, **kwargs):
        return self._criterion(*(kwargs.values()))


class CrossEntropyLoss(LossNM):
    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        predictions:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag)

        labels:
            0: AxisType(BatchTag)
        """
        return {
            "predictions": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
            "labels": NeuralType({0: AxisType(BatchTag)}),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(None)}

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

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        belief_predictions:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag)

        belief_labels:
            0: AxisType(BatchTag)

        affinity_predictions:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag)

        affinity_labels:
            0: AxisType(BatchTag)
        """
        return {
            "belief_predictions": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
            "belief_labels": NeuralType({0: AxisType(BatchTag)}),
            "affinity_predictions": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
            "affinity_labels": NeuralType({0: AxisType(BatchTag)}),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(None)}

    def __init__(self, **kwargs):
        # Neural Module API specific
        NeuralModule.__init__(self, **kwargs)

    # You need to implement this function
    def _loss_function(self, **kwargs):
        loss = 0.0

        # Belief maps loss
        # output, each belief map layers.
        for l in kwargs["belief_predictions"]:
            loss_tmp = ((l - kwargs["belief_labels"]) * (l - kwargs["belief_labels"])).mean()
            loss += loss_tmp

        # Affinities loss
        # output, each belief map layers.
        for l in kwargs["affinity_predictions"]:
            loss_tmp = ((l - kwargs["affinity_labels"]) * (l - kwargs["affinity_labels"])).mean()
            loss += loss_tmp

        return loss
