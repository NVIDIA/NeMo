# Copyright (c) 2019 NVIDIA Corporation

"""This file contains a collection of LPR specific trainable models"""
from nemo.core import NeuralModule, DeviceType
from nemo.core.neural_types import *
from nemo.backends.pytorch.nm import TrainableNM, DataLayerNM, LossNM

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as t_utils
import numpy as np
import torch as t


class SimpleCNNClassifier(TrainableNM):  # Note inheritance from TrainableNM
    """
    Module which learns Taylor's coefficients.
    From https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    @staticmethod
    def create_ports(input_size=(32, 32)):
        input_ports = {
            "image":  NeuralType({0: AxisType(BatchTag),
                                  1: AxisType(ChannelTag),
                                  2: AxisType(HeightTag, input_size[1]),
                                  3: AxisType(WidthTag, input_size[0])})}
        output_ports = {"output": NeuralType({0: AxisType(BatchTag),
                                              1: AxisType(ChannelTag)})}
        return input_ports, output_ports

    def __init__(self, **kwargs):
        # Part specific for Neural Modules API:
        #   (1) call base constructor
        #   (2) define input and output ports
        self._input_size = (32, 32)
        create_port_args = {"input_size": self._input_size}
        TrainableNM.__init__(self, create_port_args=create_port_args, **kwargs)

        # And of Neural Modules specific part. Rest is Pytorch code
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.cuda()

    # IMPORTANT: input arguments to forward must match input input ports' names
    def forward(self, image):
        x = self.pool(F.relu(self.conv1(image)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
