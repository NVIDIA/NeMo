# Copyright (c) 2019 NVIDIA Corporation
import torch

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import *


class GreedyCTCDecoder(TrainableNM):
    """
    Greedy decoder that computes the argmax over a softmax distribution
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "log_probs": NeuralType({0: AxisType(BatchTag),
                                     1: AxisType(TimeTag),
                                     2: AxisType(ChannelTag)})
        }

        output_ports = {
            "predictions": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }
        return input_ports, output_ports

    def __init__(self, **kwargs):
        TrainableNM.__init__(self, **kwargs)

    def forward(self, log_probs):
        with torch.no_grad():
            argmx = log_probs.argmax(dim=-1, keepdim=False)
            return argmx
