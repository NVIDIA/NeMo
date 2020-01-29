# Copyright (c) 2019 NVIDIA Corporation
import torch

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import AxisType, BatchTag, ChannelTag, NeuralType, TimeTag


class GreedyCTCDecoder(TrainableNM):
    """
    Greedy decoder that computes the argmax over a softmax distribution
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        log_probs:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)
        """
        return {"log_probs": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag)})}

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        predictions:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {"predictions": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)})}

    def __init__(self, **kwargs):
        TrainableNM.__init__(self, **kwargs)

    def forward(self, log_probs):
        with torch.no_grad():
            argmx = log_probs.argmax(dim=-1, keepdim=False)
            return argmx
