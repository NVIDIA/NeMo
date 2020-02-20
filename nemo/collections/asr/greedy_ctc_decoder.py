# Copyright (c) 2019 NVIDIA Corporation
import torch

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs


class GreedyCTCDecoder(TrainableNM):
    """
    Greedy decoder that computes the argmax over a softmax distribution
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        # return {"log_probs": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag),})}
        return {"log_probs": NeuralType(('B', 'T', 'D'), LogprobsType())}

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        # return {"predictions": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)})}
        return {"predictions": NeuralType(('B', 'T'), PredictionsType())}

    def __init__(self):
        super().__init__()

    def forward(self, log_probs):
        with torch.no_grad():
            argmx = log_probs.argmax(dim=-1, keepdim=False)
            return argmx
