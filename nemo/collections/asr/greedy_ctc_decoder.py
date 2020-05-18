# Copyright (c) 2019 NVIDIA Corporation
import torch

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs


class GreedyCTCDecoder(NonTrainableNM):
    """
    Greedy decoder that computes the argmax over a softmax distribution
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns:
            Definitions of module input ports.
        """
        return {"log_probs": NeuralType(('B', 'T', 'D'), LogprobsType())}

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns:
            Definitions of module output ports.
        """
        return {"predictions": NeuralType(('B', 'T'), PredictionsType())}

    def __init__(self):
        super().__init__()

    def forward(self, log_probs):
        argmx = log_probs.argmax(dim=-1, keepdim=False)
        return argmx
