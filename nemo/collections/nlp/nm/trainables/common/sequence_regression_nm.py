from torch import nn as nn

from nemo.backends.pytorch import TrainableNM, MultiLayerPerceptron
from nemo.collections.nlp.nm.trainables.common.transformer.transformer_utils import transformer_weights_init
from nemo.core import NeuralType, AxisType, BatchTag, TimeTag, ChannelTag, RegressionTag


class SequenceRegression(TrainableNM):
    """
    Neural module which consists of MLP, generates a single number prediction
    that could be used for a regression task. An example of this task would be
    semantic textual similatity task, for example, STS-B (from GLUE tasks).

    Args:
        hidden_size (int): the size of the hidden state for the dense layer
        num_layers (int): number of layers in classifier MLP
        activation (str): activation function applied in classifier MLP layers
        dropout (float): dropout ratio applied to MLP
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        hidden_states:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)
        """
        return {"hidden_states": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag)})}

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        preds:
            0: AxisType(RegressionTag)
        """
        return {"preds": NeuralType({0: AxisType(RegressionTag)})}

    def __init__(self, hidden_size, num_layers=2, activation='relu', dropout=0.0, use_transformer_pretrained=True):
        super().__init__()
        self.mlp = MultiLayerPerceptron(
            hidden_size,
            num_classes=1,
            device=self._device,
            num_layers=num_layers,
            activation=activation,
            log_softmax=False,
        )
        self.dropout = nn.Dropout(dropout)
        if use_transformer_pretrained:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))
        # self.to(self._device) # sometimes this is necessary

    def forward(self, hidden_states, idx_conditioned_on=0):
        hidden_states = self.dropout(hidden_states)
        preds = self.mlp(hidden_states[:, idx_conditioned_on])
        return preds.view(-1)