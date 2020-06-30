import torch

from nemo.collections.common.parts.multi_layer_perceptron import MultiLayerPerceptron
from nemo.collections.common.parts.transformer_utils import transformer_weights_init
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import ChannelType, LogitsType, NeuralType
from nemo.utils.decorators import experimental


@experimental
class SequenceClassifier(NeuralModule):
    """
    Neural module which consists of MLP followed by softmax classifier for each
    sequence in the batch.
    Args:
        hidden_size (int): hidden size (d_model) of the Transformer
        num_classes (int): number of classes in softmax classifier, e.g. number
            of different sentiments
        num_layers (int): number of layers in classifier MLP
        activation (str): activation function applied in classifier MLP layers
        log_softmax (bool): whether to apply log_softmax to MLP output
        dropout (float): dropout ratio applied to MLP
        use_transformer_pretrained (bool):
            TODO
    """

    @property
    # @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        hidden_states: embedding hidden states
        """
        return {"hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    @property
    # @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        logits: logits before loss
        """
        return {"logits": NeuralType(('B', 'D'), LogitsType())}

    def __init__(
        self,
        hidden_size,
        num_classes,
        num_layers=2,
        activation='relu',
        log_softmax=True,
        dropout=0.0,
        use_transformer_init=True,
    ):
        super().__init__()
        # TODO: what happens to device?
        self.mlp = MultiLayerPerceptron(
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            activation=activation,
            log_softmax=log_softmax,
        )
        self.dropout = torch.nn.Dropout(dropout)
        if use_transformer_init:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))
        # TODO: what happens to device?
        # self.to(self._device) # sometimes this is necessary

    @typecheck()
    def forward(self, hidden_states, idx_conditioned_on=0):
        hidden_states = self.dropout(hidden_states)
        logits = self.mlp(hidden_states[:, idx_conditioned_on])
        return logits

    @classmethod
    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass
