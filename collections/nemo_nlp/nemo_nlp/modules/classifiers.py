__all__ = ['TokenClassifier',
           'BertTokenClassifier',
           'SequenceClassifier',
           'JointIntentSlotClassifier',
           'SequenceRegression']

import torch.nn as nn

from nemo.backends.pytorch.common import MultiLayerPerceptron
from nemo.backends.pytorch.nm import TrainableNM, LossNM
from nemo.core.neural_types import *
from nemo_nlp.transformer.utils import gelu

from ..transformer.utils import transformer_weights_init


ACT2FN = {"gelu": gelu, "relu": nn.functional.relu}

class BertTokenClassifier(TrainableNM):
    """
    Neural module which consists of MLP followed by softmax classifier for each
    token in the sequence.

    Args:
        hidden_size (int): hidden size (d_model) of the Transformer
        num_classes (int): number of classes in softmax classifier, e.g. size
            of the vocabulary in language modeling objective
        num_layers (int): number of layers in classifier MLP
        activation (str): activation function applied in classifier MLP layers
        log_softmax (bool): whether to apply log_softmax to MLP output
        dropout (float): dropout ratio applied to MLP
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "hidden_states": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            })
        }

        output_ports = {
            "logits": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            })
        }
        return input_ports, output_ports

    def __init__(self,
                 hidden_size,
                 num_classes,
                 activation='relu',
                 log_softmax=True,
                 dropout=0.0,
                 use_transformer_pretrained=True):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = ACT2FN[activation]
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.mlp = MultiLayerPerceptron(hidden_size,
                                        num_classes,
                                        self._device,
                                        num_layers=1,
                                        activation=activation,
                                        log_softmax=log_softmax)
        self.dropout = nn.Dropout(dropout)
        if use_transformer_pretrained:
            self.apply(
                lambda module: transformer_weights_init(module, xavier=False))
        self.to(self._device)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        transform = self.norm(hidden_states)
        logits = self.mlp(transform)
        return logits


class TokenClassifier(TrainableNM):
    """
    Neural module which consists of MLP followed by softmax classifier for each
    token in the sequence.
    Args:
        hidden_size (int): hidden size (d_model) of the Transformer
        num_classes (int): number of classes in softmax classifier, e.g. size
            of the vocabulary in language modeling objective
        num_layers (int): number of layers in classifier MLP
        activation (str): activation function applied in classifier MLP layers
        log_softmax (bool): whether to apply log_softmax to MLP output
        dropout (float): dropout ratio applied to MLP
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "hidden_states": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            })
        }

        output_ports = {
            "logits": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            })
        }
        return input_ports, output_ports

    def __init__(self,
                 hidden_size,
                 num_classes,
                 num_layers=2,
                 activation='relu',
                 log_softmax=True,
                 dropout=0.0,
                 use_transformer_pretrained=True):
        super().__init__()

        self.mlp = MultiLayerPerceptron(hidden_size,
                                        num_classes,
                                        self._device,
                                        num_layers,
                                        activation,
                                        log_softmax)
        self.dropout = nn.Dropout(dropout)
        if use_transformer_pretrained:
            self.apply(
                lambda module: transformer_weights_init(module, xavier=False))
        # self.to(self._device) # sometimes this is necessary

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        logits = self.mlp(hidden_states)
        return logits


class SequenceClassifier(TrainableNM):
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
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "hidden_states": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            })
        }

        output_ports = {
            "logits": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(ChannelTag)
            }),
        }
        return input_ports, output_ports

    def __init__(self,
                 hidden_size,
                 num_classes,
                 num_layers=2,
                 activation='relu',
                 log_softmax=True,
                 dropout=0.0,
                 use_transformer_pretrained=True):
        super().__init__()
        self.mlp = MultiLayerPerceptron(hidden_size,
                                        num_classes,
                                        self._device,
                                        num_layers,
                                        activation,
                                        log_softmax)
        self.dropout = nn.Dropout(dropout)
        if use_transformer_pretrained:
            self.apply(
                lambda module: transformer_weights_init(module, xavier=False))
        # self.to(self._device) # sometimes this is necessary

    def forward(self, hidden_states, idx_conditioned_on=0):
        hidden_states = self.dropout(hidden_states)
        logits = self.mlp(hidden_states[:, idx_conditioned_on])
        return logits


class JointIntentSlotClassifier(TrainableNM):
    """
    The softmax classifier for the joint intent classification and slot
    filling task which  consists of a dense layer + relu + softmax for
    predicting the slots and similar for predicting the intents.

    Args:
        hidden_size (int): the size of the hidden state for the dense layer
        num_intents (int): number of intents
        num_slots (int): number of slots
        dropout (float): dropout to be applied to the layer
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "hidden_states": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            })
        }

        output_ports = {
            "intent_logits": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(ChannelTag)
            }),
            "slot_logits": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            })
        }
        return input_ports, output_ports

    def __init__(self,
                 hidden_size,
                 num_intents,
                 num_slots,
                 dropout=0.0,
                 use_transformer_pretrained=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.slot_mlp = MultiLayerPerceptron(hidden_size,
                                             num_classes=num_slots,
                                             device=self._device,
                                             num_layers=2,
                                             activation='relu',
                                             log_softmax=False)
        self.intent_mlp = MultiLayerPerceptron(hidden_size,
                                               num_classes=num_intents,
                                               device=self._device,
                                               num_layers=2,
                                               activation='relu',
                                               log_softmax=False)
        if use_transformer_pretrained:
            self.apply(
                lambda module: transformer_weights_init(module, xavier=False))
        # self.to(self._device)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        intent_logits = self.intent_mlp(hidden_states[:, 0])
        slot_logits = self.slot_mlp(hidden_states)
        return intent_logits, slot_logits


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
    @staticmethod
    def create_ports():
        input_ports = {
            "hidden_states": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            })
        }

        output_ports = {
            "preds": NeuralType({
                0: AxisType(RegressionTag)
            }),
        }
        return input_ports, output_ports

    def __init__(self,
                 hidden_size,
                 num_layers=2,
                 activation='relu',
                 dropout=0.0,
                 use_transformer_pretrained=True):
        super().__init__()
        self.mlp = MultiLayerPerceptron(hidden_size,
                                        num_classes=1,
                                        device=self._device,
                                        num_layers=num_layers,
                                        activation=activation,
                                        log_softmax=False)
        self.dropout = nn.Dropout(dropout)
        if use_transformer_pretrained:
            self.apply(
                lambda module: transformer_weights_init(module, xavier=False))
        # self.to(self._device) # sometimes this is necessary

    def forward(self, hidden_states, idx_conditioned_on=0):
        hidden_states = self.dropout(hidden_states)
        preds = self.mlp(hidden_states[:, idx_conditioned_on])
        return preds.view(-1)
