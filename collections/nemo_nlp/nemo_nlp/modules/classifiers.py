import torch
import torch.nn as nn

from nemo.backends.pytorch.common import MultiLayerPerceptron
from nemo.backends.pytorch.nm import TrainableNM, LossNM
from nemo.core.neural_types import *

from ..transformer.utils import transformer_weights_init


class SentenceClassificationLogSoftmaxNM(TrainableNM):
    @staticmethod
    def create_ports():
        input_ports = {
            "hidden_states":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
        }

        output_ports = {
            "log_probs":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
        }
        return input_ports, output_ports

    def __init__(self, *, d_model, num_classes, **kwargs):
        TrainableNM.__init__(self, **kwargs)

        self.log_softmax = ClassificationLogSoftmax(
            hidden_size=d_model,
            num_classes=num_classes
        )

        self.log_softmax.apply(transformer_weights_init)
        self.log_softmax.to(self._device)

    def forward(self, hidden_states):
        log_probs = self.log_softmax(hidden_states)
        return log_probs


class TokenClassifier(TrainableNM):
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

    def __init__(self, hidden_size, num_classes, dropout, **kwargs):
        TrainableNM.__init__(self, **kwargs)

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        self.apply(
            lambda module: transformer_weights_init(module, xavier=False))
        self.to(self._device)

    def forward(self, hidden_states):

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.relu(hidden_states)
        logits = self.classifier(hidden_states)
        return logits


class SequenceClassifier(TrainableNM):
    """
    The softmax classifier for sequence classifier task.
    Some examples of this task would be sentiment analysis,
    sentence classification, etc.

    Args:
        hidden_size (int): the size of the hidden state for the dense layer
        num_classes (int): number of label types
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
                 out_activation='log_softmax',
                 dropout=1.0,
                 **kwargs):
        TrainableNM.__init__(self, **kwargs)
        self.mlp = MultiLayerPerceptron(hidden_size,
                                        num_layers,
                                        num_classes,
                                        activation,
                                        out_activation)
        # self.hidden_size = hidden_size
        # self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        # self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        # self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.apply(
            lambda module: transformer_weights_init(module, xavier=False))
        self.to(self._device)

    def forward(self, hidden_states, idx_conditioned_on=0):
        hidden_states = self.dropout(hidden_states)
        logits = self.mlp(hidden_states[:, idx_conditioned_on])
        # hidden_states = self.dense(hidden_states[:, idx_conditioned_on])
        # hidden_states = torch.relu(hidden_states)
        # hidden_states = self.classifier(hidden_states)
        # logits = torch.log_softmax(hidden_states)

        return logits


class JointIntentSlotClassifier(TrainableNM):
    """
    The softmax classifier for the joint intent classification and slot
    filling task.

    It consists of a dense layer + relu + softmax for predicting the slots
    and similar for predicting the intents.

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
                 dropout,
                 **kwargs):
        TrainableNM.__init__(self, **kwargs)
        self.hidden_size = hidden_size
        self.num_intents = num_intents
        self.num_slots = num_slots
        self.dropout = nn.Dropout(dropout)
        self.intent_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.intent_classifier = nn.Linear(self.hidden_size, self.num_intents)
        self.slot_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.slot_classifier = nn.Linear(self.hidden_size, self.num_slots)
        self.apply(
            lambda module: transformer_weights_init(module, xavier=False))
        self.to(self._device)

    def forward(self, hidden_states):
        """ hidden_states: the outputs from the previous layers
        """
        hidden_states = self.dropout(hidden_states)

        intent_states = self.intent_dense(hidden_states[:, 0])
        intent_states = torch.relu(intent_states)
        intent_logits = self.intent_classifier(intent_states)

        # slot_states = self.slot_dense(hidden_states[1:, :])
        slot_states = self.slot_dense(hidden_states)
        slot_states = torch.relu(slot_states)
        slot_logits = self.slot_classifier(slot_states)

        return intent_logits, slot_logits
