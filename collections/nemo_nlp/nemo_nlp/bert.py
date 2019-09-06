# Copyright (c) 2019 NVIDIA Corporation
"""
This package contains BERT Neural Module
"""
import torch
import torch.nn as nn

from nemo.backends.pytorch.nm import TrainableNM, LossNM
from nemo.core.neural_types import *
from .transformer import ClassificationLogSoftmax
from .transformer import SmoothedCrossEntropyLoss
from .transformer import SequenceClassificationLoss
from .transformer.utils import transformer_weights_init


class MaskedLanguageModelingLossNM(TrainableNM):
    @staticmethod
    def create_ports():
        input_ports = {
            "log_probs":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            "output_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "output_mask":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
        }

        output_ports = {"loss": NeuralType(None)}
        return input_ports, output_ports

    def __init__(self, **kwargs):
        TrainableNM.__init__(self, **kwargs)
        label_smoothing = self.local_parameters.get("label_smoothing", 0.0)
        self._loss_fn = SmoothedCrossEntropyLoss(label_smoothing)

    def forward(self, log_probs, output_ids, output_mask):
        loss = self._loss_fn(log_probs, output_ids, output_mask)
        return loss


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


class NextSentencePredictionLossNM(TrainableNM):
    @staticmethod
    def create_ports():
        input_ports = {
            "log_probs":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            "labels":
            NeuralType({0: AxisType(BatchTag)}),
        }

        output_ports = {"loss": NeuralType(None)}
        return input_ports, output_ports

    def __init__(self, **kwargs):
        TrainableNM.__init__(self, **kwargs)
        self._loss_fn = SequenceClassificationLoss()

    def forward(self, log_probs, labels):
        loss = self._loss_fn(log_probs, labels)
        return loss


class LossAggregatorNM(LossNM):
    @staticmethod
    def create_ports(num_losses=2):
        input_ports = {}
        for i in range(num_losses):
            input_ports["loss_" + str(i + 1)] = NeuralType(None)

        output_ports = {"loss": NeuralType(None)}
        return input_ports, output_ports

    def __init__(self, *, num_inputs, **kwargs):
        kwargs["create_port_args"] = {"num_losses": num_inputs}
        LossNM.__init__(self, **kwargs)

    def _loss_function(self, **kwargs):
        values = [kwargs[x] for x in sorted(kwargs.keys())]
        loss = values[0]
        for loss_i in values[1:]:
            loss = loss.add(loss_i.item())
        return loss


class QuestionAnsweringPredictionLoss(TrainableNM):
    @staticmethod
    def create_ports():
        input_ports = {
            "hidden_states":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            "start_positions":
            NeuralType({0: AxisType(BatchTag)}),
            "end_positions":
            NeuralType({0: AxisType(BatchTag)})
        }

        output_ports = {
            "loss":
            NeuralType(None),
            "start_logits":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "end_logits":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }
        return input_ports, output_ports

    def __init__(self, **kwargs):
        TrainableNM.__init__(self, **kwargs)

        self.hidden_size = self.local_parameters["d_model"]

        self.qa_outputs = nn.Linear(self.hidden_size, 2)
        self.qa_outputs.apply(transformer_weights_init)
        self.qa_outputs.to(self._device)

    def forward(self, hidden_states, start_positions, end_positions):

        logits = self.qa_outputs(hidden_states)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)

        # sometimes the start/end positions are outside our model inputs,
        # we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2

        return total_loss, start_logits, end_logits


class TokenClassificationLoss(TrainableNM):
    @staticmethod
    def create_ports():
        input_ports = {
            "hidden_states":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            "labels":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_mask":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }

        output_ports = {
            "loss":
            NeuralType(None),
            "logits":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            })
        }
        return input_ports, output_ports

    def __init__(self, **kwargs):
        TrainableNM.__init__(self, **kwargs)

        self.hidden_size = self.local_parameters["d_model"]
        self.num_labels = self.local_parameters["num_labels"]
        self.dropout = nn.Dropout(self.local_parameters["dropout"])
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.apply(
            lambda module: transformer_weights_init(module, xavier=False))
        self.to(self._device)

    def forward(self, hidden_states, labels, input_mask):

        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss_fct = nn.CrossEntropyLoss()

        active_loss = input_mask.view(-1) > 0.5
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]

        loss = loss_fct(active_logits, active_labels)

        return loss, logits


class ZerosLikeNM(TrainableNM):
    @staticmethod
    def create_ports():
        input_ports = {
            "input_type_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
            })
        }

        output_ports = {
            "input_type_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
            })
        }
        return input_ports, output_ports

    def __init__(self, **kwargs):
        TrainableNM.__init__(self, **kwargs)

    def forward(self, input_type_ids):
        return torch.zeros_like(input_type_ids).long()


class SequenceClassifier(TrainableNM):
    """
    Loss function for the joint intent classification and slot
    filling task.

    The loss is a joint loss of both tasks, aim to maximize:
    p(y^i | x)P(y^s1, y^s2, ..., y^sn | x)

    with y^i being the predicted intent and y^s1, y^s2, ..., y^sn
    are the predicted slots corresponding to x1, x2, ..., xn.

    Args:
        hidden_states: output of the hidden layers
        intents: ground truth intents,
        slots: ground truth slots.
        input_mask: to differentiate from original tokens and paddings
        intent_loss_weight: the loss is the sum of:
            intent_loss_weight * intent_loss +
            (1 - intent_loss_weight) * slot_loss

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
        hidden_states = self.dense(hidden_states[:, 0])
        hidden_states = torch.relu(hidden_states)
        logits = self.classifier(hidden_states)

        return logits


class JointIntentSlotClassifier(TrainableNM):
    """
    Loss function for the joint intent classification and slot
    filling task.

    The loss is a joint loss of both tasks, aim to maximize:
    p(y^i | x)P(y^s1, y^s2, ..., y^sn | x)

    with y^i being the predicted intent and y^s1, y^s2, ..., y^sn
    are the predicted slots corresponding to x1, x2, ..., xn.

    Args:
        hidden_states: output of the hidden layers
        intents: ground truth intents,
        slots: ground truth slots.
        input_mask: to differentiate from original tokens and paddings
        intent_loss_weight: the loss is the sum of:
            intent_loss_weight * intent_loss +
            (1 - intent_loss_weight) * slot_loss

    """
    @staticmethod
    def create_ports():
        input_ports = {
            "hidden_states": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            "input_mask": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
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
        hidden_states = self.dropout(hidden_states)

        intent_states = self.intent_dense(hidden_states[:, 0])
        intent_states = torch.relu(intent_states)
        intent_logits = self.intent_classifier(intent_states)

        # slot_states = self.slot_dense(hidden_states[1:, :])
        slot_states = self.slot_dense(hidden_states)
        slot_states = torch.relu(slot_states)
        slot_logits = self.slot_classifier(slot_states)

        return intent_logits, slot_logits


class JointIntentSlotLoss(LossNM):
    """
    Loss function for the joint intent classification and slot
    filling task.

    The loss is a joint loss of both tasks, aim to maximize:
    p(y^i | x)P(y^s1, y^s2, ..., y^sn | x)

    with y^i being the predicted intent and y^s1, y^s2, ..., y^sn
    are the predicted slots corresponding to x1, x2, ..., xn.

    Args:
        hidden_states: output of the hidden layers
        intents: ground truth intents,
        slots: ground truth slots.
        input_mask: to differentiate from original tokens and paddings
        intent_loss_weight: the loss is the sum of:
            intent_loss_weight * intent_loss +
            (1 - intent_loss_weight) * slot_loss

    """
    @staticmethod
    def create_ports():
        input_ports = {
            "intent_logits": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(ChannelTag)
            }),
            "slot_logits": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            "input_mask": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "intents": NeuralType({
                0: AxisType(BatchTag),
            }),
            "slots":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            # "intent_loss_weight": NeuralType(None)
        }

        output_ports = {
            "loss": NeuralType(None),
        }
        return input_ports, output_ports

    def __init__(self, num_slots, **kwargs):
        LossNM.__init__(self, **kwargs)
        self.num_slots = num_slots
        self._criterion = nn.CrossEntropyLoss()

    def _loss_function(self,
                       intent_logits,
                       slot_logits,
                       input_mask,
                       intents,
                       slots,
                       intent_loss_weight=0.6):
        intent_loss = self._criterion(intent_logits, intents)

        active_loss = input_mask.view(-1) > 0.5
        active_logits = slot_logits.view(-1, self.num_slots)[active_loss]
        active_labels = slots.view(-1)[active_loss]

        slot_loss = self._criterion(active_logits, active_labels)
        loss = intent_loss * intent_loss_weight + \
            slot_loss * (1 - intent_loss_weight)

        return loss
