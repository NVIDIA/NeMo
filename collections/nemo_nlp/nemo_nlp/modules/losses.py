__all__ = ['MaskedLanguageModelingLossNM',
           'LossAggregatorNM',
           'TokenClassificationLoss',
           'JointIntentSlotLoss',
           'PaddedSmoothedCrossEntropyLossNM']

from torch import nn

from nemo.backends.pytorch.nm import LossNM
from nemo.core.neural_types import *

from .pytorch_utils import SmoothedCrossEntropyLoss
from ..utils.nlp_utils import mask_padded_tokens


class MaskedLanguageModelingLossNM(LossNM):
    """
    Neural module which implements Masked Language Modeling (MLM) loss.

    Args:
        label_smoothing (float): label smoothing regularization coefficient
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "logits":
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

    def __init__(self, label_smoothing=0.0, **kwargs):
        LossNM.__init__(self, **kwargs)
        self._criterion = SmoothedCrossEntropyLoss(label_smoothing)

    def _loss_function(self, logits, output_ids, output_mask):
        loss = self._criterion(logits, output_ids, output_mask)
        return loss


class LossAggregatorNM(LossNM):
    """
    Neural module which combines sums several losses into one.

    Args:
        num_inputs (int): number of input losses
    """

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
            loss = loss.add(loss_i)
        return loss


class TokenClassificationLoss(LossNM):
    """
    Neural module which implements Token Classification loss.

    Args:
        num_classes (int): number of classes in a classifier, e.g. size
            of the vocabulary in language modeling objective
        logits (float): output of the classifier
        labels (long): ground truth labels
        loss_mask (bool): to differentiate from original tokens and paddings
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "logits": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            "labels": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "loss_mask": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }

        output_ports = {
            "loss": NeuralType(None),
        }
        return input_ports, output_ports

    def __init__(self, num_classes, **kwargs):
        LossNM.__init__(self, **kwargs)
        self._criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def _loss_function(self, logits, labels, loss_mask):
        active_loss = loss_mask.view(-1) > 0.5
        active_logits = logits.view(-1, self.num_classes)[active_loss]
        active_labels = labels.view(-1)[active_loss]

        loss = self._criterion(active_logits, active_labels)
        return loss


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
            "loss_mask": NeuralType({
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
                       loss_mask,
                       intents,
                       slots,
                       intent_loss_weight=0.6):
        intent_loss = self._criterion(intent_logits, intents)

        active_loss = loss_mask.view(-1)
        active_logits = slot_logits.view(-1, self.num_slots)[active_loss]
        active_labels = slots.view(-1)[active_loss]

        # To support empty active_labels
        if len(active_labels) == 0:
            slot_loss = 0.0
        else:
            slot_loss = self._criterion(active_logits, active_labels)
        loss = intent_loss * intent_loss_weight + \
            slot_loss * (1 - intent_loss_weight)

        return loss


class PaddedSmoothedCrossEntropyLossNM(LossNM):
    """
    Neural module which calculates CrossEntropyLoss and
    1) excludes padding tokens from loss calculation
    2) allows to use label smoothing regularization
    3) allows to calculate loss for the desired number of last tokens

    Args:
        label_smoothing (float): label smoothing regularization coefficient
        predict_last_k (int): how many last tokens to use for the loss
            calculation, important for fast evaluation of LM perplexity
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "logits":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            "target_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
        }

        output_ports = {"loss": NeuralType(None)}
        return input_ports, output_ports

    def __init__(self, **kwargs):
        LossNM.__init__(self, **kwargs)

        loss_params = {
            "label_smoothing": self.local_parameters.get("label_smoothing", 0),
            "predict_last_k": self.local_parameters.get("predict_last_k", 0)
        }
        self._loss_fn = SmoothedCrossEntropyLoss(**loss_params)
        self._pad_id = self.local_parameters['pad_id']

    def _loss_function(self, logits, target_ids):
        target_mask = mask_padded_tokens(
            target_ids, self._pad_id).to(logits.dtype)
        loss = self._loss_fn(logits, target_ids, target_mask)
        return loss
