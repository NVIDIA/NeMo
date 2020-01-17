import torch
from torch import nn


from nemo.backends.pytorch.nm import LossNM
from nemo.core.neural_types import *
from .pytorch_utils import SmoothedCrossEntropyLoss
from ..utils.nlp_utils import mask_padded_tokens


__all__ = ['JointIntentSlotLoss',
           'LossAggregatorNM',
           'MaskedLanguageModelingLossNM',
           'PaddedSmoothedCrossEntropyLossNM',
           'QuestionAnsweringLoss',
           'TokenClassificationLoss']


class QuestionAnsweringLoss(LossNM):
    """
    Neural module which implements QuestionAnswering loss.
    Args:
        logits: Output of question answering head, which is a token classfier.
        start_positions: Ground truth start positions of the answer w.r.t.
            input sequence. If question is unanswerable, this will be
            pointing to start token, e.g. [CLS], of the input sequence.
        end_positions: Ground truth end positions of the answer w.r.t.
            input sequence. If question is unanswerable, this will be
            pointing to start token, e.g. [CLS], of the input sequence.
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)

        start_positions:
            0: AxisType(BatchTag)

        end_positions:
            0: AxisType(BatchTag)
        """
        return {
            "logits": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            "start_positions": NeuralType({
                0: AxisType(BatchTag)
            }),
            "end_positions": NeuralType({
                0: AxisType(BatchTag)
            })
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)

        start_logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        end_logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {
            "loss": NeuralType(None),
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

    def __init__(self, **kwargs):
        LossNM.__init__(self, **kwargs)

    def _loss_function(self, logits, start_positions, end_positions):
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss, start_logits, end_logits


class MaskedLanguageModelingLossNM(LossNM):
    """
    Neural module which implements Masked Language Modeling (MLM) loss.

    Args:
        label_smoothing (float): label smoothing regularization coefficient
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)

        output_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        output_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {
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
            })
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {
            "loss": NeuralType(None)
        }

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

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        """
        input_ports = {}
        for i in range(self.num_losses):
            input_ports["loss_" + str(i + 1)] = NeuralType(None)

        return input_ports

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {
            "loss": NeuralType(None)
        }

    def __init__(self, *, num_inputs=2, **kwargs):
        # Store number of inputs/losses.
        self.num_losses = num_inputs
        # kwargs["create_port_args"] = {"num_losses": num_inputs}
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
        loss_mask (long): to differentiate from original tokens and paddings
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)

        labels:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        loss_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {
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

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {
            "loss": NeuralType(None)
        }

    def __init__(self, num_classes, class_weights=None, **kwargs):
        LossNM.__init__(self, **kwargs)
        if class_weights:
            class_weights = torch.FloatTensor(class_weights).to(self._device)

        self._criterion = nn.CrossEntropyLoss(weight=class_weights)
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

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        intent_logits:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag)

        slot_logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)

        loss_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        intents:
            0: AxisType(BatchTag)

        slots:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {
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

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {
            "loss": NeuralType(None)
        }

    def __init__(self,
                 num_slots,
                 slot_classes_loss_weights=None,
                 intent_classes_loss_weights=None,
                 intent_loss_weight=0.6,
                 **kwargs):
        LossNM.__init__(self, **kwargs)
        self.num_slots = num_slots
        self.intent_loss_weight = intent_loss_weight
        self.slot_classes_loss_weights = slot_classes_loss_weights
        self.intent_classes_loss_weights = intent_classes_loss_weights

        # For weighted loss to tackle class imbalance
        if slot_classes_loss_weights:
            self.slot_classes_loss_weights = torch.FloatTensor(
                slot_classes_loss_weights).to(self._device)

        if intent_classes_loss_weights:
            self.intent_classes_loss_weights = torch.FloatTensor(
                intent_classes_loss_weights).to(self._device)

        self._criterion_intent = nn.CrossEntropyLoss(
            weight=self.intent_classes_loss_weights)
        self._criterion_slot = nn.CrossEntropyLoss(
            weight=self.slot_classes_loss_weights)

    def _loss_function(self,
                       intent_logits,
                       slot_logits,
                       loss_mask,
                       intents,
                       slots):
        intent_loss = self._criterion_intent(intent_logits, intents)

        active_loss = loss_mask.view(-1) > 0.5
        active_logits = slot_logits.view(-1, self.num_slots)[active_loss]
        active_labels = slots.view(-1)[active_loss]

        # To support empty active_labels
        if len(active_labels) == 0:
            slot_loss = 0.0
        else:
            slot_loss = self._criterion_slot(active_logits, active_labels)
        loss = intent_loss * self.intent_loss_weight + \
            slot_loss * (1 - self.intent_loss_weight)

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

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        logits:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

            2: AxisType(ChannelTag)

        target_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {
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
            })
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {
            "loss": NeuralType(None)
        }

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
