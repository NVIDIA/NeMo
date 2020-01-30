__all__ = ['JointIntentSlotLoss']
import torch
from torch import nn

from nemo.backends.pytorch import LossNM
from nemo.core import AxisType, BatchTag, ChannelTag, NeuralType, TimeTag


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
            "intent_logits": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)}),
            "slot_logits": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag)}),
            "loss_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "intents": NeuralType({0: AxisType(BatchTag)}),
            "slots": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(None)}

    def __init__(
        self,
        num_slots,
        slot_classes_loss_weights=None,
        intent_classes_loss_weights=None,
        intent_loss_weight=0.6,
        **kwargs
    ):
        LossNM.__init__(self, **kwargs)
        self.num_slots = num_slots
        self.intent_loss_weight = intent_loss_weight
        self.slot_classes_loss_weights = slot_classes_loss_weights
        self.intent_classes_loss_weights = intent_classes_loss_weights

        # For weighted loss to tackle class imbalance
        if slot_classes_loss_weights:
            self.slot_classes_loss_weights = torch.FloatTensor(slot_classes_loss_weights).to(self._device)

        if intent_classes_loss_weights:
            self.intent_classes_loss_weights = torch.FloatTensor(intent_classes_loss_weights).to(self._device)

        self._criterion_intent = nn.CrossEntropyLoss(weight=self.intent_classes_loss_weights)
        self._criterion_slot = nn.CrossEntropyLoss(weight=self.slot_classes_loss_weights)

    def _loss_function(self, intent_logits, slot_logits, loss_mask, intents, slots):
        intent_loss = self._criterion_intent(intent_logits, intents)

        active_loss = loss_mask.view(-1) > 0.5
        active_logits = slot_logits.view(-1, self.num_slots)[active_loss]
        active_labels = slots.view(-1)[active_loss]

        # To support empty active_labels
        if len(active_labels) == 0:
            slot_loss = 0.0
        else:
            slot_loss = self._criterion_slot(active_logits, active_labels)
        loss = intent_loss * self.intent_loss_weight + slot_loss * (1 - self.intent_loss_weight)

        return loss
