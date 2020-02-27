# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
from torch import nn

from nemo.backends.pytorch import LossNM
from nemo.core import ChannelType, LogitsType, LossType, NeuralType, LabelsType, LengthsType
from nemo.utils.decorators import add_port_docs
import nemo.collections.nlp.data.datasets.sgd_dataset.data_utils as data_utils

__all__ = ['SGDDialogueStateLoss']


class SGDDialogueStateLoss(LossNM):
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
            "logit_intent_status": NeuralType(('B', 'T', 'C'), LogitsType()),
            "logit_req_slot_status": NeuralType(('B', 'T'), LogitsType()),
            "req_slot_mask": NeuralType(('B', 'T'), ChannelType()),
            "logit_cat_slot_status": NeuralType(('B', 'T', 'C'), LogitsType()),
            "logit_cat_slot_value": NeuralType(('B', 'T', 'C'), LogitsType()),
            "cat_slot_values_mask": NeuralType(('B', 'T', 'C'), ChannelType()),
            "logit_noncat_slot_status": NeuralType(('B', 'T', 'C'), LogitsType()),
            "logit_noncat_slot_start": NeuralType(('B', 'T', 'C'), LogitsType()),
            "logit_noncat_slot_end": NeuralType(('B', 'T', 'C'), LogitsType()),
            "intent_status": NeuralType(('B'), LabelsType()),
            "requested_slot_status": NeuralType(('B', 'T'), LabelsType()),
            "categorical_slot_status": NeuralType(('B', 'T'), LabelsType()),
            "num_categorical_slots": NeuralType(('B'), LengthsType()),
            "categorical_slot_values": NeuralType(('B', 'T'), LabelsType()),
            "noncategorical_slot_status": NeuralType(('B', 'T'), LabelsType()),
            "num_noncategorical_slots": NeuralType(('B'), LengthsType()),
            "noncategorical_slot_value_start": NeuralType(('B', 'T'), LabelsType()),
            "noncategorical_slot_value_end": NeuralType(('B', 'T'), LabelsType())
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

        self._cross_entropy = nn.CrossEntropyLoss()
        self._criterion_req_slots = nn.BCEWithLogitsLoss()

    def _get_mask(self,
                  max_number,
                  values):

        mask = torch.arange(0, max_number, 1).to(self._device) < torch.unsqueeze(values, dim=-1)
        return mask.view(-1)

    def _loss_function(self,
                       logit_intent_status,
                       logit_req_slot_status,
                       logit_cat_slot_status,
                       logit_cat_slot_value,
                       logit_noncat_slot_status,
                       logit_noncat_slot_start,
                       logit_noncat_slot_end,
                       intent_status,
                       requested_slot_status,
                       req_slot_mask,
                       categorical_slot_status,
                       num_categorical_slots,
                       categorical_slot_values,
                       cat_slot_values_mask,
                       noncategorical_slot_status,
                       num_noncategorical_slots,
                       noncategorical_slot_value_start,
                       noncategorical_slot_value_end):
        """
        Obtain the loss of the model
        """

        """
        Intents:
            logit_intent_status Shape: (batch_size, max_num_intents + 1)
            intent_status (labels) Shape: (batch_size, max_num_intents) - one-hot encoded
        """

        # Intent loss
        # Add label corresponding to NONE intent.
        num_active_intents = torch.sum(intent_status, axis=1).unsqueeze(1)
        none_intent_label = torch.ones(num_active_intents.size(), dtype=torch.long).to(
            self._device) - num_active_intents
        # Shape: (batch_size, max_num_intents + 1).
        onehot_intent_labels = torch.cat([none_intent_label, intent_status], axis=1)
        # use indices for intent labels - tf uses one_hot_encoding
        _, intent_labels = onehot_intent_labels.max(dim=1)
        intent_loss = self._cross_entropy(logit_intent_status, intent_labels)

        # Requested slots.
        # Shape: (batch_size, max_num_slots)
        # mask unused slots
        # Sigmoid cross entropy is used because more than one slots can be requested in a single utterance
        requested_slot_loss = self._criterion_req_slots(logit_req_slot_status.view(-1)[req_slot_mask],
                                                        requested_slot_status.view(-1)[req_slot_mask])

        # Categorical slot status
        # Shape of logit_cat_slot_status: (batch_size, max_num_cat_slots, 3)
        max_num_cat_slots = categorical_slot_status.size()[-1]

        cat_slot_status_mask = self._get_mask(max_num_cat_slots, num_categorical_slots)

        if sum(cat_slot_status_mask) == 0:
            cat_slot_status_loss = 0
        else:
            cat_slot_status_loss = self._cross_entropy(logit_cat_slot_status.view(-1, 3)[cat_slot_status_mask],
                                                       categorical_slot_status.view(-1)[cat_slot_status_mask])

        # Categorical slot values.
        # Shape: (batch_size, max_num_cat_slots, max_num_slot_values).
        max_num_slot_values = logit_cat_slot_value.size()[-1]

        # Zero out losses for categorical slot value when the slot status is not active.
        cat_slot_value_mask = (categorical_slot_status == data_utils.STATUS_ACTIVE).view(-1)
        # to handle cases with no active categorical slot value
        if sum(cat_slot_value_mask) == 0:
            cat_slot_value_loss = 0
        else:
            slot_values_active_logits = logit_cat_slot_value.view(-1, max_num_slot_values)[cat_slot_value_mask]
            slot_values_active_labels = categorical_slot_values.view(-1)[cat_slot_value_mask]
            cat_slot_value_loss = self._cross_entropy(slot_values_active_logits, slot_values_active_labels)

        # Non-categorical slot status.
        # Shape: (batch_size, max_num_noncat_slots, 3).
        max_num_noncat_slots = noncategorical_slot_status.size()[-1]
        non_cat_slot_status_mask = self._get_mask(max_num_noncat_slots, num_noncategorical_slots)
        noncat_slot_status_loss = self._cross_entropy(logit_noncat_slot_status.view(-1, 3)[non_cat_slot_status_mask],
                                                      noncategorical_slot_status.view(-1)[non_cat_slot_status_mask])

        # Non-categorical slot spans.
        # Shape: (batch_size, max_num_noncat_slots, max_num_tokens).n
        max_num_tokens = logit_noncat_slot_start.size()[-1]
        # Zero out losses for non-categorical slot spans when the slot status is not active.
        non_cat_slot_value_mask = (noncategorical_slot_status == data_utils.STATUS_ACTIVE).view(-1)
        # to handle cases with no active categorical slot value
        if sum(non_cat_slot_value_mask) == 0:
            span_start_loss = 0
            span_end_loss = 0
        else:
            noncat_slot_start_active_logits = logit_noncat_slot_start.view(-1, max_num_tokens)[non_cat_slot_value_mask]
            noncat_slot_start_active_labels = noncategorical_slot_value_start.view(-1)[non_cat_slot_value_mask]
            span_start_loss = self._cross_entropy(noncat_slot_start_active_logits, noncat_slot_start_active_labels)

            noncat_slot_end_active_logits = logit_noncat_slot_end.view(-1, max_num_tokens)[non_cat_slot_value_mask]
            noncat_slot_end_active_labels = noncategorical_slot_value_end.view(-1)[non_cat_slot_value_mask]
            span_end_loss = self._cross_entropy(noncat_slot_end_active_logits, noncat_slot_end_active_labels)

        losses = {
            "intent_loss": intent_loss,
            "requested_slot_loss": requested_slot_loss,
            "cat_slot_status_loss": cat_slot_status_loss,
            "cat_slot_value_loss": cat_slot_value_loss,
            "noncat_slot_status_loss": noncat_slot_status_loss,
            "span_start_loss": span_start_loss,
            "span_end_loss": span_end_loss,
        }
        # for loss_name, loss in losses.items():
        #     print (f'loss_name: {loss_name}, {loss}')
        return sum(losses.values()) / len(losses), intent_labels


