# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
# Copyright 2019 The Google Research Authors.
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

'''
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst/baseline/train_and_predict.py
'''

import torch

from nemo import logging
from nemo.backends.pytorch import LossNM
from nemo.collections.nlp.data.datasets.sgd_dataset.input_example import STATUS_ACTIVE
from nemo.core import ChannelType, LabelsType, LogitsType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['SGDDialogueStateLossNM']


class SGDDialogueStateLossNM(LossNM):
    """
    Neural module which implements loss for SGD model.
    """

    @property
    @add_port_docs
    def input_ports(self):
        """Returns definitions of module input ports.
            logit_intent_status (float): Output of SGD model
            intent_status_labels (int): Intent labels
            logit_req_slot_status (float): Output of SGD model
            requested_slot_status (float): Takes value 1 if the corresponding slot is requested, 0 otherwise
            req_slot_mask (bool): Masks requested slots not used for the particular service
            logit_cat_slot_status (float): Output of SGD model
            categorical_slot_status (int): The status of each categorical slot in the service
            cat_slot_status_mask (bool): Masks categorical slots not used for the particular service
            logit_cat_slot_value (float): Output of SGD model
            categorical_slot_values (int): The index of the correct value for each categorical slot
            logit_noncat_slot_status (float): Output of SGD model
            noncategorical_slot_status (int): The status of each noncategorical slot in the service
            noncat_slot_status_mask (bool): masks noncategorical slots not used for the particular service
            logit_noncat_slot_start (float): Output of SGD model
            logit_noncat_slot_end (float): Output of SGD model
            noncategorical_slot_value_start (int): The index of the starting subword corresponding to the slot span for a non-categorical slot value
            noncategorical_slot_value_end (int): The index of the ending (inclusive) subword corresponding to the slot span for a non-categorical slot value
        """
        return {
            "logit_intent_status": NeuralType(('B', 'T', 'C'), LogitsType()),
            "intent_status_labels": NeuralType(('B'), LabelsType()),
            "logit_req_slot_status": NeuralType(('B', 'T'), LogitsType()),
            "requested_slot_status": NeuralType(('B', 'T'), LabelsType()),
            "req_slot_mask": NeuralType(('B', 'T'), ChannelType()),
            "logit_cat_slot_status": NeuralType(('B', 'T', 'C'), LogitsType()),
            "categorical_slot_status": NeuralType(('B', 'T'), LabelsType()),
            "cat_slot_status_mask": NeuralType(('B', 'T'), ChannelType()),
            "logit_cat_slot_value": NeuralType(('B', 'T', 'C'), LogitsType()),
            "categorical_slot_values": NeuralType(('B', 'T'), LabelsType()),
            "logit_noncat_slot_status": NeuralType(('B', 'T', 'C'), LogitsType()),
            "noncategorical_slot_status": NeuralType(('B', 'T'), LabelsType()),
            "noncat_slot_status_mask": NeuralType(('B', 'T'), ChannelType()),
            "logit_noncat_slot_start": NeuralType(('B', 'T', 'C'), LogitsType()),
            "logit_noncat_slot_end": NeuralType(('B', 'T', 'C'), LogitsType()),
            "noncategorical_slot_value_start": NeuralType(('B', 'T'), LabelsType()),
            "noncategorical_slot_value_end": NeuralType(('B', 'T'), LabelsType()),
        }

    @property
    def output_ports(self):
        """
        Returns definitions of module output ports.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(None)}

    def __init__(self, add_carry_status, reduction='mean'):
        """
        Args:
            add_carry_status (bool): specifies that carry status is enabled for slots
            reduction (str): specifies the reduction to apply to the final loss, choose 'mean' or 'sum'
        """
        super().__init__()

        self._add_carry_status = add_carry_status

        if reduction not in ['mean', 'sum']:
            logging.warning(f'{reduction} reduction is not supported. Setting reduction to "mean"')
            reduction = 'mean'

        self.reduction = reduction
        self._cross_entropy = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        self._criterion_req_slots = torch.nn.BCEWithLogitsLoss(reduction=self.reduction)

    def _loss_function(
        self,
        logit_intent_status,
        intent_status_labels,
        logit_req_slot_status,
        requested_slot_status,
        req_slot_mask,
        logit_cat_slot_status,
        categorical_slot_status,
        cat_slot_status_mask,
        logit_cat_slot_value,
        categorical_slot_values,
        logit_noncat_slot_status,
        noncategorical_slot_status,
        noncat_slot_status_mask,
        logit_noncat_slot_start,
        logit_noncat_slot_end,
        noncategorical_slot_value_start,
        noncategorical_slot_value_end,
    ):
        # Intent loss
        intent_loss = self._cross_entropy(logit_intent_status, intent_status_labels)

        # Requested slots.
        # Shape: (batch_size, max_num_slots)
        # mask unused slots
        # Sigmoid cross entropy is used because more than one slots can be requested in a single utterance
        req_slot_mask = req_slot_mask > 0.5
        requested_slot_loss = self._criterion_req_slots(
            logit_req_slot_status[req_slot_mask], requested_slot_status[req_slot_mask]
        )

        # Categorical slot status
        # Shape of logit_cat_slot_status: (batch_size, max_num_cat_slots, 3)
        # cat_slot_status_mask masks unused categorical padded slots for the service
        cat_slot_status_mask = cat_slot_status_mask.view(-1) > 0.5
        if sum(cat_slot_status_mask) == 0:
            logging.warning(f'No categorical slots in the batch')
            cat_slot_status_loss = torch.clamp(torch.max(logit_cat_slot_status.view(-1)), 0, 0)
        else:
            cat_slot_status_loss = self._cross_entropy(
                logit_cat_slot_status.view(-1, logit_cat_slot_status.size()[-1])[cat_slot_status_mask],
                categorical_slot_status.view(-1)[cat_slot_status_mask],
            )

        # Categorical slot values.
        # Shape: (batch_size, max_num_cat_slots, max_num_slot_values).
        max_num_slot_values = logit_cat_slot_value.size()[-1]

        # Zero out losses for categorical slot value when the slot status is not active.
        if self._add_carry_status:
            cat_slot_value_mask = (categorical_slot_status >= STATUS_ACTIVE).view(-1)
        else:
            cat_slot_value_mask = (categorical_slot_status == STATUS_ACTIVE).view(-1)

        # to handle cases with no active categorical slot value
        if sum(cat_slot_value_mask) == 0:
            logging.warning(f'No active values for categorical slots in the batch.')
            cat_slot_value_loss = torch.clamp(torch.max(logit_cat_slot_value.view(-1)), 0, 0)
        else:
            slot_values_active_logits = logit_cat_slot_value.view(-1, max_num_slot_values)[cat_slot_value_mask]
            slot_values_active_labels = categorical_slot_values.view(-1)[cat_slot_value_mask]
            cat_slot_value_loss = self._cross_entropy(slot_values_active_logits, slot_values_active_labels)

        # Non-categorical slot status.
        # Shape: (batch_size, max_num_noncat_slots, 3).
        # noncat_slot_status_mask masks unused noncat slots for the service
        noncat_slot_status_mask = noncat_slot_status_mask.view(-1) > 0.5
        if sum(noncat_slot_status_mask) == 0:
            logging.warning(f'No active non-categorical slots in the batch.')
            noncat_slot_status_loss = torch.clamp(torch.max(logit_noncat_slot_status.view(-1)), 0, 0)
        else:
            noncat_slot_status_loss = self._cross_entropy(
                logit_noncat_slot_status.view(-1, logit_noncat_slot_status.size()[-1])[noncat_slot_status_mask],
                noncategorical_slot_status.view(-1)[noncat_slot_status_mask],
            )

        # Non-categorical slot spans.
        # Shape: (batch_size, max_num_noncat_slots, max_num_tokens).n
        max_num_tokens = logit_noncat_slot_start.size()[-1]
        # Zero out losses for non-categorical slot spans when the slot status is not active.
        if self._add_carry_status:
            non_cat_slot_value_mask = (noncategorical_slot_status >= STATUS_ACTIVE).view(-1)
        else:
            non_cat_slot_value_mask = (noncategorical_slot_status == STATUS_ACTIVE).view(-1)

        # to handle cases with no active categorical slot value
        if sum(non_cat_slot_value_mask) == 0:
            logging.warning(f'No active values for non-categorical slots in the batch.')
            span_start_loss = torch.clamp(torch.max(logit_noncat_slot_start.view(-1)), 0, 0)
            span_end_loss = torch.clamp(torch.max(logit_noncat_slot_end.view(-1)), 0, 0)
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

        total_loss = sum(losses.values())
        if self.reduction == 'mean':
            total_loss = total_loss / len(losses)
        else:
            batch_size = logit_intent_status.shape[0]
            total_loss = total_loss / batch_size
        return total_loss
