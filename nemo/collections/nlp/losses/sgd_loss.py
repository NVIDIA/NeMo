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

from nemo.collections.nlp.data.dialogue_state_tracking_sgd.input_example import STATUS_ACTIVE
from nemo.core.classes import Loss, Typing, typecheck
from nemo.core.neural_types import ChannelType, LabelsType, LogitsType, LossType, NeuralType
from nemo.utils import logging

__all__ = ['SGDDialogueStateLoss']


class SGDDialogueStateLoss(Loss):
    """
    Neural module which implements loss for SGD model.
    """

    @property
    def input_types(self):
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
            "logit_intent_status": NeuralType(('B', 'T'), LogitsType()),
            "intent_status": NeuralType(('B'), LabelsType()),
            "logit_req_slot_status": NeuralType(('B', 'T'), LogitsType()),
            "requested_slot_status": NeuralType(('B'), LabelsType()),
            "logit_cat_slot_status": NeuralType(('B', 'T'), LogitsType()),
            "categorical_slot_status": NeuralType(('B'), LabelsType()),
            "logit_cat_slot_value_status": NeuralType(('B', 'T'), LogitsType()),
            "categorical_slot_value_status": NeuralType(('B'), LabelsType()),
            "logit_noncat_slot_status": NeuralType(('B', 'T'), LogitsType()),
            "noncategorical_slot_status": NeuralType(('B'), LabelsType()),
            "logit_noncat_slot_start": NeuralType(('B', 'T'), LogitsType()),
            "logit_noncat_slot_end": NeuralType(('B', 'T'), LogitsType()),
            "noncategorical_slot_value_start": NeuralType(('B'), LabelsType()),
            "noncategorical_slot_value_end": NeuralType(('B'), LabelsType()),
            "task_mask": NeuralType(('B', 'T'), ChannelType()),
        }

    @property
    def output_types(self):
        """
        Returns definitions of module output ports.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): specifies the reduction to apply to the final loss, choose 'mean' or 'sum'
        """
        super().__init__()

        if reduction not in ['mean', 'sum']:
            logging.warning(f'{reduction} reduction is not supported. Setting reduction to "mean"')
            reduction = 'mean'

        self.reduction = reduction
        self._cross_entropy = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        self._cross_entropy_bin = torch.nn.BCEWithLogitsLoss(reduction=self.reduction)

    def _helper(self, logits, labels, loss_mask):
        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-1)

        if loss_mask is not None:
            if loss_mask.dtype is not torch.bool:
                loss_mask = loss_mask > 0.5
            loss_mask_flatten = torch.flatten(loss_mask, start_dim=0, end_dim=-1)
            logits_flatten = logits_flatten[loss_mask_flatten]
            labels_flatten = labels_flatten[loss_mask_flatten]

        return logits_flatten, labels_flatten

    @typecheck()
    def forward(
        self,
        logit_intent_status,
        intent_status,
        logit_req_slot_status,
        requested_slot_status,
        logit_cat_slot_status,
        categorical_slot_status,
        logit_cat_slot_value_status,
        categorical_slot_value_status,
        logit_noncat_slot_status,
        noncategorical_slot_status,
        logit_noncat_slot_start,
        logit_noncat_slot_end,
        noncategorical_slot_value_start,
        noncategorical_slot_value_end,
        task_mask,
    ):
        # Intent loss
        old_logit_intent_status = logit_intent_status
        logit_intent_status, intent_status = self._helper(logit_intent_status, intent_status, task_mask[:, 0])
        if len(intent_status) == 0:
            intent_loss = torch.clamp(torch.max(old_logit_intent_status.view(-1)), 0, 0)
        else:
            intent_loss = self._cross_entropy_bin(logit_intent_status.squeeze(dim=-1), intent_status)

        old_logit_req_slot_status = logit_req_slot_status
        logit_req_slot_status, requested_slot_status = self._helper(
            logit_req_slot_status, requested_slot_status, task_mask[:, 1]
        )
        if len(requested_slot_status) == 0:
            requested_slot_loss = torch.clamp(torch.max(old_logit_req_slot_status.view(-1)), 0, 0)
        else:
            requested_slot_loss = self._cross_entropy_bin(logit_req_slot_status.squeeze(dim=-1), requested_slot_status)

        old_logit_cat_slot_status = logit_cat_slot_status
        logit_cat_slot_status, categorical_slot_status = self._helper(
            logit_cat_slot_status, categorical_slot_status, task_mask[:, 2]
        )
        if len(categorical_slot_status) == 0:
            cat_slot_status_loss = torch.clamp(torch.max(old_logit_cat_slot_status.view(-1)), 0, 0)
        else:
            cat_slot_status_loss = self._cross_entropy(logit_cat_slot_status, categorical_slot_status,)
        old_logit_cat_slot_value_status = logit_cat_slot_value_status
        logit_cat_slot_value_status, categorical_slot_value_status = self._helper(
            logit_cat_slot_value_status, categorical_slot_value_status, task_mask[:, 3]
        )
        if len(categorical_slot_value_status) == 0:
            cat_slot_value_status_loss = torch.clamp(torch.max(old_logit_cat_slot_value_status.view(-1)), 0, 0)
        else:
            cat_slot_value_status_loss = self._cross_entropy_bin(
                logit_cat_slot_value_status.squeeze(dim=-1), categorical_slot_value_status
            )

        old_logit_noncat_slot_status = logit_noncat_slot_status
        logit_noncat_slot_status, noncategorical_slot_status = self._helper(
            logit_noncat_slot_status, noncategorical_slot_status, task_mask[:, 4]
        )
        if len(noncategorical_slot_status) == 0:
            noncat_slot_status_loss = torch.clamp(torch.max(old_logit_noncat_slot_status.view(-1)), 0, 0)
        else:
            noncat_slot_status_loss = self._cross_entropy(logit_noncat_slot_status, noncategorical_slot_status,)

        _, max_num_tokens = logit_noncat_slot_start.size()
        old_logit_noncat_slot_start = logit_noncat_slot_start
        logit_noncat_slot_start, noncategorical_slot_value_start = self._helper(
            logit_noncat_slot_start, noncategorical_slot_value_start, task_mask[:, 5]
        )
        if len(noncategorical_slot_value_start) == 0:
            span_start_loss = torch.clamp(torch.max(old_logit_noncat_slot_start.view(-1)), 0, 0)
        else:
            span_start_loss = self._cross_entropy(logit_noncat_slot_start, noncategorical_slot_value_start)

        old_logit_noncat_slot_end = logit_noncat_slot_end
        logit_noncat_slot_end, noncategorical_slot_value_end = self._helper(
            logit_noncat_slot_end, noncategorical_slot_value_end, task_mask[:, 5]
        )
        if len(noncategorical_slot_value_end) == 0:
            span_end_loss = torch.clamp(torch.max(old_logit_noncat_slot_end.view(-1)), 0, 0)
        else:
            span_end_loss = self._cross_entropy(logit_noncat_slot_end, noncategorical_slot_value_end)

        losses = {
            "intent_loss": intent_loss,
            "requested_slot_loss": requested_slot_loss,
            "cat_slot_status_loss": cat_slot_status_loss,
            "cat_slot_value_status_loss": cat_slot_value_status_loss,
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
