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

import numpy as np
import torch

from nemo import logging
from nemo.collections.nlp.utils.callback_utils import tensor2list, tensor2numpy

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']


def eval_iter_callback(tensors, global_vars, data_desc):

    if 'loss' not in global_vars:
        global_vars['loss'] = []
    if 'comp_res' not in global_vars:
        global_vars['comp_res'] = []
    if 'gating_labels' not in global_vars:
        global_vars['gating_labels'] = []
    if 'gating_preds' not in global_vars:
        global_vars['gating_preds'] = []

    point_outputs_max_list = []
    tgt_ids_list = []
    gate_outputs_max_list = []
    for tensor_name, values_list in tensors.items():
        if tensor_name.startswith('gating_labels'):
            for values in values_list:
                global_vars['gating_labels'].extend(tensor2list(values))
        elif tensor_name.startswith('point_outputs'):
            for values in values_list:
                p_max = torch.argmax(values, dim=-1)
                point_outputs_max_list.append(tensor2numpy(p_max))
        elif tensor_name.startswith('gate_outputs'):
            for values in values_list:
                g_max = torch.argmax(values, axis=-1)
                gate_outputs_max_list.append(tensor2numpy(g_max))
        elif tensor_name.startswith('tgt_ids'):
            for values in values_list:
                tgt_ids_list.append(tensor2numpy(values))

    comp_res_list = []
    for i in range(len(point_outputs_max_list)):
        mask_paddings = tgt_ids_list[i] == data_desc.vocab.pad_id
        comp_res = (point_outputs_max_list[i] == tgt_ids_list[i]) | mask_paddings
        comp_res = np.all(comp_res, axis=-1, keepdims=False)
        comp_res_list.extend(comp_res.tolist())

    gate_outputs_max = np.concatenate(gate_outputs_max_list, axis=0).tolist()

    global_vars['comp_res'].extend(comp_res_list)
    global_vars['gating_preds'].extend(gate_outputs_max)


def eval_epochs_done_callback(global_vars, data_desc):
    joint_acc, slot_acc = evaluate_metrics(
        global_vars['comp_res'],
        global_vars['gating_labels'],
        global_vars['gating_preds'],
        data_desc.gating_dict["ptr"],
    )

    gating_comp_flatten = (np.asarray(global_vars['gating_labels']) == np.asarray(global_vars['gating_preds'])).ravel()
    gating_acc = np.sum(gating_comp_flatten) / len(gating_comp_flatten)

    evaluation_metrics = {"Joint_Goal_Acc": joint_acc, "Slot_Acc": slot_acc, "Gate_Acc": gating_acc}
    logging.info(evaluation_metrics)

    return evaluation_metrics


def evaluate_metrics(comp_res, gating_labels, gating_preds, ptr_code):
    # TODO: Calculate precision, recall, and F1
    total_slots = 0
    correct_slots = 0
    total_turns = 0
    correct_turns = 0
    for result_idx, result in enumerate(comp_res):
        turn_wrong = False
        total_turns += 1
        for slot_idx, slot_eq in enumerate(result):
            total_slots += 1
            if gating_labels[result_idx][slot_idx] == ptr_code:
                if slot_eq:
                    correct_slots += 1
                else:
                    turn_wrong = True
            elif gating_labels[result_idx][slot_idx] == gating_preds[result_idx][slot_idx] or (
                slot_eq and gating_preds[result_idx][slot_idx] == ptr_code
            ):
                correct_slots += 1
            else:
                turn_wrong = True
        if not turn_wrong:
            correct_turns += 1

    slot_acc = correct_slots / float(total_slots) if total_slots != 0 else 0
    joint_acc = correct_turns / float(total_turns) if total_turns != 0 else 0
    return joint_acc, slot_acc
