# Copyright (c) 2019 NVIDIA Corporation
__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']

import os
import random
import time

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from nemo.utils.exp_logging import get_logger

logger = get_logger('')


def eval_iter_callback(tensors,
                       global_vars,
                       eval_data_layer):
    # print(tensors)
    # print(global_vars)
    if 'loss' not in global_vars:
        global_vars['loss'] = []
    if 'point_outputs' not in global_vars:
        global_vars['point_outputs'] = []
    if 'gate_outputs' not in global_vars:
        global_vars['gate_outputs'] = []
    if 'gating_labels' not in global_vars:
        global_vars['gating_labels'] = []
    if 'gate_outputs' not in global_vars:
        global_vars['gate_outputs'] = []
    if 'tgt_ids' not in global_vars:
        global_vars['tgt_ids'] = []
    if 'tgt_lens' not in global_vars:
        global_vars['tgt_lens'] = []
    if 'eval_res' not in global_vars:
        global_vars['eval_res'] = []
    if 'eval_mask' not in global_vars:
        global_vars['eval_mask'] = []

    for kv, v in tensors.items():
        if kv.startswith('loss'):
            loss_numpy = v[0].cpu().numpy()
            global_vars['loss'].append(loss_numpy)
        if kv.startswith('point_outputs'):
            point_outputs = v[0].cpu().numpy()
            global_vars['point_outputs'].extend(point_outputs)
        if kv.startswith('gate_outputs'):
            gate_outputs = v[0].cpu().numpy()
            global_vars['gate_outputs'].extend(gate_outputs)
        if kv.startswith('gating_labels'):
            gating_labels = v[0].cpu().numpy()
            global_vars['gating_labels'].extend(gating_labels)
        if kv.startswith('tgt_ids'):
            tgt_ids = v[0].cpu().numpy()
            global_vars['tgt_ids'].extend(tgt_ids)
        if kv.startswith('tgt_lens'):
            tgt_lens = v[0].cpu().numpy()
            global_vars['tgt_lens'].extend(tgt_lens)


    # # Set to not-training mode to disable dropout
    # self.encoder.train(False)
    # self.decoder.train(False)
    #print("STARTING EVALUATION")
    #all_prediction = {}
    #inverse_unpoint_slot = dict([(v, k) for k, v in eval_data_layer.gating_dict.items()])

    #batch_size = len(data_dev['context_len'])
    #batch_size = len(gating_labels)
    #_, gates, words, class_words = self.encode_and_decode(data_dev, False, slot_temp)

    #tgt_ids = tgt_ids.ravel()
    point_outputs_max = np.argmax(point_outputs, axis=-1)
    #point_outputs = point_outputs.ravel()

    mask_paddings = (tgt_ids == eval_data_layer.pad_id)
    comp_res = np.logical_or(point_outputs_max == tgt_ids, mask_paddings)
    comp_res = np.all(comp_res, axis=-1, keepdims=False)

    # TODO: replace gating_lables with gating_outputs
    mask_gating = (gating_labels == eval_data_layer.gating_dict["ptr"])
    #comp_res = np.logical_or(comp_res, mask_gating)
    #comp_res = np.all(comp_res, axis=-1, keepdims=False)

    global_vars['eval_res'].extend(comp_res)
    global_vars['eval_mask'].extend(mask_gating)


    # all_prediction = []
    # for bi in range(batch_size):
    #     #if data_dev["ID"][bi] not in all_prediction.keys():
    #     #    all_prediction[data_dev["ID"][bi]] = {}
    #     predictions = {}
    #     predictions["turn_belief"] = turn_beliefs[bi]
    #
    #     predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
    #     gate = gating_labels[bi] #np.argmax(gates.transpose(0, 1)[bi], dim=1)
    #
    #     # pointer-generator results
    #     if eval_data_layer.use_gate:
    #         for si, sg in enumerate(gate):
    #             if sg == eval_data_layer.gating_dict["none"]:
    #                 continue
    #             elif sg == eval_data_layer.gating_dict["ptr"]:
    #                 pred = np.transpose(words[si])[bi]
    #                 st = []
    #                 for e in pred:
    #                     if e == 'EOS':
    #                         break
    #                     else:
    #                         st.append(e)
    #                 st = " ".join(st)
    #                 if st == "none":
    #                     continue
    #                 else:
    #                     predict_belief_bsz_ptr.append(slot_temp[si] + "-" + str(st))
    #             else:
    #                 predict_belief_bsz_ptr.append(slot_temp[si] + "-" + inverse_unpoint_slot[sg.item()])
    #     else:
    #         for si, _ in enumerate(gate):
    #             pred = np.transpose(words[si])[bi]
    #             st = []
    #             for e in pred:
    #                 if e == 'EOS':
    #                     break
    #                 else:
    #                     st.append(e)
    #             st = " ".join(st)
    #             if st == "none":
    #                 continue
    #             else:
    #                 predict_belief_bsz_ptr.append(slot_temp[si] + "-" + str(st))
    #
    #     all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr
    #
    #     if set(data_dev["turn_belief"][bi]) != set(predict_belief_bsz_ptr) and args["genSample"]:
    #         print("True", set(data_dev["turn_belief"][bi]))
    #         print("Pred", set(predict_belief_bsz_ptr), "\n")


def list2str(l):
    return ' '.join([str(j) for j in l])


def eval_epochs_done_callback(global_vars, graph_fold):
    #loss = np.mean(global_vars['loss'])
    #print(f'Loss: {loss}')

    joint_acc, turn_acc, F1 = \
        evaluate_metrics(global_vars['eval_res'], global_vars['eval_mask'])

    evaluation_metrics = {"Joint_Acc": joint_acc,
                          "Turn_Acc": turn_acc,
                          "Joint_F1": F1}
    print(evaluation_metrics)

    return evaluation_metrics


def evaluate_metrics(results, ptr_mask):
    total_slots = 0
    correct_slots = 0
    total_turns = 0
    correct_turns = 0
    TP, FP, FN = 0, 0, 0
    F1 = 0
    for result_idx, result in enumerate(results):
        turn_wrong = False
        total_turns += 1
        for slot_idx, slot in enumerate(result):
            if ptr_mask[result_idx][slot_idx]:
                total_slots += 1
                if slot:
                    correct_slots += 1
                else:
                    turn_wrong = True
        if not turn_wrong:
            correct_turns += 1

    turn_acc = correct_slots / float(total_slots) if total_slots != 0 else 0
    joint_acc = correct_turns / float(total_turns) if total_turns != 0 else 0
    return joint_acc, turn_acc, F1
