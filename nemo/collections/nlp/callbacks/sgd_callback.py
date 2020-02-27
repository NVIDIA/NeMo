# Copyright (c) 2019 NVIDIA Corporation

from nemo import logging
import nemo.collections.nlp.data.datasets.sgd_dataset.prediction_utils as pred_utils
from nemo.collections.nlp.data.datasets.sgd_dataset import *

import numpy as np
import torch
from fuzzywuzzy import fuzz
import os

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']

REQ_SLOT_THRESHOLD = 0.5
F1Scores = collections.namedtuple("F1Scores", ["f1", "precision", "recall"])

# Evaluation and other relevant metrics for DSTC8 Schema-guided DST.
# (1) Active intent accuracy.
ACTIVE_INTENT_ACCURACY = "active_intent_accuracy"
# (2) Slot tagging F1.
SLOT_TAGGING_F1 = "slot_tagging_f1"
SLOT_TAGGING_PRECISION = "slot_tagging_precision"
SLOT_TAGGING_RECALL = "slot_tagging_recall"
# (3) Requested slots F1.
REQUESTED_SLOTS_F1 = "requested_slots_f1"
REQUESTED_SLOTS_PRECISION = "requested_slots_precision"
REQUESTED_SLOTS_RECALL = "requested_slots_recall"
# (4) Average goal accuracy.
AVERAGE_GOAL_ACCURACY = "average_goal_accuracy"
AVERAGE_CAT_ACCURACY = "average_cat_accuracy"
AVERAGE_NONCAT_ACCURACY = "average_noncat_accuracy"
# (5) Joint goal accuracy.
JOINT_GOAL_ACCURACY = "joint_goal_accuracy"
JOINT_CAT_ACCURACY = "joint_cat_accuracy"
JOINT_NONCAT_ACCURACY = "joint_noncat_accuracy"

NAN_VAL = "NA"

ALL_SERVICES = "#ALL_SERVICES"
SEEN_SERVICES = "#SEEN_SERVICES"
UNSEEN_SERVICES = "#UNSEEN_SERVICES"

def tensor2list(tensor):
    return tensor.detach().cpu().tolist()

def tensor2numpy(tensor):
    return tensor.cpu().numpy()

def eval_iter_callback(tensors, global_vars):
    # global_vars_keys = ['example_id',
    #                     'service_id',
    #                     'is_real_example',
    #                     'intent_status',
    #                     'req_slot_status',
    #                     'cat_slot_status',
    #                     'cat_slot_value',
    #                     'noncat_slot_status',
    #                     'noncat_slot_start',
    #                     'noncat_slot_end',
    #                     'noncat_alignment_start',
    #                     'noncat_alignment_end']

    global_vars_keys = ['predictions']
    for key in global_vars_keys:
        if key not in global_vars:
            global_vars[key] = []

    output = {}
    for k, v in tensors.items():
        ind = k.find('~~~')
        if ind != -1:
            output[k[:ind]] = v[0]
    
    '''
    ['example_id', 'service_id', 'is_real_example', 'user_utterance', 'start_char_idx', 'end_char_idx',
    'logit_intent_status', 'logit_req_slot_status', 'logit_cat_slot_status', 'logit_cat_slot_value',
    'cat_slot_values_mask', 'logit_noncat_slot_status', 'logit_noncat_slot_start', 'logit_noncat_slot_end',
    'intent_status', 'requested_slot_status', 'req_slot_mask', 'categorical_slot_status', 'num_categorical_slots',
    'categorical_slot_values', 'noncategorical_slot_status',
    'num_noncategorical_slots', 'noncategorical_slot_value_start', 'noncategorical_slot_value_end'])
    '''
    predictions = {}
    predictions['example_id'] = output['example_id']
    predictions['service_id'] = output['service_id']
    predictions['is_real_example'] = output['is_real_example']
    

    # Scores are output for each intent.
    # Note that the intent indices are shifted by 1 to account for NONE intent.
    predictions['intent_status'] = torch.argmax(output['logit_intent_status'], -1)

    # Scores are output for each requested slot.
    predictions['req_slot_status'] = torch.nn.Sigmoid()(output['logit_req_slot_status'])

    # For categorical slots, the status of each slot and the predicted value are output.
    predictions['cat_slot_status'] = torch.argmax(output['logit_cat_slot_status'], axis=-1)
    predictions['cat_slot_value'] = torch.argmax(output['logit_cat_slot_value'], axis=-1)

    # For non-categorical slots, the status of each slot and the indices for spans are output.
    predictions['noncat_slot_status'] = torch.argmax(output['logit_noncat_slot_status'], axis=-1)
    softmax = torch.nn.Softmax()
    start_scores = softmax(output['logit_noncat_slot_start'])
    end_scores = softmax(output['logit_noncat_slot_end'])

    batch_size, max_num_noncat_slots, max_num_tokens = end_scores.size()
    # Find the span with the maximum sum of scores for start and end indices.
    total_scores = torch.unsqueeze(start_scores, axis=3) + torch.unsqueeze(end_scores, axis=2)
    # Mask out scores where start_index > end_index.
    device = total_scores.device
    start_idx = torch.arange(max_num_tokens).view(1, 1, -1, 1).to(device)
    end_idx = torch.arange(max_num_tokens).view(1, 1, 1, -1).to(device)
    invalid_index_mask = (start_idx > end_idx).repeat(batch_size, max_num_noncat_slots, 1, 1)
    total_scores = torch.where(invalid_index_mask, torch.zeros(total_scores.size()).to(device), total_scores)
    max_span_index = torch.argmax(total_scores.view(-1, max_num_noncat_slots, max_num_tokens**2), axis=-1)
    span_start_index = torch.div(max_span_index, max_num_tokens)
    span_end_index = torch.fmod(max_span_index, max_num_tokens)
    
    predictions['noncat_slot_start'] = span_start_index
    predictions['noncat_slot_end'] = span_end_index

    # Add inverse alignments.
    predictions['noncat_alignment_start'] = output['start_char_idx']
    predictions['noncat_alignment_end'] = output['end_char_idx']

    global_vars['predictions'].extend(combine_predictions_in_example(predictions, batch_size))

def combine_predictions_in_example(predictions,
                                   batch_size):
    '''
    Combines predicted values to a single example.
    '''
    examples_preds = [{} for _ in range(batch_size)]
    for k, v in predictions.items():
        if k != 'example_id':
            v = torch.chunk(v, batch_size)

        for i in range(batch_size):
            if k == 'example_id':
                examples_preds[i][k] = v[i]
            else:
                examples_preds[i][k] = v[i].view(-1)
    return examples_preds


def eval_epochs_done_callback(global_vars,
                              input_json_files,
                              schema_json_file,
                              prediction_dir,
                              data_dir,
                              eval_dataset,
                              output_metric_file):
    
    pred_utils.write_predictions_to_file(global_vars['predictions'],
                                         input_json_files,
                                         schema_json_file,
                                         prediction_dir)

    metrics = evaluate(prediction_dir,
                       data_dir,
                       eval_dataset,
                       output_metric_file)
    return metrics


def evaluate(prediction_dir,
             data_dir,
             eval_dataset,
             output_metric_file):


    in_domain_services = get_in_domain_services(
        os.path.join(data_dir, eval_dataset, "schema.json"),
        os.path.join(data_dir, "train", "schema.json"))

    with open(os.path.join(data_dir, eval_dataset, "schema.json")) as f:
        eval_services = {}
        list_services = json.load(f)
        for service in list_services:
            eval_services[service["service_name"]] = service

    dataset_ref = get_dataset_as_dict(os.path.join(data_dir, eval_dataset, "dialogues_*.json"))
    dataset_hyp = get_dataset_as_dict(os.path.join(prediction_dir, "*.json"))
    
    all_metric_aggregate, _ = get_metrics(dataset_ref, dataset_hyp, eval_services, in_domain_services)
    logging.info(f'Dialog metrics: {all_metric_aggregate[ALL_SERVICES]}')

    # Write the aggregated metrics values.
    with open(output_metric_file, "w") as f:
        json.dump(all_metric_aggregate, f, indent=2, separators=(",", ": "), sort_keys=True)
    # Write the per-frame metrics values with the corrresponding dialogue frames.
    with open(os.path.join(prediction_dir, PER_FRAME_OUTPUT_FILENAME), "w") as f:
        json.dump(dataset_hyp, f, indent=2, separators=(",", ": "))
    return all_metric_aggregate




def get_average_and_joint_goal_accuracy(cat_slot_correctness,
                             noncat_slot_correctness,
                             cat_batch_ids,
                             noncat_batch_ids,
                             active_cat_slots,
                             active_noncat_slots):

    cat_batch = collections.Counter(cat_batch_ids)
    noncat_batch = collections.Counter(noncat_batch_ids)

    batch_ids = sorted(cat_batch.keys() | noncat_batch.keys())

    metrics = []

    cat_start_ind = 0
    noncat_start_ind = 0
    for id in batch_ids:
        frame_slots_correctness = []
        is_active = []
        is_cat = []
        if id in cat_batch:
            num_cat_slots_in_frame = cat_batch[id]
            cat_end_ind = cat_start_ind + num_cat_slots_in_frame
            frame_slots_correctness.extend(cat_slot_correctness[cat_start_ind:cat_end_ind])
            
            is_active.extend(active_cat_slots[cat_start_ind:cat_end_ind])
            is_cat.extend([True] * num_cat_slots_in_frame)
            cat_start_ind = cat_end_ind
        if id in noncat_batch:
            num_noncat_slots_in_frame = noncat_batch[id]
            noncat_end_ind = noncat_start_ind + num_noncat_slots_in_frame
            frame_slots_correctness.extend(noncat_slot_correctness[noncat_start_ind:noncat_end_ind])
            
            is_active.extend(active_noncat_slots[noncat_start_ind:noncat_end_ind])
            is_cat.extend([False] * num_noncat_slots_in_frame)
            noncat_start_ind = noncat_end_ind
 
        frame_metrics_dict = _get_average_and_joint_goal_accuracy(frame_slots_correctness, is_active, is_cat)
        metrics.append(frame_metrics_dict)
        
    return metrics



def get_noncat_slot_value_match_score(
    slot_idx,
    noncat_slot_value_start_labels,
    noncat_slot_value_end_labels,
    noncat_slot_value_start_preds,
    noncat_slot_value_end_preds,
    start_char_idxs,
    end_char_idxs,
    batch_ids,
    user_utterances,
    system_utterances):
    
    str_true = _get_noncat_slot_value(
        slot_idx,
        noncat_slot_value_start_labels,
        noncat_slot_value_end_labels,
        start_char_idxs,
        end_char_idxs,
        batch_ids,
        user_utterances,
        system_utterances)

    str_pred = _get_noncat_slot_value(
        slot_idx,
        noncat_slot_value_start_preds,
        noncat_slot_value_end_preds,
        start_char_idxs,
        end_char_idxs,
        batch_ids,
        user_utterances,
        system_utterances)

    if str_true is None:
        if str_pred is None:
            # if the slot value was mentioned in the previous utterances of the dialogue
            # that are not part of the current turn
            score = 1  # true and prediction don't modify previously set slot value
        else:
            score = 0  # preds incorrectly modifyes previously set slot value
    else:
        score = fuzz.token_sort_ratio(str_true, str_pred) / 100.0

    return score



def _get_noncat_slot_value(
    slot_idx,
    noncat_slot_value_start,
    noncat_slot_value_end,
    start_char_idxs,
    end_char_idxs,
    batch_ids,
    user_utterances,
    system_utterances):
    tok_start_idx = noncat_slot_value_start[slot_idx]
    tok_end_idx = noncat_slot_value_end[slot_idx]
    ch_start_idx = start_char_idxs[batch_ids[slot_idx]][tok_start_idx]
    ch_end_idx = end_char_idxs[batch_ids[slot_idx]][tok_end_idx]

    if ch_start_idx < 0 and ch_end_idx < 0:
        # Add span from the system utterance
        print('system utterance required')
        slot_value = None
        # slot_values[slot] = (
        #     system_utterance[-ch_start_idx - 1:-ch_end_idx])
    elif ch_start_idx > 0 and ch_end_idx > 0:
        # Add span from the user utterance
        slot_value = user_utterances[batch_ids[slot_idx]][ch_start_idx - 1 : ch_end_idx]
    else:
        slot_value = None
    return slot_value

def get_batch_ids(slot_status_mask):
    # determine batch_id slot active slot is associated with
    # it's needed to get the corresponing user utterance correctly
    splitted_mask = torch.split(slot_status_mask, 1)
    splitted_mask = [i * x for i, x in enumerate(splitted_mask)]
    utterance_batch_ids = [i * x.type(torch.int) for i, x in enumerate(splitted_mask)]
    utterance_batch_ids = torch.cat(utterance_batch_ids)[slot_status_mask]
    return utterance_batch_ids


# def get_joint_accuracy(slot_status_mask, slot_correctness_list):
#     batch_ids = tensor2list(get_batch_ids(slot_status_mask))

#     joint_accuracy = {}
#     start_idx = 0
#     for k, v in sorted(collections.Counter(batch_ids).items()):
#         joint_accuracy[k] = np.prod(slot_correctness_list[start_idx : start_idx + v])
#         start_idx += v
#     return joint_accuracy




def OLD_eval_epochs_done_callback(global_vars,
                              input_json_files,
                              schema_json_file,
                              prediction_dir):

    active_intent_labels = np.asarray(global_vars['active_intent_labels'])
    active_intent_preds = np.asarray(global_vars['active_intent_preds'])

    if len(active_intent_labels) > 0:
        active_intent_accuracy = sum(active_intent_labels == active_intent_preds) / len(active_intent_labels)
    else:
        active_intent_accuracy = 0
        
    req_slot_predictions = np.asarray(global_vars['req_slot_predictions'], dtype=int)
    requested_slot_status = np.asarray(global_vars['requested_slot_status'], dtype=int)
    req_slot_metrics = compute_f1(req_slot_predictions, requested_slot_status)

    frame_metrics = global_vars['average_and_joint_goal_accuracy']
    metric_collections = collections.defaultdict(lambda: collections.defaultdict(list))
    for frame_metric in frame_metrics:
        for metric_key, metric_value in frame_metric.items():
            if metric_value != NAN_VAL:
                metric_collections[ALL_SERVICES][metric_key].append(metric_value)



    all_metric_aggregate = {}
    for domain_key, domain_metric_vals in metric_collections.items():
        domain_metric_aggregate = {}
        for metric_key, value_list in domain_metric_vals.items():
            if value_list:
                # Metrics are macro-averaged across all frames.
                domain_metric_aggregate[metric_key] = float(np.mean(value_list))
            else:
                domain_metric_aggregate[metric_key] = metrics.NAN_VAL
        all_metric_aggregate[domain_key] = domain_metric_aggregate

    all_metric_aggregate[ALL_SERVICES][ACTIVE_INTENT_ACCURACY] = active_intent_accuracy
    all_metric_aggregate[ALL_SERVICES][REQUESTED_SLOTS_RECALL] = req_slot_metrics.recall
    all_metric_aggregate[ALL_SERVICES][REQUESTED_SLOTS_PRECISION] = req_slot_metrics.precision
    all_metric_aggregate[ALL_SERVICES][REQUESTED_SLOTS_F1] = req_slot_metrics.f1


    # correctness_cat_slots = np.asarray(global_vars['cat_slot_correctness'], dtype=int)
    # joint_acc, turn_acc = \
    #     evaluate_metrics(global_vars['comp_res'],
    #                      global_vars['gating_labels'],
    #                      global_vars['gating_preds'],
    #                      data_desc.gating_dict["ptr"])

    # gating_comp_flatten = (np.asarray(global_vars['gating_labels']) == np.asarray(global_vars['gating_preds'])).ravel()
    # gating_acc = np.sum(gating_comp_flatten) / len(gating_comp_flatten)

    # cat_slot_correctness = np.asarray(global_vars['cat_slot_correctness'])
    # noncat_slot_correctness = np.asarray(global_vars['noncat_slot_correctness'])

    # average_cat_accuracy = np.mean(cat_slot_correctness)
    # joint_cat_accuracy = np.mean(np.asarray(global_vars['joint_cat_accuracy'], dtype=int))

    # # average_noncat_accuracy = np.mean(noncat_slot_correctness)
    # # average_goal_accuracy = np.mean(np.concatenate((cat_slot_correctness, noncat_slot_correctness)))

    # metrics = {
    #     'all_services': {
    #         # Active intent accuracy
    #         "active_intent_accuracy": active_intent_accuracy,
    #         "average_cat_accuracy": average_cat_accuracy,
    #         # "average_goal_accuracy": average_goal_accuracy,
    #         # "average_noncat_accuracy": average_noncat_accuracy,
    #         "joint_cat_accuracy": joint_cat_accuracy,
    #         # "joint_goal_accuracy": 0.4904726693494299,
    #         # "joint_noncat_accuracy": 0.6226867035546613,
    #         # Slot tagging F1
    #         "requested_slots_f1": req_slot_metrics.f1,
    #         "requested_slots_precision": req_slot_metrics.precision,
    #         "requested_slots_recall": req_slot_metrics.recall,
    #         # Average goal accuracy
    #     }
    # }

    print('\n' + '#' * 50)
    for k, v in all_metric_aggregate[ALL_SERVICES].items():
        print(f'{k}: {v}')
    print('#' * 50 + '\n')

    # # active_intent_acc = metrics.get_active_intent_accuracy(
    # #         frame_ref, frame_hyp)
    # # slot_tagging_f1_scores = metrics.get_slot_tagging_f1(
    # #     frame_ref, frame_hyp, turn_ref["utterance"], service)
    # # requested_slots_f1_scores = metrics.get_requested_slots_f1(
    # #     frame_ref, frame_hyp)
    # # goal_accuracy_dict = metrics.get_average_and_joint_goal_accuracy(
    # #     frame_ref, frame_hyp, service)

    return all_metric_aggregate


def _get_average_and_joint_goal_accuracy(list_acc, slot_active, slot_cat):
    """Get average and joint goal accuracies of a frame.

  Args:
    frame_ref: single semantic frame from reference (ground truth) file.
    frame_hyp: single semantic frame from hypothesis (prediction) file.
    service: a service data structure in the schema. We use it to obtain the
      list of slots in the service and infer whether a slot is categorical.

  Returns:
    goal_acc: a dict whose values are average / joint
        all-goal / categorical-goal / non-categorical-goal accuracies.
  """
    goal_acc = {}

    # (4) Average goal accuracy.
    active_acc = [acc for acc, active in zip(list_acc, slot_active) if active]
    goal_acc[AVERAGE_GOAL_ACCURACY] = np.mean(active_acc) if active_acc else NAN_VAL
    # (4-a) categorical.
    active_cat_acc = [acc for acc, active, cat in zip(list_acc, slot_active, slot_cat) if active and cat]
    goal_acc[AVERAGE_CAT_ACCURACY] = np.mean(active_cat_acc) if active_cat_acc else NAN_VAL
    # (4-b) non-categorical.
    active_noncat_acc = [acc for acc, active, cat in zip(list_acc, slot_active, slot_cat) if active and not cat]
    goal_acc[AVERAGE_NONCAT_ACCURACY] = np.mean(active_noncat_acc) if active_noncat_acc else NAN_VAL

    # (5) Joint goal accuracy.
    goal_acc[JOINT_GOAL_ACCURACY] = np.prod(list_acc) if list_acc else NAN_VAL
    # (5-a) categorical.
    cat_acc = [acc for acc, cat in zip(list_acc, slot_cat) if cat]
    goal_acc[JOINT_CAT_ACCURACY] = np.prod(cat_acc) if cat_acc else NAN_VAL
    # (5-b) non-categorical.
    noncat_acc = [acc for acc, cat in zip(list_acc, slot_cat) if not cat]
    goal_acc[JOINT_NONCAT_ACCURACY] = np.prod(noncat_acc) if noncat_acc else NAN_VAL

    return goal_acc


F1Scores = collections.namedtuple("F1Scores", ["f1", "precision", "recall"])


def compute_f1(predictions, labels):
    """Compute F1 score from labels (grouth truth) and predictions.

  Args:
    predictions: numpy array of predictions
    labels: numpy array of labels

  Returns:
    A F1Scores object containing F1, precision, and recall scores.
  """
    true = sum(labels)
    positive = sum(predictions)
    true_positive = sum(predictions & labels)

    precision = float(true_positive) / positive if positive else 1.0
    recall = float(true_positive) / true if true else 1.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:  # The F1-score is defined to be 0 if both precision and recall are 0.
        f1 = 0.0

    return F1Scores(f1=f1, precision=precision, recall=recall)
