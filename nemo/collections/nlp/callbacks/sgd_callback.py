# Copyright (c) 2019 NVIDIA Corporation

import collections
import json
import os

import numpy as np
import torch
from fuzzywuzzy import fuzz

import nemo.collections.nlp.data.datasets.sgd_dataset.prediction_utils as pred_utils
from nemo import logging
from nemo.collections.nlp.data.datasets.sgd_dataset.evaluate import *

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']

# REQ_SLOT_THRESHOLD = 0.5
# F1Scores = collections.namedtuple("F1Scores", ["f1", "precision", "recall"])

# # Evaluation and other relevant metrics for DSTC8 Schema-guided DST.
# # (1) Active intent accuracy.
# ACTIVE_INTENT_ACCURACY = "active_intent_accuracy"
# # (2) Slot tagging F1.
# SLOT_TAGGING_F1 = "slot_tagging_f1"
# SLOT_TAGGING_PRECISION = "slot_tagging_precision"
# SLOT_TAGGING_RECALL = "slot_tagging_recall"
# # (3) Requested slots F1.
# REQUESTED_SLOTS_F1 = "requested_slots_f1"
# REQUESTED_SLOTS_PRECISION = "requested_slots_precision"
# REQUESTED_SLOTS_RECALL = "requested_slots_recall"
# # (4) Average goal accuracy.
# AVERAGE_GOAL_ACCURACY = "average_goal_accuracy"
# AVERAGE_CAT_ACCURACY = "average_cat_accuracy"
# AVERAGE_NONCAT_ACCURACY = "average_noncat_accuracy"
# # (5) Joint goal accuracy.
# JOINT_GOAL_ACCURACY = "joint_goal_accuracy"
# JOINT_CAT_ACCURACY = "joint_cat_accuracy"
# JOINT_NONCAT_ACCURACY = "joint_noncat_accuracy"

# NAN_VAL = "NA"


def tensor2list(tensor):
    return tensor.detach().cpu().tolist()


def tensor2numpy(tensor):
    return tensor.cpu().numpy()


def eval_iter_callback(tensors, global_vars):
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
    softmax = torch.nn.Softmax(dim=2)
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
    max_span_index = torch.argmax(total_scores.view(-1, max_num_noncat_slots, max_num_tokens ** 2), axis=-1)
    span_start_index = torch.div(max_span_index, max_num_tokens)
    span_end_index = torch.fmod(max_span_index, max_num_tokens)

    predictions['noncat_slot_start'] = span_start_index
    predictions['noncat_slot_end'] = span_end_index

    # Add inverse alignments.
    predictions['noncat_alignment_start'] = output['start_char_idx']
    predictions['noncat_alignment_end'] = output['end_char_idx']

    global_vars['predictions'].extend(combine_predictions_in_example(predictions, batch_size))


def combine_predictions_in_example(predictions, batch_size):
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


def eval_epochs_done_callback(
    global_vars, input_json_files, schema_json_file, prediction_dir, data_dir, eval_dataset, output_metric_file
):

    pred_utils.write_predictions_to_file(
        global_vars['predictions'], input_json_files, schema_json_file, prediction_dir
    )

    metrics = evaluate(prediction_dir, data_dir, eval_dataset, output_metric_file)
    return metrics


def evaluate(prediction_dir, data_dir, eval_dataset, output_metric_file):

    in_domain_services = get_in_domain_services(
        os.path.join(data_dir, eval_dataset, "schema.json"), os.path.join(data_dir, "train", "schema.json")
    )

    with open(os.path.join(data_dir, eval_dataset, "schema.json")) as f:
        eval_services = {}
        list_services = json.load(f)
        for service in list_services:
            eval_services[service["service_name"]] = service

    dataset_ref = get_dataset_as_dict(os.path.join(data_dir, eval_dataset, "dialogues_*.json"))
    dataset_hyp = get_dataset_as_dict(os.path.join(prediction_dir, "*.json"))

    all_metric_aggregate, _ = get_metrics(dataset_ref, dataset_hyp, eval_services, in_domain_services)
    logging.info(f'Dialog metrics for {ALL_SERVICES}: {all_metric_aggregate[ALL_SERVICES]}')

    # Write the aggregated metrics values.
    with open(output_metric_file, "w") as f:
        json.dump(all_metric_aggregate, f, indent=2, separators=(",", ": "), sort_keys=True)
    # Write the per-frame metrics values with the corrresponding dialogue frames.
    with open(os.path.join(prediction_dir, PER_FRAME_OUTPUT_FILENAME), "w") as f:
        json.dump(dataset_hyp, f, indent=2, separators=(",", ": "))
    return all_metric_aggregate[ALL_SERVICES]
