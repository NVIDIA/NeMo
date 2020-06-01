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

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst
"""

import json
import os

import torch

import nemo.collections.nlp.data.datasets.sgd_dataset.prediction_utils as pred_utils
from nemo import logging
from nemo.collections.nlp.data.datasets.sgd_dataset.data_processor import SGDDataProcessor
from nemo.collections.nlp.data.datasets.sgd_dataset.evaluate import (
    ALL_SERVICES,
    PER_FRAME_OUTPUT_FILENAME,
    SEEN_SERVICES,
    UNSEEN_SERVICES,
    get_dataset_as_dict,
    get_in_domain_services,
    get_metrics,
)

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']


def tensor2list(tensor):
    return tensor.detach().cpu().tolist()


def get_str_example_id(eval_dataset, ids_to_service_names_dict, example_id_num):
    def format_turn_id(ex_id_num):
        dialog_id_1, dialog_id_2, turn_id, service_id = ex_id_num
        return "{}-{}_{:05d}-{:02d}-{}".format(
            eval_dataset, dialog_id_1, dialog_id_2, turn_id, ids_to_service_names_dict[service_id]
        )

    return list(map(format_turn_id, tensor2list(example_id_num)))


def eval_iter_callback(tensors, global_vars, schema_processor, eval_dataset):
    if 'predictions' not in global_vars:
        global_vars['predictions'] = []

    output = {}
    for k, v in tensors.items():
        ind = k.find('~~~')
        if ind != -1:
            output[k[:ind]] = torch.cat(v)

    predictions = {}
    ids_to_service_names_dict = schema_processor.get_ids_to_service_names_dict()
    predictions['example_id'] = get_str_example_id(eval_dataset, ids_to_service_names_dict, output['example_id_num'])

    predictions['service_id'] = output['service_id']
    predictions['is_real_example'] = output['is_real_example']

    # Scores are output for each intent.
    # Note that the intent indices are shifted by 1 to account for NONE intent.
    predictions['intent_status'] = torch.argmax(output['logit_intent_status'], -1)

    # Scores are output for each requested slot.
    predictions['req_slot_status'] = torch.nn.Sigmoid()(output['logit_req_slot_status'])

    # For categorical slots, the status of each slot and the predicted value are output.
    cat_slot_status_dist = torch.nn.Softmax(dim=-1)(output['logit_cat_slot_status'])
    cat_slot_value_dist = torch.nn.Softmax(dim=-1)(output['logit_cat_slot_value'])

    predictions['cat_slot_status'] = torch.argmax(output['logit_cat_slot_status'], axis=-1)
    predictions['cat_slot_status_p'] = torch.max(cat_slot_status_dist, axis=-1)[0]
    predictions['cat_slot_value'] = torch.argmax(output['logit_cat_slot_value'], axis=-1)
    predictions['cat_slot_value_p'] = torch.max(cat_slot_value_dist, axis=-1)[0]

    # For non-categorical slots, the status of each slot and the indices for spans are output.
    noncat_slot_status_dist = torch.nn.Softmax(dim=-1)(output['logit_noncat_slot_status'])

    predictions['noncat_slot_status'] = torch.argmax(output['logit_noncat_slot_status'], axis=-1)
    predictions['noncat_slot_status_p'] = torch.max(noncat_slot_status_dist, axis=-1)[0]

    softmax = torch.nn.Softmax(dim=-1)
    start_scores = softmax(output['logit_noncat_slot_start'])
    end_scores = softmax(output['logit_noncat_slot_end'])

    batch_size, max_num_noncat_slots, max_num_tokens = end_scores.size()
    # Find the span with the maximum sum of scores for start and end indices.
    total_scores = torch.unsqueeze(start_scores, axis=3) + torch.unsqueeze(end_scores, axis=2)
    # Mask out scores where start_index > end_index.
    # device = total_scores.device
    start_idx = torch.arange(max_num_tokens, device=total_scores.device).view(1, 1, -1, 1)
    end_idx = torch.arange(max_num_tokens, device=total_scores.device).view(1, 1, 1, -1)
    invalid_index_mask = (start_idx > end_idx).repeat(batch_size, max_num_noncat_slots, 1, 1)
    total_scores = torch.where(
        invalid_index_mask,
        torch.zeros(total_scores.size(), device=total_scores.device, dtype=total_scores.dtype),
        total_scores,
    )
    max_span_index = torch.argmax(total_scores.view(-1, max_num_noncat_slots, max_num_tokens ** 2), axis=-1)
    max_span_p = torch.max(total_scores.view(-1, max_num_noncat_slots, max_num_tokens ** 2), axis=-1)[0]
    predictions['noncat_slot_p'] = max_span_p

    span_start_index = torch.div(max_span_index, max_num_tokens)
    span_end_index = torch.fmod(max_span_index, max_num_tokens)

    predictions['noncat_slot_start'] = span_start_index
    predictions['noncat_slot_end'] = span_end_index

    # Add inverse alignments.
    predictions['noncat_alignment_start'] = output['start_char_idx']
    predictions['noncat_alignment_end'] = output['end_char_idx']

    # added for debugging
    predictions['cat_slot_status_GT'] = output['categorical_slot_status']
    predictions['noncat_slot_status_GT'] = output['noncategorical_slot_status']

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
    global_vars,
    task_name,
    eval_dataset,
    data_dir,
    prediction_dir,
    state_tracker,
    eval_debug,
    dialogues_processor,
    schema_emb_preprocessor,
    joint_acc_across_turn,
    no_fuzzy_match,
):
    # added for debugging
    in_domain_services = get_in_domain_services(
        os.path.join(data_dir, eval_dataset, "schema.json"), dialogues_processor.get_seen_services("train")
    )
    ##############
    # we'll write predictions to file in Dstc8/SGD format during evaluation callback
    prediction_dir = os.path.join(prediction_dir, 'predictions', 'pred_res_{}_{}'.format(eval_dataset, task_name))
    os.makedirs(prediction_dir, exist_ok=True)

    input_json_files = SGDDataProcessor.get_dialogue_files(data_dir, eval_dataset, task_name)
    pred_utils.write_predictions_to_file(
        global_vars['predictions'],
        input_json_files,
        prediction_dir,
        schemas=schema_emb_preprocessor.schemas,
        state_tracker=state_tracker,
        eval_debug=eval_debug,
        in_domain_services=in_domain_services,
    )
    metrics = evaluate(
        prediction_dir, data_dir, eval_dataset, in_domain_services, joint_acc_across_turn, no_fuzzy_match,
    )
    return metrics


def evaluate(prediction_dir, data_dir, eval_dataset, in_domain_services, joint_acc_across_turn, no_fuzzy_match):

    with open(os.path.join(data_dir, eval_dataset, "schema.json")) as f:
        eval_services = {}
        list_services = json.load(f)
        for service in list_services:
            eval_services[service["service_name"]] = service
        f.close()

    dataset_ref = get_dataset_as_dict(os.path.join(data_dir, eval_dataset, "dialogues_*.json"))
    dataset_hyp = get_dataset_as_dict(os.path.join(prediction_dir, "*.json"))

    all_metric_aggregate, _ = get_metrics(
        dataset_ref, dataset_hyp, eval_services, in_domain_services, joint_acc_across_turn, no_fuzzy_match
    )
    if SEEN_SERVICES in all_metric_aggregate:
        logging.info(f'Dialog metrics for {SEEN_SERVICES}  : {sorted(all_metric_aggregate[SEEN_SERVICES].items())}')
    if UNSEEN_SERVICES in all_metric_aggregate:
        logging.info(f'Dialog metrics for {UNSEEN_SERVICES}: {sorted(all_metric_aggregate[UNSEEN_SERVICES].items())}')
    if ALL_SERVICES in all_metric_aggregate:
        logging.info(f'Dialog metrics for {ALL_SERVICES}   : {sorted(all_metric_aggregate[ALL_SERVICES].items())}')

    # Write the per-frame metrics values with the corrresponding dialogue frames.
    with open(os.path.join(prediction_dir, PER_FRAME_OUTPUT_FILENAME), "w") as f:
        json.dump(dataset_hyp, f, indent=2, separators=(",", ": "))
        f.close()
    return all_metric_aggregate[ALL_SERVICES]
