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
Prediction and evaluation-related utility functions.
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst/baseline/pred_utils.py
"""

import json
import os
from collections import OrderedDict, defaultdict

from nemo import logging
from nemo.collections.nlp.data.datasets.sgd_dataset.input_example import (
    STATUS_ACTIVE,
    STATUS_DONTCARE,
    STATUS_OFF,
    STR_DONTCARE,
)

REQ_SLOT_THRESHOLD = 0.5
MIN_SLOT_RELATION = 0.1

# MIN_SLOT_RELATION specifes the minimum number of relations between two slots in the training dialogues to get considered for carry-over
# MIN_SLOT_RELATION = 25

__all__ = ['get_predicted_dialog', 'write_predictions_to_file']


def get_carryover_value(
    slot,
    cur_usr_frame,
    all_slot_values,
    slots_relation_list,
    frame_service_prev,
    sys_slots_last,
    sys_slots_agg,
    sys_rets,
):
    extracted_value = None
    if slot in sys_slots_agg[cur_usr_frame["service"]]:
        extracted_value = sys_slots_agg[cur_usr_frame["service"]][slot]
        # sys_rets[slot] = extracted_value

    elif (cur_usr_frame["service"], slot) in slots_relation_list:
        # return extracted_value
        cands_list = slots_relation_list[(cur_usr_frame["service"], slot)]
        for dmn, slt, freq in cands_list:
            if freq < MIN_SLOT_RELATION:
                continue
            if dmn in all_slot_values and slt in all_slot_values[dmn]:
                extracted_value = all_slot_values[dmn][slt]
            if dmn in sys_slots_agg and slt in sys_slots_agg[dmn]:
                extracted_value = sys_slots_agg[dmn][slt]
            if dmn in sys_slots_last and slt in sys_slots_last[dmn]:
                extracted_value = sys_slots_last[dmn][slt]
    return extracted_value


def carry_over_slots(
    cur_usr_frame,
    all_slot_values,
    slots_relation_list,
    frame_service_prev,
    slot_values,
    sys_slots_agg,
    sys_slots_last,
):
    # return
    if frame_service_prev == "" or frame_service_prev == cur_usr_frame["service"]:
        return
    for (service_dest, slot_dest), cands_list in slots_relation_list.items():
        if service_dest != cur_usr_frame["service"]:
            continue
        for service_src, slot_src, freq in cands_list:
            if freq < MIN_SLOT_RELATION or service_src != frame_service_prev:
                continue

            if service_src in all_slot_values and slot_src in all_slot_values[service_src]:
                slot_values[slot_dest] = all_slot_values[service_src][slot_src]
            if service_src in sys_slots_agg and slot_src in sys_slots_agg[service_src]:
                slot_values[slot_dest] = sys_slots_agg[service_src][slot_src]
            if service_src in sys_slots_last and slot_src in sys_slots_last[service_src]:
                slot_values[slot_dest] = sys_slots_last[service_src][slot_src]


def set_cat_slot(
    predictions_status,
    predictions_value,
    cat_slots,
    cat_slot_values,
    sys_rets,
    cat_value_thresh,
    probavg,
    frame,
    all_slot_values,
    slots_relation_list,
    frame_service_prev,
    sys_slots_agg,
    sys_slots_last,
):
    """
    write predicted slot and values into out_dict 
    """
    out_dict = {}
    for slot_idx, slot in enumerate(cat_slots):
        slot_status = predictions_status[slot_idx][0]["cat_slot_status"]
        slot_status_active_prob = predictions_status[slot_idx][0]["cat_slot_status_p"][STATUS_ACTIVE].item()
        tmp = predictions_value[slot_idx]
        value_idx = max(tmp, key=lambda k: tmp[k]['cat_slot_value_status'][0].item())
        value_prob = max([v['cat_slot_value_status'][0].item() for k, v in predictions_value[slot_idx].items()])
        if slot_status == STATUS_DONTCARE:
            out_dict[slot] = STR_DONTCARE
        elif slot_status == STATUS_ACTIVE:
            # cross over:
            carryover_value = get_carryover_value(
                slot,
                frame,
                all_slot_values,
                slots_relation_list,
                frame_service_prev,
                sys_slots_last,
                sys_slots_agg,
                sys_rets,
            )
            if carryover_value:
                out_dict[slot] = carryover_value
                continue
            if not probavg:
                if value_prob > cat_value_thresh:
                    out_dict[slot] = cat_slot_values[slot][value_idx]
                elif sys_slots_agg and slot in sys_slots_agg:
                    out_dict[slot] = sys_slots_agg[slot]
            else:
                if (slot_status_active_prob + value_prob) / 2 > cat_value_thresh:
                    out_dict[slot] = cat_slot_values[slot][value_idx]
                elif sys_slots_agg and slot in sys_slots_agg:
                    # retrieval
                    out_dict[slot] = sys_slots_agg[slot]
        else:
            if probavg and (slot_status_active_prob + value_prob) / 2 > cat_value_thresh:
                out_dict[slot] = cat_slot_values[slot][value_idx]

    return out_dict


def set_noncat_slot(
    predictions_status,
    predictions_value,
    non_cat_slots,
    user_utterance,
    non_cat_value_thresh,
    frame,
    all_slot_values,
    slots_relation_list,
    frame_service_prev,
    sys_slots_agg,
    sys_slots_last,
    sys_rets,
):
    """
    write predicted slot and values into out_dict 
    """
    out_dict = {}
    for slot_idx, slot in enumerate(non_cat_slots):
        slot_status = predictions_status[slot_idx][0]["noncat_slot_status"]
        slot_status_active_prob = predictions_status[slot_idx][0]["noncat_slot_status_p"][STATUS_ACTIVE].item()
        if slot_status == STATUS_DONTCARE:
            out_dict[slot] = STR_DONTCARE
        elif slot_status == STATUS_ACTIVE:
            value_prob = predictions_value[slot_idx][0]["noncat_slot_p"]
            tok_start_idx = predictions_value[slot_idx][0]["noncat_slot_start"]
            tok_end_idx = predictions_value[slot_idx][0]["noncat_slot_end"]
            ch_start_idx = predictions_value[slot_idx][0]["noncat_alignment_start"][tok_start_idx]
            ch_end_idx = predictions_value[slot_idx][0]["noncat_alignment_end"][tok_end_idx]
            if ch_start_idx > 0 and ch_end_idx > 0 and value_prob > non_cat_value_thresh:
                # Add span from the user utterance.
                out_dict[slot] = user_utterance[ch_start_idx - 1 : ch_end_idx]
            else:
                carryover_value = get_carryover_value(
                    slot,
                    frame,
                    all_slot_values,
                    slots_relation_list,
                    frame_service_prev,
                    sys_slots_last,
                    sys_slots_agg,
                    sys_rets,
                )
                if carryover_value:
                    out_dict[slot] = carryover_value
    return out_dict


def get_predicted_dialog(
    dialog, all_predictions, schemas, state_tracker, cat_value_thresh, non_cat_value_thresh, probavg
):
    # Overwrite the labels in the turn with the predictions from the model. For
    # test set, these labels are missing from the data and hence they are added.
    dialog_id = dialog["dialogue_id"]
    if state_tracker == "baseline":
        sys_slots_agg = {}
    else:
        sys_slots_agg = defaultdict(OrderedDict)
        all_slot_values = defaultdict(OrderedDict)
        sys_slots_last = defaultdict(OrderedDict)

    sys_rets = OrderedDict()
    frame_service_prev = ""
    slots_relation_list = schemas._slots_relation_list

    for turn_idx, turn in enumerate(dialog["turns"]):
        if turn["speaker"] == "SYSTEM" and state_tracker == 'nemotracker':
            sys_slots_last = defaultdict(OrderedDict)
            for frame in turn["frames"]:
                if frame["service"] not in sys_slots_agg:
                    sys_slots_agg[frame["service"]] = OrderedDict()
                if frame["service"] not in sys_slots_last:
                    sys_slots_last[frame["service"]] = OrderedDict()
                for action in frame["actions"]:
                    if action["slot"] and len(action["values"]) > 0:
                        sys_slots_agg[frame["service"]][action["slot"]] = action["values"][0]
                        sys_slots_last[frame["service"]][action["slot"]] = action["values"][0]
        if turn["speaker"] == "USER":
            user_utterance = turn["utterance"]
            system_utterance = dialog["turns"][turn_idx - 1]["utterance"] if turn_idx else ""
            system_user_utterance = system_utterance + ' ' + user_utterance
            turn_id = "{:02d}".format(turn_idx)

            if len(turn["frames"]) == 2:
                if frame_service_prev != "" and turn["frames"][0]["service"] != frame_service_prev:
                    frame_tmp = turn["frames"][0]
                    turn["frames"][0] = turn["frames"][1]
                    turn["frames"][1] = frame_tmp

            for frame in turn["frames"]:

                predictions = all_predictions[(dialog_id, turn_id, frame["service"])]
                slot_values = all_slot_values[frame["service"]]
                service_schema = schemas.get_service_schema(frame["service"])
                # Remove the slot spans and state if present.
                frame.pop("slots", None)
                frame.pop("state", None)

                # The baseline model doesn't predict slot spans. Only state predictions
                # are added.
                state = {}

                # Add prediction for active intent. No Offset is subtracted since schema has now NONE intent at index 0
                state["active_intent"] = get_predicted_intent(
                    predictions=predictions[0], intents=service_schema.intents
                )
                # state["active_intent"] = "NONE"
                # Add prediction for requested slots.
                state["requested_slots"] = get_requested_slot(predictions=predictions[1], slots=service_schema.slots)
                # state["requested_slots"] = []

                # Add prediction for user goal (slot values).
                # Categorical slots.
                # cat_out_dict = set_cat_slot(predictions_status=predictions[2], predictions_value=predictions[3], cat_slots=service_schema.categorical_slots, cat_slot_values=service_schema.categorical_slot_values, sys_slots_agg=sys_slots_agg.get(frame["service"], None), cat_value_thresh=cat_value_thresh)
                cat_out_dict = set_cat_slot(
                    predictions_status=predictions[2],
                    predictions_value=predictions[3],
                    cat_slots=service_schema.categorical_slots,
                    cat_slot_values=service_schema.categorical_slot_values,
                    sys_slots_agg=sys_slots_agg,
                    cat_value_thresh=cat_value_thresh,
                    probavg=probavg,
                    frame=frame,
                    all_slot_values=all_slot_values,
                    slots_relation_list=slots_relation_list,
                    frame_service_prev=frame_service_prev,
                    sys_slots_last=sys_slots_last,
                    sys_rets=sys_rets
                )
                for k, v in cat_out_dict.items():
                    slot_values[k] = v

                # # Non-categorical slots.
                noncat_out_dict = set_noncat_slot(
                    predictions_status=predictions[4],
                    predictions_value=predictions[5],
                    non_cat_slots=service_schema.non_categorical_slots,
                    user_utterance=system_user_utterance,
                    sys_slots_agg=sys_slots_agg,
                    non_cat_value_thresh=non_cat_value_thresh,
                    frame=frame,
                    all_slot_values=all_slot_values,
                    slots_relation_list=slots_relation_list,
                    frame_service_prev=frame_service_prev,
                    sys_slots_last=sys_slots_last,
                    sys_rets=sys_rets
                )
                for k, v in noncat_out_dict.items():
                    slot_values[k] = v
                
                carry_over_slots(
                    frame,
                    all_slot_values,
                    slots_relation_list,
                    frame_service_prev,
                    slot_values,
                    sys_slots_agg,
                    sys_slots_last,
                )
                # Create a new dict to avoid overwriting the state in previous turns
                # because of use of same objects.
                state["slot_values"] = {s: [v] for s, v in slot_values.items()}
                frame["state"] = state
    return dialog


def get_predicted_intent(predictions, intents):
    """
    returns intent name with maximum score
    """
    assert len(predictions) == len(intents)
    active_intent_id = max(predictions, key=lambda k: predictions[k][0]['intent_status'])
    return intents[active_intent_id]


def get_requested_slot(predictions, slots):
    """
    returns list of slots which are predicted to be requested
    """
    active_indices = [k for k in predictions if predictions[k][0]["req_slot_status"] > REQ_SLOT_THRESHOLD]
    requested_slots = list(map(lambda k: slots[k], active_indices))
    return requested_slots


def write_predictions_to_file(
    predictions,
    input_json_files,
    output_dir,
    schemas,
    state_tracker,
    eval_debug,
    in_domain_services,
    cat_value_thresh,
    non_cat_value_thresh,
    probavg,
):
    """Write the predicted dialogues as json files.

  Args:
    predictions: An iterator containing model predictions. This is the output of
      the predict method in the estimator.
    input_json_files: A list of json paths containing the dialogues to run
      inference on.
    schemas: Schemas to all services in the dst dataset (train, dev and test splits).
    output_dir: The directory where output json files will be created.
  """
    logging.info(f"Writing predictions to {output_dir} started.")

    # Index all predictions.
    all_predictions = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for idx, prediction in enumerate(predictions):
        if not prediction["is_real_example"]:
            continue
        eval_dataset, dialog_id, turn_id, service_name, model_task, slot_intent_id, value_id = prediction[
            'example_id'
        ].split('-')
        all_predictions[(dialog_id, turn_id, service_name)][int(model_task)][int(slot_intent_id)][
            int(value_id)
        ] = prediction
    logging.info(f'Predictions for {idx} examples in {eval_dataset} dataset are getting processed.')

    # Read each input file and write its predictions.
    for input_file_path in input_json_files:
        with open(input_file_path) as f:
            dialogs = json.load(f)
            logging.debug(f'{input_file_path} file is loaded')
            pred_dialogs = []
            for d in dialogs:
                pred_dialog = get_predicted_dialog(
                    d, all_predictions, schemas, state_tracker, cat_value_thresh, non_cat_value_thresh, probavg
                )
                pred_dialogs.append(pred_dialog)
            f.close()
        input_file_name = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_dir, input_file_name)
        with open(output_file_path, "w") as f:
            json.dump(pred_dialogs, f, indent=2, separators=(",", ": "), sort_keys=True)
            f.close()
