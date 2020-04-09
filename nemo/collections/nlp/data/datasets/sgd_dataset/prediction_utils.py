# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Prediction and evaluation-related utility functions."""

import collections
import json
import os

import nemo
from nemo import logging
from nemo.collections.nlp.data.datasets.sgd_dataset import data_utils, schema

REQ_SLOT_THRESHOLD = 0.5

__all__ = ['get_predicted_dialog_baseline', 'write_predictions_to_file']


def get_predicted_dialog_ret_sys_act(dialog, all_predictions, schemas, eval_debug):
    """Update labels in a dialogue based on model predictions.
  Args:
    dialog: A json object containing dialogue whose labels are to be updated.
    all_predictions: A dict mapping prediction name to the predicted value. See
      SchemaGuidedDST class for the contents of this dict.
    schemas: A Schema object wrapping all the schemas for the dataset.
  Returns:
    A json object containing the dialogue with labels predicted by the model.
  """
    if eval_debug:
        logging.setLevel(10)
    # This approach retreives slot values from the history of system actions if slot is active but it can not find it in user utterance
    # Overwrite the labels in the turn with the predictions from the model. For
    # test set, these labels are missing from the data and hence they are added.
    dialog_id = dialog["dialogue_id"]
    # The slot values tracked for each service.
    all_slot_values = collections.defaultdict(dict)
    sys_prev_slots = collections.defaultdict(dict)
    for turn_idx, turn in enumerate(dialog["turns"]):
        if turn["speaker"] == "SYSTEM":
            for frame in turn["frames"]:
                for action in frame["actions"]:
                    if action["slot"] and len(action["values"]) > 0:
                        sys_prev_slots[frame["service"]][action["slot"]] = action["values"][0]
        elif turn["speaker"] == "USER":
            user_utterance = turn["utterance"]
            system_utterance = dialog["turns"][turn_idx - 1]["utterance"] if turn_idx else ""
            turn_id = "{:02d}".format(turn_idx)
            for frame in turn["frames"]:
                logging.debug("-----------------------------------New Frame----------------------------")
                logging.debug(f'sys:{system_utterance}, user:{user_utterance}')
                logging.debug("Slots - Ground Truth:", frame['slots'])
                logging.debug("Frame - Ground Trurh:", frame['state'], "\n")

                predictions = all_predictions[(dialog_id, turn_id, frame["service"])]
                slot_values = all_slot_values[frame["service"]]
                service_schema = schemas.get_service_schema(frame["service"])

                # Remove the slot spans and state if present.
                frame.pop("slots", None)
                frame.pop("state", None)

                # The baseline model doesn't predict slot spans. Only state predictions are added.
                state = {}

                # Add prediction for active intent. Offset is subtracted to account for NONE intent.
                active_intent_id = predictions["intent_status"]
                state["active_intent"] = (
                    service_schema.get_intent_from_id(active_intent_id - 1) if active_intent_id else "NONE"
                )

                # Add prediction for requested slots.
                requested_slots = []
                for slot_idx, slot in enumerate(service_schema.slots):
                    if predictions["req_slot_status"][slot_idx] > REQ_SLOT_THRESHOLD:
                        requested_slots.append(slot)
                state["requested_slots"] = requested_slots

                # Add prediction for user goal (slot values).
                # Categorical slots.
                categorical_slots_dict = {}
                predictions["cat_slot_status_p"] = predictions["cat_slot_status_p"].cpu().numpy()
                predictions["cat_slot_status"] = predictions["cat_slot_status"].cpu().numpy()

                for slot_idx, slot in enumerate(service_schema.categorical_slots):
                    categorical_slots_dict[slot] = (
                        predictions["cat_slot_status"][slot_idx],
                        predictions["cat_slot_status_p"][slot_idx],
                        service_schema.get_categorical_slot_values(slot)[predictions["cat_slot_value"][slot_idx]],
                    )

                    slot_status = predictions["cat_slot_status"][slot_idx]
                    if slot_status == data_utils.STATUS_DONTCARE:
                        slot_values[slot] = data_utils.STR_DONTCARE
                    elif slot_status == data_utils.STATUS_ACTIVE:
                        value_idx = predictions["cat_slot_value"][slot_idx]
                        slot_values[slot] = service_schema.get_categorical_slot_values(slot)[value_idx]

                if eval_debug:
                    logging.debug(f"Predicted categorical_slots_dict: {categorical_slots_dict}")

                # Non-categorical slots.
                non_categorical_slots_dict = {}
                predictions["noncat_slot_status_p"] = predictions["noncat_slot_status_p"].cpu().numpy()
                predictions["noncat_slot_status"] = predictions["noncat_slot_status"].cpu().numpy()
                predictions["noncat_slot_p"] = predictions["noncat_slot_p"].cpu().numpy()

                predictions["noncat_alignment_start"] = predictions["noncat_alignment_start"].cpu().numpy()
                predictions["noncat_alignment_end"] = predictions["noncat_alignment_end"].cpu().numpy()

                for slot_idx, slot in enumerate(service_schema.non_categorical_slots):
                    tok_start_idx = predictions["noncat_slot_start"][slot_idx]
                    tok_end_idx = predictions["noncat_slot_end"][slot_idx]
                    ch_start_idx = predictions["noncat_alignment_start"][tok_start_idx]
                    ch_end_idx = predictions["noncat_alignment_end"][tok_end_idx]

                    non_categorical_slots_dict[slot] = (
                        predictions["noncat_slot_status"][slot_idx],
                        predictions["noncat_slot_status_p"][slot_idx],
                        predictions["noncat_slot_p"][slot_idx],
                        (ch_start_idx, ch_end_idx),
                        user_utterance[ch_start_idx - 1 : ch_end_idx]
                        if (ch_start_idx > 0 and ch_end_idx > 0)
                        else system_utterance[-ch_start_idx - 1 : -ch_end_idx],
                    )

                    slot_status = predictions["noncat_slot_status"][slot_idx]
                    if slot_status == data_utils.STATUS_DONTCARE:
                        slot_values[slot] = data_utils.STR_DONTCARE
                    elif slot_status == data_utils.STATUS_ACTIVE:
                        tok_start_idx = predictions["noncat_slot_start"][slot_idx]
                        tok_end_idx = predictions["noncat_slot_end"][slot_idx]
                        ch_start_idx = predictions["noncat_alignment_start"][tok_start_idx]
                        ch_end_idx = predictions["noncat_alignment_end"][tok_end_idx]
                        if ch_start_idx > 0 and ch_end_idx > 0:
                            # Add span from the user utterance.
                            slot_values[slot] = user_utterance[ch_start_idx - 1 : ch_end_idx]
                        else:
                            if slot in sys_prev_slots[frame["service"]]:
                                logging.debug("Sys Ret:", sys_prev_slots[frame["service"]][slot])
                                slot_values[slot] = sys_prev_slots[frame["service"]][slot]

                logging.debug(f"Predicted non_categorical_slots_dict: {non_categorical_slots_dict}")

                # Create a new dict to avoid overwriting the state in previous turns
                # because of use of same objects.
                state["slot_values"] = {s: [v] for s, v in slot_values.items()}
                frame["state"] = state
                logging.debug("Predicted state:", state)
    return dialog


def get_predicted_dialog_baseline(dialog, all_predictions, schemas):
    """Update labels in a dialogue based on model predictions.
  Args:
    dialog: A json object containing dialogue whose labels are to be updated.
    all_predictions: A dict mapping prediction name to the predicted value. See
      SchemaGuidedDST class for the contents of this dict.
    schemas: A Schema object wrapping all the schemas for the dataset.
  Returns:
    A json object containing the dialogue with labels predicted by the model.
  """
    # Overwrite the labels in the turn with the predictions from the model. For
    # test set, these labels are missing from the data and hence they are added.
    dialog_id = dialog["dialogue_id"]
    # The slot values tracked for each service.
    all_slot_values = collections.defaultdict(dict)
    for turn_idx, turn in enumerate(dialog["turns"]):
        if turn["speaker"] == "USER":
            user_utterance = turn["utterance"]
            system_utterance = dialog["turns"][turn_idx - 1]["utterance"] if turn_idx else ""
            turn_id = "{:02d}".format(turn_idx)
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

                # Add prediction for active intent. Offset is subtracted to account for
                # NONE intent.
                active_intent_id = predictions["intent_status"]
                state["active_intent"] = (
                    service_schema.get_intent_from_id(active_intent_id - 1) if active_intent_id else "NONE"
                )

                # Add prediction for requested slots.
                requested_slots = []
                for slot_idx, slot in enumerate(service_schema.slots):
                    if predictions["req_slot_status"][slot_idx] > REQ_SLOT_THRESHOLD:
                        requested_slots.append(slot)
                state["requested_slots"] = requested_slots

                # Add prediction for user goal (slot values).
                # Categorical slots.
                for slot_idx, slot in enumerate(service_schema.categorical_slots):
                    slot_status = predictions["cat_slot_status"][slot_idx]
                    if slot_status == data_utils.STATUS_DONTCARE:
                        slot_values[slot] = data_utils.STR_DONTCARE
                    elif slot_status == data_utils.STATUS_ACTIVE:
                        value_idx = predictions["cat_slot_value"][slot_idx]
                        slot_values[slot] = service_schema.get_categorical_slot_values(slot)[value_idx]
                # Non-categorical slots.
                for slot_idx, slot in enumerate(service_schema.non_categorical_slots):
                    slot_status = predictions["noncat_slot_status"][slot_idx]
                    if slot_status == data_utils.STATUS_DONTCARE:
                        slot_values[slot] = data_utils.STR_DONTCARE
                    elif slot_status == data_utils.STATUS_ACTIVE:
                        tok_start_idx = predictions["noncat_slot_start"][slot_idx]
                        tok_end_idx = predictions["noncat_slot_end"][slot_idx]
                        ch_start_idx = predictions["noncat_alignment_start"][tok_start_idx]
                        ch_end_idx = predictions["noncat_alignment_end"][tok_end_idx]
                        if ch_start_idx < 0 and ch_end_idx < 0:
                            # Add span from the system utterance.
                            slot_values[slot] = system_utterance[-ch_start_idx - 1 : -ch_end_idx]
                        elif ch_start_idx > 0 and ch_end_idx > 0:
                            # Add span from the user utterance.
                            slot_values[slot] = user_utterance[ch_start_idx - 1 : ch_end_idx]
                # Create a new dict to avoid overwriting the state in previous turns
                # because of use of same objects.
                state["slot_values"] = {s: [v] for s, v in slot_values.items()}
                frame["state"] = state
    return dialog


def write_predictions_to_file(predictions, input_json_files, schema_json_file, output_dir, state_tracker, eval_debug):
    """Write the predicted dialogues as json files.

  Args:
    predictions: An iterator containing model predictions. This is the output of
      the predict method in the estimator.
    input_json_files: A list of json paths containing the dialogues to run
      inference on.
    schema_json_file: Path for the json file containing the schemas.
    output_dir: The directory where output json files will be created.
  """
    nemo.logging.info(f"Writing predictions to {output_dir} started.")
    schemas = schema.Schema(schema_json_file)
    # Index all predictions.
    all_predictions = {}
    for idx, prediction in enumerate(predictions):
        if not prediction["is_real_example"]:
            continue
        _, dialog_id, turn_id, service_name = prediction['example_id'].split('-')
        all_predictions[(dialog_id, turn_id, service_name)] = prediction
    nemo.logging.info(f'Predictions for {idx} examples are getting processed.')

    # Read each input file and write its predictions.
    for input_file_path in input_json_files:
        with open(input_file_path) as f:
            dialogs = json.load(f)
            nemo.logging.info(f'{input_file_path} file is loaded')
            pred_dialogs = []
            for d in dialogs:
                if state_tracker == 'baseline':
                    pred_dialog = get_predicted_dialog_baseline(d, all_predictions, schemas)
                elif state_tracker == 'ret_sys_act':
                    pred_dialog = get_predicted_dialog_ret_sys_act(d, all_predictions, schemas, eval_debug)
                else:
                    raise ValueError(f"tracker_mode {state_tracker} is not defined.")
                pred_dialogs.append(pred_dialog)
            f.close()
        input_file_name = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_dir, input_file_name)
        with open(output_file_path, "w") as f:
            json.dump(pred_dialogs, f, indent=2, separators=(",", ": "), sort_keys=True)
            f.close()
