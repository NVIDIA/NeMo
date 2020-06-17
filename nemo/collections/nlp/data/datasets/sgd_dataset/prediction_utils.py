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
    STATUS_CARRY,
    STATUS_DONTCARE,
    STATUS_OFF,
    STR_DONTCARE,
)

REQ_SLOT_THRESHOLD = 0.5

# MIN_SLOT_RELATION specifes the minimum number of relations between two slots in the training dialogues to get considered for carry-over
MIN_SLOT_RELATION = 0.1

__all__ = ['get_predicted_dialog_baseline', 'write_predictions_to_file']


def carry_over_slots(
    cur_usr_frame,
    all_slot_values,
    slots_relation_list,
    frame_service_prev,
    slot_values,
    sys_slots_agg,
    sys_slots_last,
):
    """This function searches the candidate list for cross-service cases to find and update the values for all the slots in the current predicted state
    It is called when state is predicted for a frame.
    Args:
        cur_usr_frame: the current frame of the user
        all_slot_values: dictionary of all the slots and their values extracted from the dialogue until the current turn for all services
        slots_relation_list: list of the candidates for carry-over for each (service, slot)
        frame_service_prev: the service of the last system's frame
        sys_slots_last: dictionary of all the slots and values mentioned in the last system utterance
        sys_slots_agg:  dictionary of all the slots and values mentioned in the all the system utterances until the current turn
        slot_values: dictionary of all the slots and their values extracted from the dialogue until the current turn for the current service
      Returns:
        the extracted value for the slot
    """

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
    """This function searches the previous system actions and also the candidate list for cross-service cases to find a value for a slot
    It is called when a value for a slot can not be found the last user utterance
    Args:
        slot: name of the slot to find and extract a value for
        cur_usr_frame: the current frame of the user
        frame_service_prev: the service of the last system's frame
        all_slot_values: dictionary of all the slots and their values extracted from the dialogue until the current turn for all services
        sys_slots_last: dictionary of all the slots and values mentioned in the last system utterance
        sys_slots_agg:  dictionary of all the slots and values mentioned in the all the system utterances until the current turn
        slots_relation_list: list of the candidates for carry-over for each (service, slot)
        sys_rets: list of the extracted slots and values from system utterances until the current turn, used for debugging
      Returns:
        the extracted value for the slot
    """
    extracted_value = None
    if slot in sys_slots_agg[cur_usr_frame["service"]]:
        extracted_value = sys_slots_agg[cur_usr_frame["service"]][slot]
        sys_rets[slot] = extracted_value
    elif (cur_usr_frame["service"], slot) in slots_relation_list:
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


def get_predicted_dialog_nemotracker(dialog, all_predictions, schemas, eval_debug, in_domain_services):
    """This is NeMo Tracker which would be enabled by passing "--tracker_model=nemotracker".
    It improves the performance significantly by employing carry-over mechanism for in-service and cross-service.

    * **In-service carry-over mechanism**: There are cases that the value for some slots are not mentioned in the last user utterrance, but in the previous system utterances or actions.\
    Therefore, whenever the status of a non-categorical slot is active but no value can be found in the user utterance, we search the list of slots and their values mentioned in the previous system actions to find a value for this slot.
    The most recent value is used as the value for the slot. It is called in-domain carry-over as it happens inside a service.

    * **Cross-service carry-over mechanism**: In multi-domain dialogues, switching between two services can happen in the dialogue. In such cases, there can be some values to get transfered to the new service automatically.
    For instance when user is reserving flight tickets for two persons, it can be assumed that number of people for hotel reservation should also be two. To handle such cases, when we process the dialogues, we also record the list of these carry-over between two services from the training data.
    A candidate list for each (service, slot) is produced which show the list possible carry-over for that slot. These lists are stored in a file along with the processed dialogues and would be read and used in the state tracker to carry values when switches happens from one service to another.
    Whenever we find a switch and have an active non-categorical slot without any value, we would try to use that candidate list to retrieve a value for that slot from other slots in other services in previous turns. The latest value is used if multiple values are found.

    Args:
        dialog: A json object containing dialogue whose labels are to be updated.
        all_predictions: A dict mapping prediction name to the predicted value.
        schemas: A Schema object wrapping all the schemas for the dataset.
        eval_debug: specifies if it is running in DEBUG mode, so to generate the error analysis outputs
        in_domain_services: list of the seen services
    Returns:
        A json object containing the dialogue with labels predicted by the model.
  """

    dialog_id = dialog["dialogue_id"]
    # The slot values tracked for each service.
    all_slot_values = defaultdict(OrderedDict)
    sys_slots_agg = defaultdict(OrderedDict)
    sys_slots_last = defaultdict(OrderedDict)

    sys_rets = OrderedDict()
    true_state_prev = OrderedDict()
    true_state = OrderedDict()
    frame_service_prev = ""
    slots_relation_list = schemas._slots_relation_list

    for turn_idx, turn in enumerate(dialog["turns"]):
        if turn["speaker"] == "SYSTEM":
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
        elif turn["speaker"] == "USER":
            user_utterance = turn["utterance"]
            system_utterance = dialog["turns"][turn_idx - 1]["utterance"] if turn_idx else ""
            turn_id = "{:02d}".format(turn_idx)
            if len(turn["frames"]) == 2:
                if frame_service_prev != "" and turn["frames"][0]["service"] != frame_service_prev:
                    frame_tmp = turn["frames"][0]
                    turn["frames"][0] = turn["frames"][1]
                    turn["frames"][1] = frame_tmp

            for frame in turn["frames"]:
                cat_slot_status_acc = 0
                cat_slot_status_num = 0
                noncat_slot_status_num = 0
                noncat_slot_status_acc = 0
                cat_slot_value_acc = 0
                cat_slot_value_num = 0
                noncat_slot_value_acc = 0
                noncat_slot_value_num = 0

                predictions = all_predictions[(dialog_id, turn_id, frame["service"])]
                slot_values = all_slot_values[frame["service"]]
                service_schema = schemas.get_service_schema(frame["service"])

                predictions["cat_slot_status_p"] = predictions["cat_slot_status_p"].cpu().numpy()
                predictions["cat_slot_status"] = predictions["cat_slot_status"].cpu().numpy()
                predictions["cat_slot_value"] = predictions["cat_slot_value"].cpu().numpy()
                predictions["cat_slot_value_p"] = predictions["cat_slot_value_p"].cpu().numpy()

                predictions["noncat_slot_status_p"] = predictions["noncat_slot_status_p"].cpu().numpy()
                predictions["noncat_slot_status"] = predictions["noncat_slot_status"].cpu().numpy()
                predictions["noncat_slot_p"] = predictions["noncat_slot_p"].cpu().numpy()

                predictions["noncat_alignment_start"] = predictions["noncat_alignment_start"].cpu().numpy()
                predictions["noncat_alignment_end"] = predictions["noncat_alignment_end"].cpu().numpy()
                predictions["cat_slot_status_GT"] = predictions["cat_slot_status_GT"].cpu().numpy()
                predictions["noncat_slot_status_GT"] = predictions["noncat_slot_status_GT"].cpu().numpy()

                # Remove the slot spans and state if present.
                true_state_prev = [] if len(true_state) == 0 else true_state["slot_values"]
                true_slots = frame.pop("slots", None)
                true_state = frame.pop("state", None)

                # The baseline model doesn't predict slot spans. Only state predictions
                # are added.
                state = OrderedDict()

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
                categorical_slots_dict = OrderedDict()
                non_categorical_slots_dict = OrderedDict()
                for slot_idx, slot in enumerate(service_schema.categorical_slots):
                    cat_slot_status_num += 1

                    slot_status = predictions["cat_slot_status"][slot_idx]
                    extracted_value = None
                    if slot_status == STATUS_DONTCARE:
                        extracted_value = STR_DONTCARE
                    elif slot_status == STATUS_ACTIVE:
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

                        if (
                            service_schema.get_categorical_slot_values(slot)[predictions["cat_slot_value"][slot_idx]]
                            != "#CARRYVALUE#"
                            or carryover_value is None
                        ):
                            value_idx = predictions["cat_slot_value"][slot_idx]
                            extracted_value = service_schema.get_categorical_slot_values(slot)[value_idx]
                        else:
                            extracted_value = carryover_value
                    elif slot_status == STATUS_CARRY:
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

                        if carryover_value is None:
                            extracted_value = None
                        else:
                            extracted_value = carryover_value

                    elif slot_status == STATUS_OFF:
                        extracted_value = None

                    if extracted_value is not None:
                        slot_values[slot] = extracted_value

                    # debugging info processing
                    if predictions["cat_slot_status_GT"][slot_idx] != predictions["cat_slot_status"][slot_idx] or (
                        predictions["cat_slot_status_GT"][slot_idx] == predictions["cat_slot_status"][slot_idx]
                        and predictions["cat_slot_status_GT"][slot_idx] != STATUS_OFF
                        and extracted_value not in true_state['slot_values'][slot]
                    ):
                        categorical_slots_dict[slot] = (
                            predictions["cat_slot_status_GT"][slot_idx],
                            predictions["cat_slot_status"][slot_idx],
                            predictions["cat_slot_status_p"][slot_idx],
                            service_schema.get_categorical_slot_values(slot)[predictions["cat_slot_value"][slot_idx]],
                            service_schema.get_categorical_slot_values(slot)[
                                predictions["cat_slot_value_GT"][slot_idx]
                            ],
                            extracted_value,
                            predictions["cat_slot_value_p"][slot_idx],
                        )

                    if predictions["cat_slot_status_GT"][slot_idx] == predictions["cat_slot_status"][slot_idx]:
                        cat_slot_status_acc += 1
                    if predictions["cat_slot_status_GT"][slot_idx] != STATUS_OFF:
                        cat_slot_value_num += 1
                        if extracted_value in true_state['slot_values'][slot]:
                            cat_slot_value_acc += 1
                    # debugging info processing ended

                for slot_idx, slot in enumerate(service_schema.non_categorical_slots):
                    noncat_slot_status_num += 1

                    tok_start_idx = predictions["noncat_slot_start"][slot_idx]
                    tok_end_idx = predictions["noncat_slot_end"][slot_idx]
                    ch_start_idx = predictions["noncat_alignment_start"][tok_start_idx]
                    ch_end_idx = predictions["noncat_alignment_end"][tok_end_idx]
                    extracted_value = None

                    slot_status = predictions["noncat_slot_status"][slot_idx]
                    if slot_status == STATUS_DONTCARE:
                        slot_values[slot] = STR_DONTCARE
                    elif slot_status == STATUS_ACTIVE:
                        if ch_start_idx > 0 and ch_end_idx > 0:
                            # Add span from the user utterance.
                            extracted_value = user_utterance[ch_start_idx - 1 : ch_end_idx]
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
                            extracted_value = carryover_value
                    elif slot_status == STATUS_CARRY:
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

                        if carryover_value is None:
                            extracted_value = None
                        else:
                            extracted_value = carryover_value

                    if extracted_value is not None:
                        slot_values[slot] = extracted_value

                    # debugging info processing ended
                    if predictions["noncat_slot_status_GT"][slot_idx] != predictions["noncat_slot_status"][
                        slot_idx
                    ] or (
                        predictions["noncat_slot_status_GT"][slot_idx] == predictions["noncat_slot_status"][slot_idx]
                        and predictions["noncat_slot_status_GT"][slot_idx] != STATUS_OFF
                        and extracted_value not in true_state['slot_values'][slot]
                    ):
                        non_categorical_slots_dict[slot] = (
                            predictions["noncat_slot_status_GT"][slot_idx],
                            predictions["noncat_slot_status"][slot_idx],
                            predictions["noncat_slot_status_p"][slot_idx],
                            (ch_start_idx, ch_end_idx),
                            user_utterance[ch_start_idx - 1 : ch_end_idx]
                            if (ch_start_idx > 0 and ch_end_idx > 0)
                            else system_utterance[-ch_start_idx - 1 : -ch_end_idx],
                            extracted_value,
                            predictions["noncat_slot_p"][slot_idx],
                        )

                    if predictions["noncat_slot_status_GT"][slot_idx] != STATUS_OFF:
                        noncat_slot_value_num += 1
                        if extracted_value is not None and extracted_value in true_state['slot_values'][slot]:
                            noncat_slot_value_acc += 1

                    if predictions["noncat_slot_status_GT"][slot_idx] == predictions["noncat_slot_status"][slot_idx]:
                        noncat_slot_status_acc += 1
                    # debugging info processing ended

                carry_over_slots(
                    frame,
                    all_slot_values,
                    slots_relation_list,
                    frame_service_prev,
                    slot_values,
                    sys_slots_agg,
                    sys_slots_last,
                )
                # in debug mode, the following outputs would get generated which can be used for performing error analysis.
                # It prints out the information about the frames in the evaluation set which contain errors and those error in the predictaed state are not originated from previous frames or turns.
                # Therefore, these frames would be the origin of errors in the evaluation dialogues.
                # It just prints out the frames for seen services as our model is designed mostly for seen services and it does not work great on unseen ones.
                if eval_debug and frame["service"] in in_domain_services:
                    equal_state = True
                    for s, v in true_state['slot_values'].items():
                        if s not in slot_values or slot_values[s] not in v:
                            equal_state = False
                            break
                    for s, v in slot_values.items():
                        if s not in true_state['slot_values'] or v not in true_state['slot_values'][s]:
                            equal_state = False
                            break
                    if not equal_state:
                        cat_slot_status_acc = (
                            "NAN" if cat_slot_status_num == 0 else cat_slot_status_acc / cat_slot_status_num
                        )
                        noncat_slot_status_acc = (
                            "NAN" if noncat_slot_status_num == 0 else noncat_slot_status_acc / noncat_slot_status_num
                        )
                        cat_slot_value_acc = (
                            "NAN" if cat_slot_value_num == 0 else cat_slot_value_acc / cat_slot_value_num
                        )
                        noncat_slot_value_acc = (
                            "NAN" if noncat_slot_value_num == 0 else noncat_slot_value_acc / noncat_slot_value_num
                        )

                        found_err = False
                        if cat_slot_status_acc != "NAN" and cat_slot_status_acc < 1.0:
                            found_err = True
                        if noncat_slot_status_acc != "NAN" and noncat_slot_status_acc < 1.0:
                            found_err = True
                        if cat_slot_value_acc != "NAN" and cat_slot_value_acc != 1.0:
                            found_err = True
                        if noncat_slot_value_acc != "NAN" and noncat_slot_value_acc != 1.0:
                            found_err = True

                        if found_err:
                            logging.debug("-----------------------------------New Frame------------------------------")
                            logging.debug(
                                f'DIALOGUE ID : {dialog_id}, TURN ID: {turn_id}, SERVICE: {frame["service"]}, PREV_SERVICE: {frame_service_prev}'
                            )

                            logging.debug(f'SYS : {system_utterance}')
                            logging.debug(f'USER: {user_utterance}')

                            logging.debug("\n")
                            logging.debug(f"PRED CAT: {categorical_slots_dict}")
                            logging.debug(f"PRED NON-CAT: {non_categorical_slots_dict}")

                            logging.debug("\n")
                            logging.debug(f"STATE - LABEL: {sorted(true_state['slot_values'].items())}")
                            logging.debug(f"STATE - PRED : {sorted(slot_values.items())}")
                            logging.debug(f"STATE - PREV: {true_state_prev}")

                            logging.debug("\n")
                            logging.debug(f"SLOTS - LABEL: {true_slots}")
                            logging.debug(f"SYS SLOT AGG: {sys_slots_agg}")
                            logging.debug(f"SYS SLOT LAST: {sys_slots_last}")
                            logging.debug(f"SYS RETS: {sys_rets}")

                            logging.debug("\n")
                            logging.debug(f"CAT STATUS ACC: {cat_slot_status_acc}")

                            logging.debug(f"NONCAT STATUS ACC: {noncat_slot_status_acc}")

                            logging.debug(
                                f"CAT VALUES ACC: {cat_slot_value_acc} ,NONCAT VALUES ACC: {noncat_slot_value_acc}"
                            )

                            found_err = False
                            if cat_slot_status_acc != "NAN" and cat_slot_status_acc < 1.0:
                                logging.debug("CAT_STATUS_ERR")
                                found_err = True
                            if noncat_slot_status_acc != "NAN" and noncat_slot_status_acc < 1.0:
                                logging.debug("NONCAT_STATUS_ERR")
                                found_err = True
                            if (
                                noncat_slot_status_acc != "NAN"
                                and noncat_slot_status_acc < 1.0
                                and cat_slot_status_acc != "NAN"
                                and cat_slot_status_acc < 1.0
                            ):
                                logging.debug("BOTH_STATUS_ERR")
                                found_err = True

                            if cat_slot_value_acc != "NAN" and cat_slot_value_acc < 1.0:
                                logging.debug("CAT_VALUE_ERR")
                                found_err = True
                            if noncat_slot_value_acc != "NAN" and noncat_slot_value_acc < 1.0:
                                logging.debug("NONCAT_VALUE_ERR")
                                found_err = True
                            if (
                                noncat_slot_value_acc != "NAN"
                                and noncat_slot_value_acc != 1.0
                                and cat_slot_value_acc != "NAN"
                                and cat_slot_value_acc != 1.0
                            ):
                                logging.debug("BOTH_VALUE_ERR")
                                found_err = True
                            if not found_err:
                                logging.debug("CLEAN_FRAME")

                # Create a new dict to avoid overwriting the state in previous turns
                # because of use of same objects.
                state["slot_values"] = {s: [v] for s, v in slot_values.items()}
                frame["state"] = state
                frame_service_prev = frame["service"]

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
    all_slot_values = defaultdict(dict)
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
                    if slot_status == STATUS_DONTCARE:
                        slot_values[slot] = STR_DONTCARE
                    elif slot_status == STATUS_ACTIVE:
                        value_idx = predictions["cat_slot_value"][slot_idx]
                        slot_values[slot] = service_schema.get_categorical_slot_values(slot)[value_idx]
                # Non-categorical slots.
                for slot_idx, slot in enumerate(service_schema.non_categorical_slots):
                    slot_status = predictions["noncat_slot_status"][slot_idx]
                    if slot_status == STATUS_DONTCARE:
                        slot_values[slot] = STR_DONTCARE
                    elif slot_status == STATUS_ACTIVE:
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


def write_predictions_to_file(
    predictions, input_json_files, output_dir, schemas, tracker_model, eval_debug, in_domain_services
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
    all_predictions = {}
    for idx, prediction in enumerate(predictions):
        if not prediction["is_real_example"]:
            continue
        eval_dataset, dialog_id, turn_id, service_name = prediction['example_id'].split('-')
        all_predictions[(dialog_id, turn_id, service_name)] = prediction
    logging.info(f'Predictions for {idx} examples in {eval_dataset} dataset are getting processed.')

    # Read each input file and write its predictions.
    for input_file_path in input_json_files:
        with open(input_file_path) as f:
            dialogs = json.load(f)
            logging.debug(f'{input_file_path} file is loaded')
            pred_dialogs = []
            for d in dialogs:
                if tracker_model == 'baseline':
                    pred_dialog = get_predicted_dialog_baseline(d, all_predictions, schemas)
                elif tracker_model == 'nemotracker':
                    pred_dialog = get_predicted_dialog_nemotracker(
                        d, all_predictions, schemas, eval_debug, in_domain_services
                    )
                else:
                    raise ValueError(f"tracker_mode {tracker_model} is not defined.")
                pred_dialogs.append(pred_dialog)
            f.close()
        input_file_name = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_dir, input_file_name)
        with open(output_file_path, "w") as f:
            json.dump(pred_dialogs, f, indent=2, separators=(",", ": "), sort_keys=True)
            f.close()
