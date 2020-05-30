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

import collections
import json
import os
from collections import OrderedDict

from nemo import logging
from nemo.collections.nlp.data.datasets.sgd_dataset.input_example import (
    STATUS_ACTIVE,
    STATUS_DONTCARE,
    STATUS_OFF,
    STR_DONTCARE,
)

REQ_SLOT_THRESHOLD = 0.5
MIN_SLOT_RELATION = 25

__all__ = ['get_predicted_dialog_baseline', 'write_predictions_to_file']


def carry_over_slots(
    cur_usr_frame,
    all_slot_values,
    slots_relation_list,
    prev_frame_service,
    slot_values,
    sys_prev_slots,
    last_sys_slots,
):
    # return
    if prev_frame_service == cur_usr_frame["service"]:
        return
    for (service_dest, slot_dest), cands_list in slots_relation_list.items():
        if service_dest != cur_usr_frame["service"]:
            continue
        for service_src, slot_src, freq in cands_list:
            if freq < MIN_SLOT_RELATION:
                continue
            if (
                service_src == prev_frame_service
                and service_src in all_slot_values
                and slot_src in all_slot_values[service_src]
            ):
                slot_values[slot_dest] = all_slot_values[service_src][slot_src]
            if (
                service_src == prev_frame_service
                and service_src in sys_prev_slots
                and slot_src in sys_prev_slots[service_src]
            ):
                slot_values[slot_dest] = sys_prev_slots[service_src][slot_src]
            if (
                service_src == prev_frame_service
                and service_src in last_sys_slots
                and slot_src in last_sys_slots[service_src]
            ):
                slot_values[slot_dest] = last_sys_slots[service_src][slot_src]


def get_carryover_value(
    slot,
    cur_usr_frame,
    frame_service_prev,
    all_slot_values,
    sys_slots_last,
    sys_slots_agg,
    slots_relation_list,
    sys_rets,
):
    ext_value = None
    if slot in sys_slots_agg[cur_usr_frame["service"]]:
        ext_value = sys_slots_agg[cur_usr_frame["service"]][slot]
        sys_rets[slot] = ext_value
    elif (cur_usr_frame["service"], slot) in slots_relation_list:
        cands_list = slots_relation_list[(cur_usr_frame["service"], slot)]
        for dmn, slt, freq in cands_list:
            if freq < MIN_SLOT_RELATION:
                continue
            if dmn in all_slot_values and slt in all_slot_values[dmn]:
                ext_value = all_slot_values[dmn][slt]
            if dmn in sys_slots_agg and slt in sys_slots_agg[dmn]:
                ext_value = sys_slots_agg[dmn][slt]
            if dmn in sys_slots_last and slt in sys_slots_last[dmn]:
                ext_value = sys_slots_last[dmn][slt]
    return ext_value


def get_predicted_dialog_ret_sys_act(dialog, all_predictions, schemas, eval_debug, in_domain_services):
    """Update labels in a dialogue based on model predictions.
  Args:
    dialog: A json object containing dialogue whose labels are to be updated.
    all_predictions: A dict mapping prediction name to the predicted value. See
      SchemaGuidedDST class for the contents of this dict.
    schemas: A Schema object wrapping all the schemas for the dataset.
  Returns:
    A json object containing the dialogue with labels predicted by the model.
  """
    # This approach retreives slot values from the history of system actions if slot is active but it can not find it in user utterance
    # Overwrite the labels in the turn with the predictions from the model. For
    # test set, these labels are missing from the data and hence they are added.
    dialog_id = dialog["dialogue_id"]
    # The slot values tracked for each service.
    all_slot_values = collections.defaultdict(OrderedDict)
    sys_slots_agg = collections.defaultdict(OrderedDict)
    sys_slots_last = collections.defaultdict(OrderedDict)

    sys_rets = OrderedDict()
    true_state_prev = OrderedDict()
    true_state = OrderedDict()
    frame_service_prev = ""
    for turn_idx, turn in enumerate(dialog["turns"]):
        if turn["speaker"] == "SYSTEM":
            sys_slots_last = collections.defaultdict(OrderedDict)
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

            for frame in turn["frames"]:
                cat_slot_status_acc = 0
                cat_slot_status_num = 0
                noncat_slot_status_num = 0
                noncat_slot_status_acc = 0
                cat_slot_value_acc = 0
                cat_slot_value_num = 0
                noncat_slot_value_acc = 0
                noncat_slot_value_num = 0

                # changed here
                # if frame["service"] not in in_domain_services:
                #     continue
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
                    ext_value = None
                    if slot_status == data_utils.STATUS_DONTCARE:
                        ext_value = data_utils.STR_DONTCARE
                    elif slot_status == data_utils.STATUS_ACTIVE:
                        # value_idx = predictions["cat_slot_value"][slot_idx]
                        # slot_values[slot] = service_schema.get_categorical_slot_values(slot)[value_idx]
                        if (
                            service_schema.get_categorical_slot_values(slot)[predictions["cat_slot_value"][slot_idx]]
                            != "#CARRYVALUE#"
                        ):
                            # if predictions["cat_slot_status_p"][slot_idx] > 0.6:
                            value_idx = predictions["cat_slot_value"][slot_idx]
                            ext_value = service_schema.get_categorical_slot_values(slot)[value_idx]
                        else:
                            carryover_value = get_carryover_value(
                                slot,
                                frame,
                                frame_service_prev,
                                all_slot_values,
                                sys_slots_last,
                                sys_slots_agg,
                                schemas.slots_relation_list,
                                sys_rets,
                            )
                            if carryover_value is not None:
                                ext_value = carryover_value
                                print(f'slot:{slot} with value:{carryover_value} extratced with CARRYVALUE')
                    elif slot_status == data_utils.STATUS_OFF:
                        ext_value = None

                    if ext_value is not None:
                        slot_values[slot] = ext_value
                    # elif predictions["cat_slot_status_p"][slot_idx] < 0.6:
                    #     value_idx = predictions["cat_slot_value"][slot_idx]
                    #     slot_values[slot] = service_schema.get_categorical_slot_values(slot)[value_idx]
                    # print(predictions["cat_slot_status_p"][slot_idx])
                    # predictions["cat_slot_value"][slot_idx] != "##NONE##"
                    # and
                    # if (predictions["cat_slot_status_p"][slot_idx] + predictions["cat_slot_value_p"][slot_idx]) / 2 > 0.9:
                    #     value_idx = predictions["cat_slot_value"][slot_idx]
                    #     slot_values[slot] = service_schema.get_categorical_slot_values(slot)[value_idx]
                    # else:
                    #     if slot in sys_slots_agg[frame["service"]]:
                    #         # debugging info
                    #         sys_rets[slot] = sys_slots_agg[frame["service"]][slot]
                    #         ##
                    #         slot_values[slot] = sys_slots_agg[frame["service"]][slot]
                    #         #print("pooooy", slot_values[slot])
                    #     else:
                    #         value_idx = predictions["cat_slot_value"][slot_idx]
                    #         slot_values[slot] = service_schema.get_categorical_slot_values(slot)[value_idx]

                    ############################################### debugging info
                    if predictions["cat_slot_status_GT"][slot_idx] != predictions["cat_slot_status"][slot_idx] or (
                        predictions["cat_slot_status_GT"][slot_idx] == predictions["cat_slot_status"][slot_idx]
                        and predictions["cat_slot_status_GT"][slot_idx] != data_utils.STATUS_OFF
                        and ext_value not in true_state['slot_values'][slot]
                    ):
                        categorical_slots_dict[slot] = (
                            predictions["cat_slot_status_GT"][slot_idx],
                            predictions["cat_slot_status"][slot_idx],
                            predictions["cat_slot_status_p"][slot_idx],
                            service_schema.get_categorical_slot_values(slot)[predictions["cat_slot_value"][slot_idx]],
                            service_schema.get_categorical_slot_values(slot)[
                                predictions["cat_slot_value_GT"][slot_idx]
                            ],
                            ext_value,
                            predictions["cat_slot_value_p"][slot_idx],
                        )

                    if predictions["cat_slot_status_GT"][slot_idx] == predictions["cat_slot_status"][slot_idx]:
                        cat_slot_status_acc += 1
                    if predictions["cat_slot_status_GT"][slot_idx] != data_utils.STATUS_OFF:
                        cat_slot_value_num += 1
                        if (
                            ext_value in true_state['slot_values'][slot]
                        ):  # service_schema.get_categorical_slot_values(slot)[predictions["cat_slot_value_GT"][slot_idx]]:
                            cat_slot_value_acc += 1
                    ################################################################

                for slot_idx, slot in enumerate(service_schema.non_categorical_slots):
                    noncat_slot_status_num += 1

                    tok_start_idx = predictions["noncat_slot_start"][slot_idx]
                    tok_end_idx = predictions["noncat_slot_end"][slot_idx]
                    ch_start_idx = predictions["noncat_alignment_start"][tok_start_idx]
                    ch_end_idx = predictions["noncat_alignment_end"][tok_end_idx]
                    ext_value = None
                    if ch_start_idx > 0 and ch_end_idx > 0:
                        # Add span from the user utterance.
                        ext_value = user_utterance[ch_start_idx - 1 : ch_end_idx]
                    # elif ch_start_idx < 0 and ch_end_idx < 0:
                    # # Add span from the system utterance.
                    #    ext_value = system_utterance[-ch_start_idx - 1 : -ch_end_idx]
                    else:
                        # if (
                        #     predictions["noncat_slot_status_GT"][slot_idx] == data_utils.STATUS_ACTIVE
                        #     and frame["service"] in in_domain_services
                        # ):
                        #     print("=================================================")
                        #     print(system_utterance, "####", user_utterance)
                        #     print(
                        #         "ridi baz noncat (slot,sys_ext,true_val):",
                        #         slot,
                        #         system_utterance[-ch_start_idx - 1 : -ch_end_idx],
                        #         true_state['slot_values'],
                        #     )
                        #     print("predicted slots:", all_slot_values)

                        ext_value = get_carryover_value(
                            slot,
                            frame,
                            frame_service_prev,
                            all_slot_values,
                            sys_slots_last,
                            sys_slots_agg,
                            schemas.slots_relation_list,
                            sys_rets,
                        )

                    slot_status = predictions["noncat_slot_status"][slot_idx]
                    if slot_status == data_utils.STATUS_DONTCARE:
                        slot_values[slot] = data_utils.STR_DONTCARE
                    elif slot_status == data_utils.STATUS_ACTIVE:
                        if ext_value is not None:
                            slot_values[slot] = ext_value
                            # elif ch_start_idx < 0 and ch_end_idx < 0:
                            #     slot_values[slot] = system_utterance[-ch_start_idx - 1 : -ch_end_idx]
                            #     print("hoooy", slot_values[slot])

                    ############################################### debugging info
                    if predictions["noncat_slot_status_GT"][slot_idx] != predictions["noncat_slot_status"][
                        slot_idx
                    ] or (
                        predictions["noncat_slot_status_GT"][slot_idx] == predictions["noncat_slot_status"][slot_idx]
                        and predictions["noncat_slot_status_GT"][slot_idx] != data_utils.STATUS_OFF
                        and ext_value not in true_state['slot_values'][slot]
                    ):
                        non_categorical_slots_dict[slot] = (
                            predictions["noncat_slot_status_GT"][slot_idx],
                            predictions["noncat_slot_status"][slot_idx],
                            predictions["noncat_slot_status_p"][slot_idx],
                            (ch_start_idx, ch_end_idx),
                            user_utterance[ch_start_idx - 1 : ch_end_idx]
                            if (ch_start_idx > 0 and ch_end_idx > 0)
                            else system_utterance[-ch_start_idx - 1 : -ch_end_idx],
                            ext_value,
                            predictions["noncat_slot_p"][slot_idx],
                        )

                    if predictions["noncat_slot_status_GT"][slot_idx] != data_utils.STATUS_OFF:
                        noncat_slot_value_num += 1
                        if ext_value is not None and ext_value in true_state['slot_values'][slot]:
                            noncat_slot_value_acc += 1

                    if predictions["noncat_slot_status_GT"][slot_idx] == predictions["noncat_slot_status"][slot_idx]:
                        noncat_slot_status_acc += 1

                carry_over_slots(
                    frame,
                    all_slot_values,
                    schemas.slots_relation_list,
                    frame_service_prev,
                    slot_values,
                    sys_slots_agg,
                    sys_slots_last,
                )
                #############################################################################

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
                                f'DIALOGUE ID : {dialog_id}, TURN ID: {turn_id}, SERVICE: {frame["service"]}'
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
    predictions, input_json_files, output_dir, schemas, state_tracker, eval_debug, in_domain_services
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
        _, dialog_id, turn_id, service_name = prediction['example_id'].split('-')
        all_predictions[(dialog_id, turn_id, service_name)] = prediction
    logging.info(f'Predictions for {idx} examples are getting processed.')

    # Read each input file and write its predictions.
    for input_file_path in input_json_files:
        with open(input_file_path) as f:
            dialogs = json.load(f)
            logging.info(f'{input_file_path} file is loaded')
            pred_dialogs = []
            for d in dialogs:
                if state_tracker == 'baseline':
                    pred_dialog = get_predicted_dialog_baseline(d, all_predictions, schemas)
                elif state_tracker == 'ret_sys_act':
                    pred_dialog = get_predicted_dialog_ret_sys_act(
                        d, all_predictions, schemas, eval_debug, in_domain_services
                    )
                else:
                    raise ValueError(f"tracker_mode {state_tracker} is not defined.")
                pred_dialogs.append(pred_dialog)
            f.close()
        input_file_name = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_dir, input_file_name)
        with open(output_file_path, "w") as f:
            json.dump(pred_dialogs, f, indent=2, separators=(",", ": "), sort_keys=True)
            f.close()
