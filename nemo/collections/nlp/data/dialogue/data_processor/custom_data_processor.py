# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import json
import os
from typing import List

from tqdm import tqdm

from nemo.collections.nlp.data.dialogue.data_processor.data_processor import DialogueDataProcessor
from nemo.collections.nlp.data.dialogue.input_example.input_example import DialogueInputExample
from nemo.utils import logging


class DialogueCustomDataProcessor(DialogueDataProcessor):
    """ Data processor for a custom dialog bot dataset """

    def __init__(self, data_dir: str, tokenizer: object = None, cfg=None):
        """ Constructs the dialog data processor """

        self.data_dir = data_dir
        self.cfg = cfg
        self._task_name = self.cfg.task_name
        self._tokenizer = tokenizer

    def get_dialog_examples(self, dataset_split) -> List[object]:
        """
        Reads dialogue examples from self.data_dir/dataset_split/examples.json
        and formats into SGD structure
        """

        dialogues_filepath = os.path.join(self.data_dir, dataset_split, "examples.json")
        if not os.path.exists(dialogues_filepath):
            raise ValueError(f"Dialogues file {dialogues_filepath} does not exist")

        logging.info(f"Reading dialogue examples from {dialogues_filepath}")
        with open(dialogues_filepath, "r") as f:
            data = json.load(f)["data"]

        logging.info(f"Processing dialogues...")
        examples = []
        for dialogue_idx, dialogue in enumerate(tqdm(data)):

            # skip dialogues where system slots are absent
            if not dialogue["Fulfillment"]:
                continue

            DialogueCustomDataProcessor.add_metadata_to_fulfillment_slots(dialogue)
            dialogue_dict = DialogueCustomDataProcessor.create_dialogue_sgd_format_dict(dialogue, dialogue_idx)
            if dialogue_dict:
                curr_examples = DialogueCustomDataProcessor.create_examples_from_dialogue_dict(dialogue_dict)
                examples.extend(curr_examples)

        return examples

    @staticmethod
    def get_user_entity_to_slot_mapping(dialogue):
        """
        Formats entities detected in user utterance into slots structure to match SGD format
        """

        entity_to_slot_mapping = {}
        for slot in dialogue["Slots"]:
            slot_name = slot["Name"]
            if isinstance(slot["Values"][0], str):
                slot_value = slot["Values"][0]
            elif isinstance(slot["Values"][0], dict):
                slot_value = slot["Values"][0]["Text"]

            for entity in dialogue["Entities"]:
                if entity["Token"] == slot_value:
                    entity_to_slot_mapping.update({slot_name: entity})
                    entity_to_slot_mapping[slot_name].update({"value": slot_value})
                    break

        return entity_to_slot_mapping

    @staticmethod
    def add_metadata_to_fulfillment_slots(dialogue):
        """
        Appends metadata from dialog state to current slots in the dialogue
        """

        for slot in dialogue["Fulfillment"][0]["FulfillmentSlots"]:
            for slot_metadata in dialogue["DialogState"]["GlobalSlot"]:
                if slot["Name"] in slot_metadata["Name"]:
                    slot.update({"Values": [f"{slot['Values'][0]} {slot_metadata['Values'][0]}"]})
                    break

    @staticmethod
    def create_dialogue_sgd_format_dict(dialogue, dialogue_idx):
        """
        Formats a raw dialogue read from file into SGD-format dict with user and system slots
        """

        service = dialogue["Domain"]

        entity2slot = DialogueCustomDataProcessor.get_user_entity_to_slot_mapping(dialogue)
        user_slots = []
        for entity_slot in entity2slot:
            user_slots.append(
                {
                    "slot": entity_slot,
                    "start": entity2slot[entity_slot]["Span"]["start"],
                    "exclusive_end": entity2slot[entity_slot]["Span"]["end"],
                }
            )

        system_slots = []
        for slot in dialogue["Fulfillment"][0]["FulfillmentSlots"]:
            system_slots.append(
                {"act": "", "canonical_values": [], "slot": slot["Name"], "values": slot["Values"],}
            )

        dialogue_dict = {
            "dialogue_id": dialogue_idx,
            "services": [service],
            "turns": [
                {
                    "frames": [
                        {
                            "actions": [],
                            "service": service,
                            "slots": user_slots,
                            "state": {
                                "active_intent": dialogue["Intent"],
                                "requested_slots": [],
                                "slot_values": {slot: [entity2slot[slot]["value"]] for slot in entity2slot},
                            },
                        }
                    ],
                    "speaker": "USER",
                    "utterance": dialogue["Query"],
                },
                {
                    "frames": [{"actions": system_slots, "service": service, "slots": [],}],
                    "speaker": "SYSTEM",
                    "utterance": dialogue["Response"]["CleanedText"],
                },
            ],
        }

        return dialogue_dict

    @staticmethod
    def create_examples_from_dialogue_dict(dialogue_dict):
        """
        Processes a dialogue with multiple USER/SYSTEM turns and creates input examples
        """

        dialogue_id = dialogue_dict["dialogue_id"]
        examples = []
        for turn_idx, turn in enumerate(dialogue_dict["turns"]):
            if turn["speaker"] == "USER":

                # get system turn present at +1 index from user turn
                system_turn = dialogue_dict["turns"][turn_idx + 1]
                system_utterance = system_turn["utterance"]
                system_frames = {f["service"]: f for f in system_turn["frames"]}

                user_utterance = turn["utterance"]
                user_frames = {f["service"]: f for f in turn["frames"]}
                curr_examples = DialogueCustomDataProcessor.create_example_from_turn(
                    dialogue_id, turn_idx, system_utterance, user_utterance, system_frames, user_frames,
                )
                examples.extend(curr_examples)

        return examples

    @staticmethod
    def create_example_from_turn(
        dialogue_id, turn_id, system_utterance, user_utterance, system_frames, user_frames,
    ):
        """ 
        Processes a single turn in the dialogue and creates DialogueInputExample in SGD format
        """

        examples = []
        for frame_idx, (service, user_frame) in enumerate(user_frames.items()):
            slot_values = user_frame["state"]["slot_values"]
            intent = user_frames[service]["state"]['active_intent']
            system_frame = system_frames.get(service, None)
            one_example = {
                "example_id": f"DIAL{dialogue_id}_TURN{turn_id}_FRAME{frame_idx}",
                "example_id_num": frame_idx,
                "utterance": user_utterance,
                "system_utterance": system_utterance,
                "system_slots": {slot["slot"]: slot for slot in system_frame["slots"]}
                if system_frame is not None
                else None,
                "system_actions": system_frame["actions"] if system_frame is not None else None,
                "labels": {"service": service, "intent": intent, "slots": slot_values,},
                "label_positions": {"slots": {slot["slot"]: slot for slot in user_frames[service]["slots"]}},
                "possible_labels": {"service": "", "intent": [], "slots": {},},
                "description": {"service": "", "intent": "", "slots": {},},
            }
            examples.append(DialogueInputExample(one_example))

        return examples
