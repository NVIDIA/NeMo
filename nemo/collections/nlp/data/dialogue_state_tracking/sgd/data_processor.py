# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst/baseline/data_utils.py
"""

import collections
import json
import os
import pickle
import re
from typing import Dict, List

import numpy as np

from nemo.collections.nlp.data.dialogue_state_tracking.sgd.input_example import InputExample
from nemo.utils import logging

__all__ = ['SGDDataProcessor']

FILE_RANGES = {
    "sgd_single_domain": {"train": range(1, 44), "dev": range(1, 8), "test": range(1, 12)},
    "sgd_multi_domain": {"train": range(44, 128), "dev": range(8, 21), "test": range(12, 35)},
    "sgd_all": {"train": range(1, 128), "dev": range(1, 21), "test": range(1, 35)},
    "sgd_all_single": {"train": range(1, 128), "dev": range(1, 8), "test": range(1, 12)},
    "multiwoz": {"train": range(1, 18), "dev": range(1, 3), "test": range(1, 3)},
    "debug_sample": {"train": range(1, 2), "dev": range(1, 2), "test": range(1, 2)},
}


class SGDDataProcessor(object):
    """Data generator for SGD dialogues."""

    def __init__(
        self,
        task_name: str,
        data_dir: str,
        dialogues_example_dir: str,
        tokenizer: object,
        schemas: object,
        schema_config: Dict[str, int],
        subsample: bool = False,
    ):
        """
        Constructs SGDDataProcessor
        Args:
            task_name: task name, e.g. "sgd_single_domain"
            data_dir: path to data directory
            dialogues_example_dir: path to store processed dialogue examples
            tokenizer: tokenizer object
            schemas: schema object 
            schema_config: schema configuration
            subsample: whether to balance positive and negative samples in dataset
        """
        self.data_dir = data_dir

        self._task_name = task_name
        self.schemas = schemas
        self.schema_config = schema_config

        train_file_range = FILE_RANGES[task_name]["train"]
        dev_file_range = FILE_RANGES[task_name]["dev"]
        test_file_range = FILE_RANGES[task_name]["test"]

        self._file_ranges = {
            "train": train_file_range,
            "dev": dev_file_range,
            "test": test_file_range,
        }

        self._seen_services = {
            "train": set(),
            "dev": set(),
            "test": set(),
        }

        self._tokenizer = tokenizer
        self._subsample = subsample
        self._dialogues_example_dir = dialogues_example_dir

        self.dial_files = {}

        # slots_relation_list.np would contain the candidate list of slots for each (service, slot) which would be
        # looked into when a switch between two services happens in the dialogue and we can not find any value for a slot in the current user utterance.
        # This file would get generated from the dialogues in the training set.
        self.slots_relation_file = os.path.join(dialogues_example_dir, f"{task_name}_train_slots_relation_list.np")
        for dataset in ["train", "dev", "test"]:
            # Process dialogue files
            dial_file = f"{task_name}_{dataset}_examples.processed"
            dial_file = os.path.join(dialogues_example_dir, dial_file)
            self.dial_files[(task_name, dataset)] = dial_file

            dialog_paths = SGDDataProcessor.get_dialogue_files(data_dir, dataset, task_name)
            dialogs = SGDDataProcessor.load_dialogues(dialog_paths)
            for dialog in dialogs:
                self._seen_services[dataset].update(set(dialog['services']))

    def save_dialog_examples(self, overwrite_dial_files: bool):
        """
        Preprocesses dialogues and saves to disk.
        Args:
            overwrite_dial_files: whether or not to overwrite saved file if already exists
        """
        for dataset in ["train", "dev", "test"]:
            dial_file = self.dial_files[(self._task_name, dataset)]
            if not os.path.exists(dial_file) or overwrite_dial_files:
                logging.info(f"Start generating the dialogue examples for {dataset} dataset.")
                if not os.path.exists(self._dialogues_example_dir):
                    os.makedirs(self._dialogues_example_dir)
                dial_examples, slots_relation_list = self._generate_dialog_examples(
                    dataset, self.schemas, self._subsample
                )
                for ex in dial_examples:
                    del ex.service_schema
                    del ex._tokenizer
                    del ex.schema_config
                    del ex.user_utterance
                    del ex.categorical_slot_id
                    del ex.system_utterance
                    del ex.noncategorical_slot_id
                    del ex.categorical_slot_value_id
                    del ex.requested_slot_id
                    del ex.intent_id
                with open(dial_file, "wb") as f:
                    np.save(f, dial_examples)

                if dataset == "train":
                    with open(self.slots_relation_file, "wb") as f:
                        pickle.dump(slots_relation_list, f)
                    logging.info(f"The slot carry-over list for train set is stored at {self.slots_relation_file}")

                logging.info(f"The dialogue examples for {dataset} dataset saved at {dial_file}")
                logging.info(f"Finish generating the dialogue examples for {dataset} dataset.")

    def get_dialog_examples(self, dataset_split: str) -> List[object]:
        """
        Loads preprocessed dialogue examples from disk. 
        Args:
            dataset_split: dataset split
        Returns:
            dial_examples:  list of InputExample's.
        """
        if (self._task_name, dataset_split) not in self.dial_files or not os.path.exists(
            self.dial_files[(self._task_name, dataset_split)]
        ):
            raise ValueError(
                f"{dataset_split} dialogue examples were not processed for {self._task_name} task. Re-initialize SGDDataProcessor and add {dataset_split} dataset split to datasets arg."
            )
        dial_file = self.dial_files[(self._task_name, dataset_split)]
        logging.info(f"Loading dialogue examples from {dial_file}.")

        with open(dial_file, "rb") as f:
            dial_examples = np.load(f, allow_pickle=True)
            f.close()

        if not os.path.exists(self.slots_relation_file):
            raise ValueError(
                f"Slots relation file {self.slots_relation_file} does not exist. It is needed for the carry-over mechanism of state tracker for switches between services."
            )
        if os.path.getsize(self.slots_relation_file) > 0:
            with open(self.slots_relation_file, "rb") as f:
                self.schemas._slots_relation_list = pickle.load(f)
            logging.info(
                f"Loaded the slot relation list for value carry-over between services from {self.slots_relation_file}."
            )

        return dial_examples

    def get_seen_services(self, dataset_split: str):
        """
        Returns list of seen services, i.e. both in given and training split
        Args:
            dataset_split: data split
        Returns:
            seen_services: list of seen services
        """
        seen_services = self._seen_services[dataset_split]
        return seen_services

    def _generate_dialog_examples(self, dataset_split: str, schemas: object, subsample: bool):
        """
        Returns a list of `InputExample`s of the data splits' dialogues.
        Args:
            dataset_split: data split, can be "train", "dev", or "test".
            schemas: schema for all services of all datasets 
            subsample: whether to balance postive and negative samples in the dataset
        Returns:
            examples: a list of `InputExample`s.
        """
        logging.info(f'Creating examples and slot relation list from the dialogues started...')
        dialog_paths = [
            os.path.join(self.data_dir, dataset_split, "dialogues_{:03d}.json".format(i))
            for i in self._file_ranges[dataset_split]
        ]
        dialogs = SGDDataProcessor.load_dialogues(dialog_paths)

        examples = []
        slot_carryover_candlist = collections.defaultdict(int)
        for dialog_idx, dialog in enumerate(dialogs):
            if dialog_idx % 1000 == 0:
                logging.info(f'Processed {dialog_idx} dialogues.')
            examples.extend(
                self._create_examples_from_dialog(dialog, schemas, dataset_split, slot_carryover_candlist, subsample)
            )

        slots_relation_list = collections.defaultdict(list)
        for slots_relation, relation_size in slot_carryover_candlist.items():
            if relation_size > 0:
                slots_relation_list[(slots_relation[0], slots_relation[1])].append(
                    (slots_relation[2], slots_relation[3], relation_size)
                )
                slots_relation_list[(slots_relation[2], slots_relation[3])].append(
                    (slots_relation[0], slots_relation[1], relation_size)
                )

        return examples, slots_relation_list

    def _create_examples_from_dialog(
        self, dialog: dict, schemas: object, dataset_split: str, slot_carryover_candlist: dict, subsample: bool
    ):
        """
        Create examples for every turn in the dialogue.
        Args:
            dialog: dialogue example
            schemas: schema for all services of all datasets
            dataset_split: data split
            slot_carryover_candlist: a dictionary to keep and count the number of carry-over cases between two slots from two different services
            subsample: whether to balance postive and negative samples in the dataset
        Returns:
            examples: a list of `InputExample`s.
        """
        dialog_id = dialog["dialogue_id"]
        prev_states = {}
        examples = []
        for turn_idx, turn in enumerate(dialog["turns"]):
            # Generate an example for every frame in every user turn.
            if turn["speaker"] == "USER":
                user_utterance = turn["utterance"]
                user_frames = {f["service"]: f for f in turn["frames"]}
                if turn_idx > 0:
                    system_turn = dialog["turns"][turn_idx - 1]
                    system_utterance = system_turn["utterance"]
                    system_frames = {f["service"]: f for f in system_turn["frames"]}
                else:
                    system_utterance = ""
                    system_frames = {}

                turn_id = "{}-{}-{:02d}".format(dataset_split, dialog_id, turn_idx)
                turn_examples, prev_states, slot_carryover_values = self._create_examples_from_turn(
                    turn_id,
                    system_utterance,
                    user_utterance,
                    system_frames,
                    user_frames,
                    prev_states,
                    schemas,
                    subsample,
                )
                examples.extend(turn_examples)

                for value, slots_list in slot_carryover_values.items():
                    if value in ["True", "False"]:
                        continue
                    if len(slots_list) > 1:
                        for service1, slot1 in slots_list:
                            for service2, slot2 in slots_list:
                                if service1 == service2:
                                    continue
                                if service1 > service2:
                                    service1, service2 = service2, service1
                                    slot1, slot2 = slot2, slot1
                                slot_carryover_candlist[(service1, slot1, service2, slot2)] += 1
        return examples

    def _get_state_update(self, current_state: dict, prev_state: dict) -> dict:
        """
        Updates dialogue state
        Args:
            current_state: slot values pairs for the current dialogue turn
            prev_state: slot values pairs for the previous dialogue turns
        Returns:
            state_update: slot values pairs that are added/updated during the current dialogue turn
        """
        state_update = dict(current_state)
        for slot, values in current_state.items():
            if slot in prev_state and prev_state[slot][0] in values:
                # Remove the slot from state if its value didn't change.
                state_update.pop(slot)
        return state_update

    def _create_examples_from_turn(
        self,
        turn_id: int,
        system_utterance: str,
        user_utterance: str,
        system_frames: dict,
        user_frames: dict,
        prev_states: dict,
        schemas: object,
        subsample: bool,
    ):
        """
        Creates an example for each frame in the user turn.
        Args:
            turn_id: turn number
            system_utterance: last system utterance
            user_utterance: lst user utterance
            system_frames: all system utterances and slot - slot value pairs
            user_frames: all user utterances and slot - slot value pairs
            prev_states: slot - slot value pairs from the previous turns
            schemas: schema for all services of all datasets
            subsample: whether to balance postive and negative samples in the dataset
        Returns:
            examples: a list of `InputExample`s.
            prev_states: updated dialogue state e.g. {'Restaurants_1': {'city': ['San Jose'], 'cuisine': ['American']}}
        """
        user_tokens, user_alignments, user_inv_alignments = self._tokenize(user_utterance)
        system_tokens, system_alignments, system_inv_alignments = self._tokenize(system_utterance)
        system_user_utterance = system_utterance + ' ' + user_utterance
        system_user_tokens, system_user_alignments, system_user_inv_alignments = self._tokenize(system_user_utterance)
        states = {}

        examples = []
        slot_carryover_values = collections.defaultdict(list)
        for service, user_frame in user_frames.items():

            base_example = InputExample(schema_config=self.schema_config, tokenizer=self._tokenizer,)
            base_example.service_schema = schemas.get_service_schema(service)
            base_example.service_id = schemas.get_service_schema(service).service_id
            system_frame = system_frames.get(service, None)
            state = user_frame["state"]["slot_values"]
            state_update = self._get_state_update(state, prev_states.get(service, {}))
            states[service] = state
            dataset_split, dialog_id, turn_id_ = turn_id.split('-')
            dialog_id_1, dialog_id_2 = dialog_id.split('_')

            base_example.example_id = f"{turn_id}-{service}"
            base_example.example_id_num = [
                int(dialog_id_1),
                int(dialog_id_2),
                int(turn_id_),
                schemas.get_service_id(service),
            ]

            for model_task in range(self.schema_config["NUM_TASKS"]):
                if model_task == 0:
                    for intent_id, intent in enumerate(schemas.get_service_schema(service).intents):
                        task_example = base_example.make_copy()
                        task_example.task_mask[model_task] = 1
                        task_example.intent_id = intent_id
                        task_example.example_id += f"-{model_task}-{intent_id}-0"
                        task_example.example_id_num.extend([model_task, intent_id, 0])
                        intent_description = (
                            intent + " " + schemas.get_service_schema(service).intent_descriptions[intent]
                        )
                        intent_tokens, intent_alignments, intent_inv_alignments = self._tokenize(intent_description)
                        task_example.add_utterance_features(
                            intent_tokens,
                            intent_inv_alignments,
                            system_user_tokens,
                            system_user_inv_alignments,
                            intent_description,
                            system_user_utterance,
                        )
                        task_example.add_intents(user_frame)
                        examples.append(task_example)

                if model_task == 1:
                    for slot_id, slot in enumerate(schemas.get_service_schema(service).slots):
                        task_example = base_example.make_copy()
                        task_example.task_mask[model_task] = 1
                        task_example.requested_slot_id = slot_id
                        task_example.example_id += f"-{model_task}-{slot_id}-0"
                        task_example.example_id_num.extend([model_task, slot_id, 0])
                        slot_description = slot + " " + schemas.get_service_schema(service).slot_descriptions[slot]
                        slot_tokens, slot_alignments, slot_inv_alignments = self._tokenize(slot_description)
                        task_example.add_utterance_features(
                            slot_tokens,
                            slot_inv_alignments,
                            user_tokens,
                            user_inv_alignments,
                            slot_description,
                            user_utterance,
                        )
                        task_example.add_requested_slots(user_frame)
                        examples.append(task_example)
                if model_task == 2:
                    off_slots = []
                    on_slots = []
                    for slot_id, slot in enumerate(schemas.get_service_schema(service).categorical_slots):
                        task_example = base_example.make_copy()
                        task_example.task_mask[model_task] = 1

                        # assert task_example.task_mask == [0, 0, 1, 0, 0, 0]
                        task_example.categorical_slot_id = slot_id
                        task_example.example_id += f"-{model_task}-{slot_id}-0"
                        task_example.example_id_num.extend([model_task, slot_id, 0])
                        slot_description = slot + " " + schemas.get_service_schema(service).slot_descriptions[slot]
                        slot_tokens, slot_alignments, slot_inv_alignments = self._tokenize(slot_description)
                        task_example.add_utterance_features(
                            slot_tokens,
                            slot_inv_alignments,
                            system_user_tokens,
                            system_user_inv_alignments,
                            slot_description,
                            system_user_utterance,
                        )
                        task_example.add_categorical_slots(state_update)
                        if task_example.categorical_slot_status == 0:
                            off_slots.append(task_example)
                        else:
                            on_slots.append(task_example)
                            examples.append(task_example)
                        old_example = task_example

                        for value_id, value in enumerate(
                            schemas.get_service_schema(service).get_categorical_slot_values(slot)
                        ):
                            if dataset_split != 'train' or task_example.categorical_slot_status == 1:
                                task_example = old_example.make_copy_of_categorical_features()
                                task_example.task_mask[3] = 1
                                # assert task_example.task_mask == [0, 0, 0, 1, 0, 0]
                                task_example.categorical_slot_id = slot_id
                                task_example.categorical_slot_value_id = value_id
                                task_example.example_id = base_example.example_id + f"-3-{slot_id}-{value_id}"
                                task_example.example_id_num = base_example.example_id_num + [3, slot_id, value_id]
                                slot_description = slot + " " + value  # add slot description
                                slot_tokens, slot_alignments, slot_inv_alignments = self._tokenize(slot_description)
                                task_example.add_utterance_features(
                                    slot_tokens,
                                    slot_inv_alignments,
                                    system_user_tokens,
                                    system_user_inv_alignments,
                                    slot_description,
                                    system_user_utterance,
                                )
                                task_example.add_categorical_slots(state_update)
                                assert task_example.categorical_slot_status == old_example.categorical_slot_status
                                examples.append(task_example)

                    if dataset_split == 'train' and subsample:
                        num_on_slots = len(on_slots)
                        examples.extend(
                            np.random.choice(off_slots, replace=False, size=min(max(num_on_slots, 1), len(off_slots)))
                        )
                    else:
                        examples.extend(off_slots)

                if model_task == 4:  # noncat slot status
                    off_slots = []
                    on_slots = []
                    for slot_id, slot in enumerate(schemas.get_service_schema(service).non_categorical_slots):
                        task_example = base_example.make_copy()
                        task_example.task_mask[model_task] = 1
                        # assert task_example.task_mask == [0, 0, 0, 0, 1, 0]
                        task_example.noncategorical_slot_id = slot_id
                        task_example.example_id += f"-{model_task}-{slot_id}-0"
                        task_example.example_id_num.extend([model_task, slot_id, 0])
                        slot_description = slot + " " + schemas.get_service_schema(service).slot_descriptions[slot]
                        slot_tokens, slot_alignments, slot_inv_alignments = self._tokenize(slot_description)
                        task_example.add_utterance_features(
                            slot_tokens,
                            slot_inv_alignments,
                            system_user_tokens,
                            system_user_inv_alignments,
                            slot_description,
                            system_user_utterance,
                        )

                        user_span_boundaries = self._find_subword_indices(
                            state_update,
                            user_utterance,
                            user_frame["slots"],
                            user_alignments,
                            user_tokens,
                            2 + len(slot_tokens) + len(system_tokens),
                        )
                        if system_frame is not None:
                            system_span_boundaries = self._find_subword_indices(
                                state_update,
                                system_utterance,
                                system_frame["slots"],
                                system_alignments,
                                system_tokens,
                                2 + len(slot_tokens),
                            )
                        else:
                            system_span_boundaries = {}
                        task_example.add_noncategorical_slots(
                            state_update, user_span_boundaries, system_span_boundaries
                        )
                        if task_example.noncategorical_slot_status == 0:
                            off_slots.append(task_example)
                        else:
                            on_slots.append(task_example)
                            examples.append(task_example)

                        if dataset_split != 'train' or task_example.noncategorical_slot_status == 1:
                            task_example = task_example.make_copy_of_non_categorical_features()
                            task_example.task_mask[5] = 1
                            # assert task_example.task_mask == [0, 0, 0, 0, 0, 1]
                            task_example.example_id = base_example.example_id + f"-5-{slot_id}-0"
                            task_example.example_id_num = base_example.example_id_num + [5, slot_id, 0]
                            examples.append(task_example)
                    if dataset_split == 'train' and subsample:
                        num_on_slots = len(on_slots)
                        examples.extend(
                            np.random.choice(off_slots, replace=False, size=min(max(num_on_slots, 1), len(off_slots)))
                        )
                    else:
                        examples.extend(off_slots)

            if service not in prev_states and int(turn_id_) > 0:
                for slot_name, values in state_update.items():
                    for value in values:
                        slot_carryover_values[value].append((service, slot_name))
                for prev_service, prev_slot_value_list in prev_states.items():
                    if prev_service == service:
                        continue
                    if prev_service in state:
                        prev_slot_value_list = state[prev_service]
                    for prev_slot_name, prev_values in prev_slot_value_list.items():
                        for prev_value in prev_values:
                            slot_carryover_values[prev_value].append((prev_service, prev_slot_name))

        return examples, states, slot_carryover_values

    def _find_subword_indices(
        self,
        slot_values: dict,
        utterance: str,
        char_slot_spans: dict,
        alignments: List[int],
        subwords: List[str],
        bias: int,
    ) -> dict:
        """
        Find indices for subwords corresponding to slot values.
        Args:
            slot_values: slot - slot value pairs
            utterance: utterance
            char_slot_spans: char - slot spans
            alignments: alignments
            subwords: subtokens mapping
            bias: offset
        Returns:
            span_boundaries: span boundaries
        """
        span_boundaries = {}
        for slot, values in slot_values.items():
            # Get all values present in the utterance for the specified slot.
            value_char_spans = {}
            for slot_span in char_slot_spans:
                if slot_span["slot"] == slot:
                    value = utterance[slot_span["start"] : slot_span["exclusive_end"]]
                    start_tok_idx = alignments[slot_span["start"]]
                    end_tok_idx = alignments[slot_span["exclusive_end"] - 1]
                    if 0 <= start_tok_idx < len(subwords):
                        end_tok_idx = min(end_tok_idx, len(subwords) - 1)
                        value_char_spans[value] = (start_tok_idx + bias, end_tok_idx + bias)
            for v in values:
                if v in value_char_spans:
                    span_boundaries[slot] = value_char_spans[v]
                    break
        return span_boundaries

    def _tokenize(self, utterance: str):
        """
        Tokenize the utterance

        Args:
            utterance: A string containing the utterance to be tokenized.

        Returns:
            bert_tokens: A list of tokens obtained by word-piece tokenization of the
                utterance.
            alignments: A dict mapping indices of characters corresponding to start
                and end positions of words (not subwords) to corresponding indices in
                bert_tokens list.
            inverse_alignments: A list of size equal to bert_tokens. Each element is a
                tuple containing the index of the starting and inclusive ending
                character of the word corresponding to the subword. This list is used
                during inference to map word-piece indices to spans in the original
                utterance.
        """
        # utterance = tokenization.convert_to_unicode(utterance)

        # After _naive_tokenize, spaces and punctuation marks are all retained, i.e.
        # direct concatenation of all the tokens in the sequence will be the
        # original string.
        tokens = SGDDataProcessor._naive_tokenize(utterance)
        # ['I', ' ', 'am', ' ', 'feeling', ' ', 'hungry', ' ', 'so', ' ', 'I', ' ', 'would', ' ', 'like', ' ', 'to', ' ', 'find', ' ', 'a', ' ', 'place', ' ', 'to', ' ', 'eat', '.']
        # Filter out empty tokens and obtain aligned character index for each token.
        alignments = {}
        char_index = 0
        bert_tokens = (
            []
        )  # ['I', 'am', 'feeling', 'hungry', 'so', 'I', 'would', 'like', 'to', 'find', 'a', 'place', 'to', 'eat', '.']
        # These lists store inverse alignments to be used during inference.
        bert_tokens_start_chars = []
        bert_tokens_end_chars = []
        for token in tokens:
            if token.strip():
                subwords = self._tokenizer.text_to_tokens(token)
                # Store the alignment for the index of starting character and the
                # inclusive ending character of the token.
                alignments[char_index] = len(bert_tokens)
                bert_tokens_start_chars.extend([char_index] * len(subwords))
                bert_tokens.extend(subwords)
                # The inclusive ending character index corresponding to the word.
                inclusive_char_end = char_index + len(token) - 1
                alignments[inclusive_char_end] = len(bert_tokens) - 1
                bert_tokens_end_chars.extend([inclusive_char_end] * len(subwords))
            char_index += len(token)
        inverse_alignments = list(zip(bert_tokens_start_chars, bert_tokens_end_chars))
        return bert_tokens, alignments, inverse_alignments

    @classmethod
    def _naive_tokenize(cls, s: str):
        """
        Tokenizes a string, separating words, spaces and punctuations.
        Args:
            s: a string
        Returns:
            seq_tok: list of words, spaces and punctuations from the string
        """
        # Spaces and punctuation marks are all retained, i.e. direct concatenation
        # of all the tokens in the sequence will be the original string.
        seq_tok = [tok for tok in re.split(r"([^a-zA-Z0-9])", s) if tok]
        return seq_tok

    @classmethod
    def load_dialogues(cls, dialog_json_filepaths: List[str]) -> List[dict]:
        """
        Obtain the list of all dialogues from specified json files.
        Args:
            dialog_json_filepaths: list of json files
        Returns:
            dialogs: the list of all dialogues
        """
        dialogs = []
        for dialog_json_filepath in sorted(dialog_json_filepaths):
            with open(dialog_json_filepath, 'r') as f:
                dialogs.extend(json.load(f))
                f.close()
        return dialogs

    @classmethod
    def get_dialogue_files(cls, data_dir: str, dataset_split: str, task_name: str):
        """
        Obtain the list of all dialogue json files
        Args:
            data_dir: path to the data folder
            dataset_split: data split
            task_name: SGD task name, see keys of the FILE_RANGES
        Returns:
            dialog: the list of all dialogue json files paths
        """
        return [
            os.path.join(data_dir, dataset_split, 'dialogues_{:03d}.json'.format(fid))
            for fid in FILE_RANGES[task_name][dataset_split]
        ]
