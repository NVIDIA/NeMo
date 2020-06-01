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
https://github.com/google-research/google-research/blob/master/schema_guided_dst/baseline/data_utils.py
"""

import json
import os
import re

import numpy as np
import torch

from nemo.collections.nlp.data.datasets.sgd_dataset.input_example import InputExample
from nemo.utils import logging

__all__ = ['FILE_RANGES', 'PER_FRAME_OUTPUT_FILENAME', 'SGDDataProcessor', 'get_dialogue_files']


FILE_RANGES = {
    "sgd_single_domain": {"train": range(1, 44), "dev": range(1, 8), "test": range(1, 12)},
    "sgd_multi_domain": {"train": range(44, 128), "dev": range(8, 21), "test": range(12, 35)},
    "sgd_all": {"train": range(1, 128), "dev": range(1, 21), "test": range(1, 35)},
    "multiwoz": {"train": range(1, 18), "dev": range(1, 3), "test": range(1, 3)},
    "debug_sample": {"train": range(1, 2), "dev": range(1, 2), "test": range(1, 2)},
}

# Name of the file containing all predictions and their corresponding frame metrics.
PER_FRAME_OUTPUT_FILENAME = "dialogues_and_metrics.json"


class SGDDataProcessor(object):
    """Data generator for SGD dialogues."""

    def __init__(
        self, task_name, data_dir, dialogues_example_dir, tokenizer, schema_emb_processor, overwrite_dial_files=False,
    ):
        """
        Constructs SGD8DataProcessor
        Args:
            task_name (str): task  name, for  example, "single_domain"
            data_dir (str): path to data directory
            dialogues_example_dir (str): path to  store processed dialogue examples
            tokenizer (Tokenizer): such as NemoBertTokenizer
            schema_emb_processor (Obj): contains information about schemas
            overwrite_dial_files (bool): whether to overwite dialogue files
        """
        self.data_dir = data_dir
        self.dialogues_examples_dir = dialogues_example_dir

        self._task_name = task_name
        self.schema_config = schema_emb_processor.schema_config

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
        self._max_seq_length = self.schema_config["MAX_SEQ_LENGTH"]

        self.dial_files = {}

        for dataset in ["train", "dev", "test"]:
            # Process dialogue files
            dial_file = f"{task_name}_{dataset}_examples.processed"
            dial_file = os.path.join(dialogues_example_dir, dial_file)
            self.dial_files[(task_name, dataset)] = dial_file

            dialog_paths = SGDDataProcessor.get_dialogue_files(data_dir, dataset, task_name)
            dialogs = SGDDataProcessor.load_dialogues(dialog_paths)
            for dialog in dialogs:
                self._seen_services[dataset].update(set(dialog['services']))

            if not os.path.exists(dial_file) or overwrite_dial_files:
                logging.debug(f"Start generating the dialogue examples for {dataset} dataset.")
                master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
                if master_device:
                    if not os.path.exists(dialogues_example_dir):
                        os.makedirs(dialogues_example_dir)
                    dial_examples = self._generate_dialog_examples(dataset, schema_emb_processor.schemas)
                    with open(dial_file, "wb") as f:
                        np.save(f, dial_examples)
                        f.close()
                    logging.debug(f"The dialogue examples for {dataset} dataset saved at {dial_file}")
                logging.debug(f"Finish generating the dialogue examples for {dataset} dataset.")

            # wait until the master process writes to the dialogue processed file
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    def get_dialog_examples(self, dataset):
        """
        Returns a list of `InputExample`s of the data splits' dialogues.
        Args:
          dataset(str): can be "train", "dev", or "test".
        Returns:
          examples: a list of `InputExample`s.
        """
        if (self._task_name, dataset) not in self.dial_files or not os.path.exists(
            self.dial_files[(self._task_name, dataset)]
        ):
            raise ValueError(
                f"{dataset} dialogue examples were not processed for {self._task_name} task. Re-initialize SGDDataProcessor and add {dataset} dataset to datasets arg."
            )
        dial_file = self.dial_files[(self._task_name, dataset)]
        logging.info(f"Loading dialogue examples from {dial_file}.")
        with open(dial_file, "rb") as f:
            dial_examples = np.load(f, allow_pickle=True)
            f.close()
        return dial_examples

    def get_seen_services(self, dataset_split):
        return self._seen_services[dataset_split]

    def _generate_dialog_examples(self, dataset, schemas):
        """
        Returns a list of `InputExample`s of the data splits' dialogues.
        Args:
          dataset(str): can be "train", "dev", or "test".
          schemas(Schema): for all services and all datasets processed by the schema_processor
        Returns:
          examples: a list of `InputExample`s.
        """
        logging.info(f'Creating examples from the dialogues started...')
        dialog_paths = [
            os.path.join(self.data_dir, dataset, "dialogues_{:03d}.json".format(i)) for i in self._file_ranges[dataset]
        ]
        dialogs = SGDDataProcessor.load_dialogues(dialog_paths)

        examples = []
        for dialog_idx, dialog in enumerate(dialogs):
            if dialog_idx % 1000 == 0:
                logging.info(f'Processed {dialog_idx} dialogs.')
            examples.extend(self._create_examples_from_dialog(dialog, schemas, dataset))

        logging.info(f'Finished creating the examples from {len(dialogs)} dialogues.')
        return examples

    def _create_examples_from_dialog(self, dialog, schemas, dataset):
        """
        Create examples for every turn in the dialog.
        Args:
            dialog (dict): dialogue example
            schemas(Schema): for all services and all datasets processed by the schema_processor
            dataset(str): can be "train", "dev", or "test".
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

                turn_id = "{}-{}-{:02d}".format(dataset, dialog_id, turn_idx)
                turn_examples, prev_states = self._create_examples_from_turn(
                    turn_id, system_utterance, user_utterance, system_frames, user_frames, prev_states, schemas
                )
                examples.extend(turn_examples)
        return examples

    def _get_state_update(self, current_state, prev_state):
        """
        Updates dialogue state
        Args:
            current_state (dict): dict of slot - slot values pairs for the current dialogue turn
            prev_state (dict): dict of slot - slot values pairs for the previous dialogue turns
        Returns:
            state_update (dict): dict of slot - slot values pairs that very added/updated during the current
                dialogue turn
        """
        state_update = dict(current_state)
        for slot, values in current_state.items():
            if slot in prev_state and prev_state[slot][0] in values:
                # Remove the slot from state if its value didn't change.
                state_update.pop(slot)
        return state_update

    def _create_examples_from_turn(
        self, turn_id, system_utterance, user_utterance, system_frames, user_frames, prev_states, schemas
    ):
        """
        Creates an example for each frame in the user turn.
        Args:
            turn_id (int): turn number
            system_utterance (str): last system utterance
            user_utterance (str): lst user utterance
            system_frames (dict): all system utterances and slot - slot value pairs
            user_frames (dict): all user utterances and slot - slot value pairs
            prev_states (dict): slot - slot value pairs from the previous turns
            schemas (obj): carries information about the service from the current turn
        Returns:
            examples: a list of `InputExample`s.
            prev_states (dict): updated dialogue state
        """
        system_tokens, system_alignments, system_inv_alignments = self._tokenize(system_utterance)
        user_tokens, user_alignments, user_inv_alignments = self._tokenize(user_utterance)
        states = {}
        base_example = InputExample(schema_config=self.schema_config, is_real_example=True, tokenizer=self._tokenizer,)
        base_example.example_id = turn_id

        _, dialog_id, turn_id_ = turn_id.split('-')
        dialog_id_1, dialog_id_2 = dialog_id.split('_')
        base_example.example_id_num = [int(dialog_id_1), int(dialog_id_2), int(turn_id_)]
        base_example.add_utterance_features(
            system_tokens, system_inv_alignments, user_tokens, user_inv_alignments, system_utterance, user_utterance
        )
        examples = []
        for service, user_frame in user_frames.items():
            # Create an example for this service.
            example = base_example.make_copy_with_utterance_features()

            example.example_id = "{}-{}".format(turn_id, service)
            _, dialog_id, turn_id_ = turn_id.split('-')
            dialog_id_1, dialog_id_2 = dialog_id.split('_')
            example.example_id_num = [
                int(dialog_id_1),
                int(dialog_id_2),
                int(turn_id_),
                schemas.get_service_id(service),
            ]

            example.service_schema = schemas.get_service_schema(service)
            system_frame = system_frames.get(service, None)
            state = user_frame["state"]["slot_values"]
            state_update = self._get_state_update(state, prev_states.get(service, {}))
            states[service] = state
            # Populate features in the example.
            example.add_categorical_slots(state_update)
            # The input tokens to bert are in the format [CLS] [S1] [S2] ... [SEP]
            # [U1] [U2] ... [SEP] [PAD] ... [PAD]. For system token indices a bias of
            # 1 is added for the [CLS] token and for user tokens a bias of 2 +
            # len(system_tokens) is added to account for [CLS], system tokens and
            # [SEP].
            user_span_boundaries = self._find_subword_indices(
                state_update, user_utterance, user_frame["slots"], user_alignments, user_tokens, 2 + len(system_tokens)
            )
            if system_frame is not None:
                system_span_boundaries = self._find_subword_indices(
                    state_update, system_utterance, system_frame["slots"], system_alignments, system_tokens, 1
                )
            else:
                system_span_boundaries = {}
            example.add_noncategorical_slots(state_update, user_span_boundaries, system_span_boundaries)
            example.add_requested_slots(user_frame)
            example.add_intents(user_frame)
            examples.append(example)
        return examples, states

    def _find_subword_indices(self, slot_values, utterance, char_slot_spans, alignments, subwords, bias):
        """Find indices for subwords corresponding to slot values."""
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

    def _tokenize(self, utterance):
        """Tokenize the utterance using word-piece tokenization used by BERT.

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
        # Filter out empty tokens and obtain aligned character index for each token.
        alignments = {}
        char_index = 0
        bert_tokens = []
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

    def get_num_dialog_examples(self, dataset):
        """
        Gets the number of dilaog examples in the data split.
        Args:
          dataset: str. can be "train", "dev", or "test".
        Returns:from nemo_nlp.data.datasets.sgd import data_utils
          example_count: int. number of examples in the specified dataset.
        """
        example_count = 0
        dialog_paths = [
            os.path.join(self.data_dir, dataset, "dialogues_{:03d}.json".format(i)) for i in self._file_ranges[dataset]
        ]
        dst_set = SGDDataProcessor.load_dialogues(dialog_paths)
        for dialog in dst_set:
            for turn in dialog["turns"]:
                if turn["speaker"] == "USER":
                    example_count += len(turn["frames"])
        return example_count

    @classmethod
    def _naive_tokenize(cls, s):
        """
        Tokenizes a string, separating words, spaces and punctuations.
        Args:
            s (str): a string
        Returns:
            seq_tok (list): list of words, spaces and punctuations from the s
        """
        # Spaces and punctuation marks are all retained, i.e. direct concatenation
        # of all the tokens in the sequence will be the original string.
        seq_tok = [tok for tok in re.split(r"([^a-zA-Z0-9])", s) if tok]
        return seq_tok

    @classmethod
    def load_dialogues(cls, dialog_json_filepaths):
        """
        Obtain the list of all dialogues from specified json files.
        Args:
            dialog_json_filepaths (list): list of json files
        Returns:
            dialogs  (list): the list of all dialogues
        """
        dialogs = []
        for dialog_json_filepath in sorted(dialog_json_filepaths):
            with open(dialog_json_filepath, 'r') as f:
                dialogs.extend(json.load(f))
                f.close()
        return dialogs

    @classmethod
    def get_dialogue_files(cls, data_dir, dataset_split, task_name):
        """
        Obtain the list of all dialogue json files
        Args:
            data_dir (str): path to the data folde
            dataset_split (str): dev, test or train
            task_name (str): SGD task name, see keys of the FILE_RANGES
        Returns:
            dialogs (list): the list of all dialogue json files paths
        """
        return [
            os.path.join(data_dir, dataset_split, 'dialogues_{:03d}.json'.format(fid))
            for fid in FILE_RANGES[task_name][dataset_split]
        ]
