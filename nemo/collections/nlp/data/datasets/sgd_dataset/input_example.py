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

from nemo import logging

__all__ = ['InputExample', 'STR_DONTCARE', 'STATUS_OFF', 'STATUS_ACTIVE', 'STATUS_DONTCARE', 'truncate_seq_pair']

STR_DONTCARE = "dontcare"

# These are used to represent the status of slots (off, active, dontcare) and
# intents (off, active) in dialogue state tracking.
STATUS_OFF = 0
STATUS_ACTIVE = 1
STATUS_DONTCARE = 2


class InputExample(object):
    """An example for training/inference."""

    def __init__(
        self,
        schema_config,
        service_schema=None,
        example_id="NONE",
        example_id_num=[],
        is_real_example=False,
        tokenizer=None,
    ):
        """Constructs an InputExample.

        Args:
          max_seq_length: The maximum length of the sequence. Sequences longer than
            this value will be truncated.
          service_schema: A ServiceSchema object wrapping the schema for the service
            corresponding to this example.
          example_id: Unique identifier for the example, like: 'train-1_00000-00-Restaurants_1'
          example_id_num: dialogue_id and turn_id combined and service id combined into a list of ints,
            like: [1, 0, 0, 18]
          is_real_example: Indicates if an example is real or used for padding in a
            minibatch.
          tokenizer (Tokenizer): such as NemoBertTokenizer
        """
        self.schema_config = schema_config
        self.service_schema = service_schema
        self.example_id = example_id
        self.example_id_num = example_id_num

        self.is_real_example = is_real_example
        self._max_seq_length = schema_config["MAX_SEQ_LENGTH"]
        self._tokenizer = tokenizer
        if self.is_real_example and self._tokenizer is None:
            raise ValueError("Must specify tokenizer when input is a real example.")

        self.user_utterance = ''
        self.system_utterance = ''
        # The id of each subword in the vocabulary for BERT.
        self.utterance_ids = [0] * self._max_seq_length
        # Denotes the identity of the sequence. Takes values 0 (system utterance) and 1 (user utterance).
        self.utterance_segment = [0] * self._max_seq_length
        # Mask which takes the value 0 for padded tokens and 1 otherwise.
        self.utterance_mask = [0] * self._max_seq_length
        # Start and inclusive end character indices in the original utterance
        # corresponding to the tokens. This is used to obtain the character indices
        # from the predicted subword indices during inference.
        # NOTE: A positive value indicates the character indices in the user
        # utterance whereas a negative value indicates the character indices in the
        # system utterance. The indices are offset by 1 to prevent ambiguity in the
        # 0 index, which could be in either the user or system utterance by the
        # above convention. Now the 0 index corresponds to padded tokens.
        self.start_char_idx = [0] * self._max_seq_length
        self.end_char_idx = [0] * self._max_seq_length

        # Number of categorical slots present in the service.
        self.num_categorical_slots = 0
        # The status of each categorical slot in the service.
        self.categorical_slot_status = [STATUS_OFF] * schema_config["MAX_NUM_CAT_SLOT"]
        # Masks out categorical status for padded cat slots
        self.cat_slot_status_mask = [0] * len(self.categorical_slot_status)
        # Number of values taken by each categorical slot.
        self.num_categorical_slot_values = [0] * schema_config["MAX_NUM_CAT_SLOT"]
        # The index of the correct value for each categorical slot.
        self.categorical_slot_values = [0] * schema_config["MAX_NUM_CAT_SLOT"]
        # Masks out categorical slots values for slots not used in the service
        self.cat_slot_values_mask = [
            [0] * schema_config["MAX_NUM_VALUE_PER_CAT_SLOT"] for _ in range(schema_config["MAX_NUM_CAT_SLOT"])
        ]

        # Number of non-categorical slots present in the service.
        self.num_noncategorical_slots = 0
        # The status of each non-categorical slot in the service.
        self.noncategorical_slot_status = [STATUS_OFF] * schema_config["MAX_NUM_NONCAT_SLOT"]
        # Masks out non-categorical status for padded cat slots
        self.noncat_slot_status_mask = [0] * len(self.noncategorical_slot_status)
        # The index of the starting subword corresponding to the slot span for a
        # non-categorical slot value.
        self.noncategorical_slot_value_start = [0] * schema_config["MAX_NUM_NONCAT_SLOT"]
        # The index of the ending (inclusive) subword corresponding to the slot span
        # for a non-categorical slot value.
        self.noncategorical_slot_value_end = [0] * schema_config["MAX_NUM_NONCAT_SLOT"]

        # Total number of slots present in the service. All slots are included here
        # since every slot can be requested.
        self.num_slots = 0
        # Takes value 1 if the corresponding slot is requested, 0 otherwise.
        self.requested_slot_status = [STATUS_OFF] * (
            schema_config["MAX_NUM_CAT_SLOT"] + schema_config["MAX_NUM_NONCAT_SLOT"]
        )
        # Masks out requested slots that are not used for the service
        self.requested_slot_mask = [0] * len(self.requested_slot_status)

        # Total number of intents present in the service.
        self.num_intents = 0
        # Takes value 1 if the intent is active, 0 otherwise.
        self.intent_status = [STATUS_OFF] * schema_config["MAX_NUM_INTENT"]
        # Masks out intents that are not used for the service, [1] for none intent
        self.intent_status_mask = [1] + [0] * len(self.intent_status)
        # Label for active intent in the turn
        self.intent_status_labels = 0

    @property
    def readable_summary(self):
        """Get a readable dict that summarizes the attributes of an InputExample."""
        seq_length = sum(self.utterance_mask)
        utt_toks = self._tokenizer.ids_to_tokens(self.utterance_ids[:seq_length])
        utt_tok_mask_pairs = list(zip(utt_toks, self.utterance_segment[:seq_length]))
        active_intents = [
            self.service_schema.get_intent_from_id(idx)
            for idx, s in enumerate(self.intent_status)
            if s == STATUS_ACTIVE
        ]
        if len(active_intents) > 1:
            raise ValueError("Should not have multiple active intents in a single service.")
        active_intent = active_intents[0] if active_intents else ""
        slot_values_in_state = {}
        for idx, s in enumerate(self.categorical_slot_status):
            if s == STATUS_ACTIVE:
                value_id = self.categorical_slot_values[idx]
                slot_values_in_state[
                    self.service_schema.get_categorical_slot_from_id(idx)
                ] = self.service_schema.get_categorical_slot_value_from_id(idx, value_id)
            elif s == STATUS_DONTCARE:
                slot_values_in_state[self.service_schema.get_categorical_slot_from_id(idx)] = STR_DONTCARE
        for idx, s in enumerate(self.noncategorical_slot_status):
            if s == STATUS_ACTIVE:
                slot = self.service_schema.get_non_categorical_slot_from_id(idx)
                start_id = self.noncategorical_slot_value_start[idx]
                end_id = self.noncategorical_slot_value_end[idx]
                # Token list is consisted of the subwords that may start with "##". We
                # remove "##" to reconstruct the original value. Note that it's not a
                # strict restoration of the original string. It's primarily used for
                # debugging.
                # ex. ["san", "j", "##ose"] --> "san jose"
                readable_value = " ".join(utt_toks[start_id : end_id + 1]).replace(" ##", "")
                slot_values_in_state[slot] = readable_value
            elif s == STATUS_DONTCARE:
                slot = self.service_schema.get_non_categorical_slot_from_id(idx)
                slot_values_in_state[slot] = STR_DONTCARE

        summary_dict = {
            "utt_tok_mask_pairs": utt_tok_mask_pairs,
            "utt_len": seq_length,
            "num_categorical_slots": self.num_categorical_slots,
            "num_categorical_slot_values": self.num_categorical_slot_values,
            "num_noncategorical_slots": self.num_noncategorical_slots,
            "service_name": self.service_schema.service_name,
            "active_intent": active_intent,
            "slot_values_in_state": slot_values_in_state,
        }
        return summary_dict

    def add_utterance_features(
        self, system_tokens, system_inv_alignments, user_tokens, user_inv_alignments, system_utterance, user_utterance
    ):
        """Add utterance related features input to bert.

        Note: this method modifies the system tokens and user_tokens in place to
        make their total length <= the maximum input length for BERT model.

        Args:
          system_tokens: a list of strings which represents system utterance.
          system_inv_alignments: a list of tuples which denotes the start and end
            charater of the tpken that a bert token originates from in the original
            system utterance.
          user_tokens: a list of strings which represents user utterance.
          user_inv_alignments: a list of tuples which denotes the start and end
            charater of the token that a bert token originates from in the original
            user utterance.
        """
        # Make user-system utterance input (in BERT format)
        # Input sequence length for utterance BERT encoder
        max_utt_len = self._max_seq_length

        # Modify lengths of sys & usr utterance so that length of total utt
        # (including cls_token, setp_token, sep_token) is no more than max_utt_len
        is_too_long = truncate_seq_pair(system_tokens, user_tokens, max_utt_len - 3)
        if is_too_long:
            logging.debug(f'Utterance sequence truncated in example id - {self.example_id}.')

        # Construct the tokens, segment mask and valid token mask which will be
        # input to BERT, using the tokens for system utterance (sequence A) and
        # user utterance (sequence B).
        utt_subword = []
        utt_seg = []
        utt_mask = []
        start_char_idx = []
        end_char_idx = []

        utt_subword.append(self._tokenizer.cls_token)
        utt_seg.append(0)
        utt_mask.append(1)
        start_char_idx.append(0)
        end_char_idx.append(0)

        for subword_idx, subword in enumerate(system_tokens):
            utt_subword.append(subword)
            utt_seg.append(0)
            utt_mask.append(1)
            st, en = system_inv_alignments[subword_idx]
            start_char_idx.append(-(st + 1))
            end_char_idx.append(-(en + 1))

        utt_subword.append(self._tokenizer.sep_token)
        utt_seg.append(0)
        utt_mask.append(1)
        start_char_idx.append(0)
        end_char_idx.append(0)

        for subword_idx, subword in enumerate(user_tokens):
            utt_subword.append(subword)
            utt_seg.append(1)
            utt_mask.append(1)
            st, en = user_inv_alignments[subword_idx]
            start_char_idx.append(st + 1)
            end_char_idx.append(en + 1)

        utt_subword.append(self._tokenizer.sep_token)
        utt_seg.append(1)
        utt_mask.append(1)
        start_char_idx.append(0)
        end_char_idx.append(0)

        utterance_ids = self._tokenizer.tokens_to_ids(utt_subword)

        # Zero-pad up to the BERT input sequence length.
        while len(utterance_ids) < max_utt_len:
            utterance_ids.append(0)
            utt_seg.append(0)
            utt_mask.append(0)
            start_char_idx.append(0)
            end_char_idx.append(0)
        self.utterance_ids = utterance_ids
        self.utterance_segment = utt_seg
        self.utterance_mask = utt_mask
        self.start_char_idx = start_char_idx
        self.end_char_idx = end_char_idx

        self.user_utterances = user_utterance
        self.system_utterance = system_utterance

    def make_copy_with_utterance_features(self):
        """Make a copy of the current example with utterance features."""
        new_example = InputExample(
            schema_config=self.schema_config,
            service_schema=self.service_schema,
            example_id=self.example_id,
            example_id_num=self.example_id_num,
            is_real_example=self.is_real_example,
            tokenizer=self._tokenizer,
        )
        new_example.utterance_ids = list(self.utterance_ids)
        new_example.utterance_segment = list(self.utterance_segment)
        new_example.utterance_mask = list(self.utterance_mask)
        new_example.start_char_idx = list(self.start_char_idx)
        new_example.end_char_idx = list(self.end_char_idx)
        new_example.user_utterance = self.user_utterance
        new_example.system_utterance = self.system_utterance
        return new_example

    def add_categorical_slots(self, state_update):
        """Add features for categorical slots."""
        categorical_slots = self.service_schema.categorical_slots
        self.num_categorical_slots = len(categorical_slots)
        for slot_idx, slot in enumerate(categorical_slots):
            values = state_update.get(slot, [])
            # Add categorical slot value features.
            slot_values = self.service_schema.get_categorical_slot_values(slot)
            self.num_categorical_slot_values[slot_idx] = len(slot_values)
            # set slot mask to 1, i.e. the slot exists in the service
            self.cat_slot_status_mask[slot_idx] = 1
            # set the number of active slot values for this slots in the service
            for slot_value_idx in range(len(self.service_schema._categorical_slot_values[slot])):
                self.cat_slot_values_mask[slot_idx][slot_value_idx] = 1

            if not values:
                self.categorical_slot_status[slot_idx] = STATUS_OFF
            elif values[0] == STR_DONTCARE:
                self.categorical_slot_status[slot_idx] = STATUS_DONTCARE
            else:
                self.categorical_slot_status[slot_idx] = STATUS_ACTIVE
                self.categorical_slot_values[slot_idx] = self.service_schema.get_categorical_slot_value_id(
                    slot, values[0]
                )

    def add_noncategorical_slots(self, state_update, system_span_boundaries, user_span_boundaries):
        """Add features for non-categorical slots."""
        noncategorical_slots = self.service_schema.non_categorical_slots
        self.num_noncategorical_slots = len(noncategorical_slots)
        for slot_idx, slot in enumerate(noncategorical_slots):
            values = state_update.get(slot, [])
            self.noncat_slot_status_mask[slot_idx] = 1
            if not values:
                self.noncategorical_slot_status[slot_idx] = STATUS_OFF
            elif values[0] == STR_DONTCARE:
                self.noncategorical_slot_status[slot_idx] = STATUS_DONTCARE
            else:
                self.noncategorical_slot_status[slot_idx] = STATUS_ACTIVE
                # Add indices of the start and end tokens for the first encountered
                # value. Spans in user utterance are prioritized over the system
                # utterance. If a span is not found, the slot value is ignored.
                if slot in user_span_boundaries:
                    start, end = user_span_boundaries[slot]
                elif slot in system_span_boundaries:
                    start, end = system_span_boundaries[slot]
                else:
                    # A span may not be found because the value was cropped out or because
                    # the value was mentioned earlier in the dialogue. Since this model
                    # only makes use of the last two utterances to predict state updates,
                    # it will fail in such cases.
                    logging.debug(
                        f'"Slot values {str(values)} not found in user or system utterance in example with id - {self.example_id}.'
                    )

                    continue
                self.noncategorical_slot_value_start[slot_idx] = start
                self.noncategorical_slot_value_end[slot_idx] = end

    def add_requested_slots(self, frame):
        all_slots = self.service_schema.slots
        self.num_slots = len(all_slots)
        for slot_idx, slot in enumerate(all_slots):
            self.requested_slot_mask[slot_idx] = 1
            if slot in frame["state"]["requested_slots"]:
                self.requested_slot_status[slot_idx] = STATUS_ACTIVE

    def add_intents(self, frame):
        all_intents = self.service_schema.intents
        self.num_intents = len(all_intents)
        for intent_idx, intent in enumerate(all_intents):
            if intent == frame["state"]["active_intent"]:
                self.intent_status[intent_idx] = STATUS_ACTIVE
                # adding +1 to take none intent into account
                # supports only 1 active intent in the turn
                self.intent_status_labels = intent_idx + 1
            self.intent_status_mask[intent_idx + 1] = 1


# Modified from run_classifier._truncate_seq_pair in the public bert model repo.
# https://github.com/google-research/bert/blob/master/run_classifier.py.
def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncate a seq pair in place so that their total length <= max_length."""
    is_too_long = False
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        is_too_long = True
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    return is_too_long
