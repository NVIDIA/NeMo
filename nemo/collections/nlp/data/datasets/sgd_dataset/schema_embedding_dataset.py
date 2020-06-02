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
Extract BERT embeddings for slots, values, intents in schema.

This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst/baseline/extract_schema_embedding.py
"""

import collections
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from nemo import logging
from nemo.collections.nlp.data.datasets.sgd_dataset.input_example import truncate_seq_pair

# Separator to separate the two sentences in BERT's input sequence.
_NL_SEPARATOR = "|||"

__all__ = ['SchemaEmbeddingDataset']


class SchemaEmbeddingDataset(Dataset):
    def __init__(self, schema_config, tokenizer, schemas):
        """Generate the embeddings for a schema's elements.

        Args:
          tokenizer (tokenizer): such as NemoBertTokenizer
          max_seq_length: Sequence length used for BERT model
          schemas: Schemas for all services in the datasets
        """
        self._tokenizer = tokenizer
        self.schema_config = schema_config
        self.schemas = schemas

        input_features = self._get_input_features()

        self.features = collections.defaultdict(list)

        for feature in input_features:
            self.features["input_ids"].append(feature.input_ids)
            self.features["input_mask"].append(feature.input_mask)
            self.features["input_type_ids"].append(feature.input_type_ids)
            self.features["embedding_tensor_name"].append(feature.embedding_tensor_name)
            self.features["service_id"].append(feature.service_id)
            self.features["intent_or_slot_id"].append(feature.intent_or_slot_id)
            self.features["value_id"].append(feature.value_id)

    def __len__(self):
        return len(self.features['input_ids'])

    def __getitem__(self, idx):
        return (
            np.array(self.features['input_ids'][idx]),
            np.array(self.features['input_mask'][idx], dtype=np.long),
            np.array(self.features['input_type_ids'][idx]),
        )

    def _create_feature(self, line, embedding_tensor_name, service_id, intent_or_slot_id, value_id=-1):
        """Create a single InputFeatures instance."""
        seq_length = self.schema_config["MAX_SEQ_LENGTH"]
        # line = tokenization.convert_to_unicode(input_line)
        line = line.strip()
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)

        tokens_a = self._tokenizer.text_to_tokens(text_a)
        tokens_b = None
        if text_b:
            tokens_b = self._tokenizer.text_to_tokens(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0 : (seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it
        # makes it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense
        # because the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append(self._tokenizer.cls_token)
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append(self._tokenizer.sep_token)
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append(self._tokenizer.sep_token)
            input_type_ids.append(1)

        input_ids = self._tokenizer.tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        return InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids,
            embedding_tensor_name=embedding_tensor_name,
            service_id=service_id,
            intent_or_slot_id=intent_or_slot_id,
            value_id=value_id,
        )

    def _get_intents_input_features(self, service_schema):
        """Create features for BERT inference for all intents of a service.

      We use "[service description] ||| [intent name] [intent description]" as an
        intent's full description.

      Args:
        service_schema: A ServiceSchema object containing the schema for the
          corresponding service.

      Returns:
        A list of InputFeatures containing features to be given as input to the
        BERT model.
      """
        service_des = service_schema.description

        features = []
        intent_descriptions = {i["name"]: i["description"] for i in service_schema.schema_json["intents"]}
        for intent_id, intent in enumerate(service_schema.intents):
            nl_seq = " ".join([service_des, _NL_SEPARATOR, intent, intent_descriptions[intent]])
            features.append(self._create_feature(nl_seq, "intent_emb", service_schema.service_id, intent_id))
        return features

    def _get_req_slots_input_features(self, service_schema):
        """Create features for BERT inference for all requested slots of a service.

      We use "[service description] ||| [slot name] [slot description]" as a
        slot's full description.

      Args:
        service_schema: A ServiceSchema object containing the schema for the
          corresponding service.

      Returns:
        A list of InputFeatures containing features to be given as input to the
        BERT model.
      """
        service_des = service_schema.description

        slot_descriptions = {s["name"]: s["description"] for s in service_schema.schema_json["slots"]}
        features = []
        for slot_id, slot in enumerate(service_schema.slots):
            nl_seq = " ".join([service_des, _NL_SEPARATOR, slot, slot_descriptions[slot]])
            features.append(self._create_feature(nl_seq, "req_slot_emb", service_schema.service_id, slot_id))
        return features

    def _get_goal_slots_and_values_input_features(self, service_schema):
        """Get BERT input features for all goal slots and categorical values.

      We use "[service description] ||| [slot name] [slot description]" as a
        slot's full description.
      We use ""[slot name] [slot description] ||| [value name]" as a categorical
        slot value's full description.

      Args:
        service_schema: A ServiceSchema object containing the schema for the
          corresponding service.

      Returns:
        A list of InputFeatures containing features to be given as input to the
        BERT model.
      """
        service_des = service_schema.description

        features = []
        slot_descriptions = {s["name"]: s["description"] for s in service_schema.schema_json["slots"]}

        for slot_id, slot in enumerate(service_schema.non_categorical_slots):
            nl_seq = " ".join([service_des, _NL_SEPARATOR, slot, slot_descriptions[slot]])
            features.append(self._create_feature(nl_seq, "noncat_slot_emb", service_schema.service_id, slot_id))

        for slot_id, slot in enumerate(service_schema.categorical_slots):
            nl_seq = " ".join([service_des, _NL_SEPARATOR, slot, slot_descriptions[slot]])
            features.append(self._create_feature(nl_seq, "cat_slot_emb", service_schema.service_id, slot_id))
            for value_id, value in enumerate(service_schema.get_categorical_slot_values(slot)):
                nl_seq = " ".join([slot, slot_descriptions[slot], _NL_SEPARATOR, value])
                features.append(
                    self._create_feature(nl_seq, "cat_slot_value_emb", service_schema.service_id, slot_id, value_id)
                )
        return features

    def _get_input_features(self):
        """Get the input function to compute schema element embeddings.

        Args:
          schemas: A wrapper for all service schemas in the dataset to be embedded.

        Returns:
          The input_fn to be passed to the estimator.
        """
        # Obtain all the features.
        features = []
        for service in self.schemas.services:
            service_schema = self.schemas.get_service_schema(service)
            features.extend(self._get_intents_input_features(service_schema))
            features.extend(self._get_req_slots_input_features(service_schema))
            features.extend(self._get_goal_slots_and_values_input_features(service_schema))

        return features

    def _populate_schema_embeddings(self, schema_embeddings, hidden_states, mode):
        """
        Populate all schema embeddings with BERT embeddings.
        """
        completed_services = set()
        batch_size, seq_len, hidden_size = hidden_states[0].shape

        for idx in range(len(self)):
            service_id = self.features['service_id'][idx]
            service = self.schemas.get_service_from_id(service_id)

            if service not in completed_services:
                logging.debug(f"Generating embeddings for service {service}.")
                completed_services.add(service)
            tensor_name = self.features["embedding_tensor_name"][idx]
            emb_mat = schema_embeddings[service_id][tensor_name]

            if mode == 'random':
                # randomly initialize schema embeddings
                random_token = random.randint(0, seq_len - 1)
                embedding = [round(float(x), 6) for x in hidden_states[0][idx, random_token, :].flat]
            elif mode == 'last_layer_average':
                # Obtain the encoding of the [CLS] token.
                embedding = [round(float(x), 6) for x in np.mean(hidden_states[0][idx, :], 0).flat]
            elif mode == 'baseline':
                # Obtain the encoding of the [CLS] token.
                embedding = [round(float(x), 6) for x in hidden_states[0][idx, 0, :].flat]
            else:
                raise ValueError(f'Mode {mode} for generation schema embeddings is not supported')
            intent_or_slot_id = self.features['intent_or_slot_id'][idx]
            value_id = self.features['value_id'][idx]

            if tensor_name == "cat_slot_value_emb":
                emb_mat[intent_or_slot_id, value_id] = embedding
            else:
                emb_mat[intent_or_slot_id] = embedding

    def save_embeddings(self, bert_hidden_states, output_file, mode):
        """Generate schema element embeddings and save it as a numpy file."""
        schema_embeddings = []
        max_num_intent = self.schema_config["MAX_NUM_INTENT"]
        max_num_cat_slot = self.schema_config["MAX_NUM_CAT_SLOT"]
        max_num_noncat_slot = self.schema_config["MAX_NUM_NONCAT_SLOT"]
        max_num_slot = max_num_cat_slot + max_num_noncat_slot
        max_num_value = self.schema_config["MAX_NUM_VALUE_PER_CAT_SLOT"]
        embedding_dim = self.schema_config["EMBEDDING_DIMENSION"]

        for _ in self.schemas.services:
            schema_embeddings.append(
                {
                    "intent_emb": np.zeros([max_num_intent, embedding_dim]),
                    "req_slot_emb": np.zeros([max_num_slot, embedding_dim]),
                    "cat_slot_emb": np.zeros([max_num_cat_slot, embedding_dim]),
                    "noncat_slot_emb": np.zeros([max_num_noncat_slot, embedding_dim]),
                    "cat_slot_value_emb": np.zeros([max_num_cat_slot, max_num_value, embedding_dim]),
                }
            )

        # Populate the embeddings based on bert inference results and save them.
        self._populate_schema_embeddings(schema_embeddings, bert_hidden_states, mode)

        master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if master_device:
            with open(output_file, "wb") as f_s:
                np.save(f_s, schema_embeddings)
                logging.info(f"The schema embeddings saved at {output_file}")
                f_s.close()


class InputFeatures(object):
    """A single set of features for BERT inference."""

    def __init__(
        self, input_ids, input_mask, input_type_ids, embedding_tensor_name, service_id, intent_or_slot_id, value_id
    ):
        # The ids in the vocabulary for input tokens.
        self.input_ids = input_ids
        # A boolean mask indicating which tokens in the input_ids are valid.
        self.input_mask = input_mask
        # Denotes the sequence each input token belongs to.
        self.input_type_ids = input_type_ids
        # The name of the embedding tensor corresponding to this example.
        self.embedding_tensor_name = embedding_tensor_name
        # The id of the service corresponding to this example.
        self.service_id = service_id
        # The id of the intent (for intent embeddings) or slot (for slot or slot
        # value embeddings) corresponding to this example.
        self.intent_or_slot_id = intent_or_slot_id
        # The id of the value corresponding to this example. Only set if slot value
        # embeddings are being calculated.
        self.value_id = value_id
