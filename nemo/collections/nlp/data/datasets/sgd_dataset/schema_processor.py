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

import collections
import os

import numpy as np
import torch

from nemo import logging
from nemo.collections.nlp.data.datasets.sgd_dataset import schema
from nemo.collections.nlp.data.datasets.sgd_dataset.schema_embedding_dataset import SchemaEmbeddingDataset
from nemo.collections.nlp.nm.data_layers.bert_inference_datalayer import BertInferDataLayer
from nemo.collections.nlp.utils.data_utils import concatenate

__all__ = ['SchemaPreprocessor']


class SchemaPreprocessor:
    """ 
    Convert the raw data to the standard format supported by
    StateTrackingSGDData.
    
    Args:
        data_dir (str) - Directory for the downloaded DSTC8/SGD data, which contains
            the dialogue files and schema files of all datasets (eg train, dev)
        dialogues_example_dir (str) - Directory where preprocessed DSTC8/SGD dialogues are stored
        schema_embedding_dir (str) - Directory where .npy file for embedding of
            entities (slots, values, intents) in the dataset_split's
            schema are stored.
        task_name (str) - The name of the task to train
        vocab_file (str) - The path to BERT vocab file
        do_lower_case - (bool) - Whether to lower case the input text.
            Should be True for uncased models and False for cased models.
        max_seq_length (int) - The maximum total input sequence length after
            WordPiece tokenization. Sequences longer than this will be
            truncated, and sequences shorter than this will be padded."
        tokenizer - tokenizer
        bert_model - pretrained BERT model
        dataset_split (str) - Dataset split for training / prediction (train/dev/test)
        overwrite_dial_file (bool) - Whether to generate a new file saving
            the dialogue examples overwrite_schema_emb_file,
        bert_ckpt_dir (str) - Directory containing pre-trained BERT checkpoint
        nf - NeuralModuleFactory
        mode(str): Schema embeddings initialization mode, baseline is ['CLS'] token embeddings
        from the last BERT layer
    """

    def __init__(
        self,
        data_dir,
        schema_embedding_dir,
        schema_config,
        tokenizer,
        bert_model,
        overwrite_schema_emb_files,
        bert_ckpt_dir,
        nf,
        add_carry_value,
        add_carry_status,
        datasets=['train', 'test', 'dev'],
        mode='baseline',
        is_trainable=False,
    ):

        # Dimension of the embedding for intents, slots and categorical slot values in
        # Maximum allowed number of categorical trackable slots for a service.
        self.schema_config = schema_config.copy()

        self.is_trainable = is_trainable
        self.datasets = datasets

        self._add_carry_value = add_carry_value
        self._add_carry_status = add_carry_status
        if self._add_carry_status:
            self._slot_status_size = 4
        else:
            self._slot_status_size = 3

        for dataset_split in ['train', 'test', 'dev']:
            if dataset_split not in self.datasets:
                logging.warning(
                    'WARNING: %s set was not included and won\'t be processed. Services from this dataset split '
                    + 'won\'t be supported',
                    dataset_split,
                )
        os.makedirs(schema_embedding_dir, exist_ok=True)

        tokenizer_type = type(tokenizer.tokenizer).__name__
        vocab_size = getattr(tokenizer, "vocab_size", 0)
        self.schema_embedding_file = os.path.join(
            schema_embedding_dir,
            "{}_{}_{}_{}_pretrained_schema_embedding.npy".format(
                '_'.join(self.datasets), mode, tokenizer_type, vocab_size
            ),
        )
        all_schema_json_paths = []
        for dataset_split in self.datasets:
            all_schema_json_paths.append(os.path.join(data_dir, dataset_split, "schema.json"))
        self.schemas = schema.Schema(
            all_schema_json_paths, add_carry_value=self._add_carry_value, add_carry_status=self._add_carry_status
        )

        if not os.path.exists(self.schema_embedding_file) or overwrite_schema_emb_files:
            # Generate the schema embeddings if needed or specified
            logging.info(f"Start generating the schema embeddings.")
            dataset_params = {
                "schema_config": schema_config,
                "tokenizer": tokenizer,
                "schemas": self.schemas,
            }
            emb_datalayer = BertInferDataLayer(
                dataset_type=SchemaEmbeddingDataset, dataset_params=dataset_params, batch_size=1, shuffle=False,
            )

            input_ids, input_mask, input_type_ids = emb_datalayer()

            hidden_states = bert_model(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
            evaluated_tensors = nf.infer(tensors=[hidden_states], checkpoint_dir=bert_ckpt_dir)

            master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if master_device:
                hidden_states = [concatenate(tensors) for tensors in evaluated_tensors]
                emb_datalayer.dataset.save_embeddings(hidden_states, self.schema_embedding_file, mode)
                logging.info(f"Finish generating the schema embeddings.")

        # wait until the master process writes to the schema embedding file
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        with open(self.schema_embedding_file, "rb") as f:
            self.schema_embeddings = np.load(f, allow_pickle=True)
            f.close()

    def get_schema_embeddings(self):
        # Convert from list of dict to dict of list
        schema_data_dict = collections.defaultdict(list)
        for service in self.schema_embeddings:
            schema_data_dict["cat_slot_emb"].append(service["cat_slot_emb"])
            schema_data_dict["cat_slot_value_emb"].append(service["cat_slot_value_emb"])
            schema_data_dict["noncat_slot_emb"].append(service["noncat_slot_emb"])
            schema_data_dict["req_slot_emb"].append(service["req_slot_emb"])
            schema_data_dict["intent_emb"].append(service["intent_emb"])
        return schema_data_dict

    def _get_schema_embedding_file_name(self):
        return self.schema_embedding_file

    def get_service_names_to_id_dict(self):
        return self.schemas._services_vocab

    def get_ids_to_service_names_dict(self):
        return self.schemas._services_id_to_vocab

    def update_slots_relation_list(self, slots_relation_list):
        self.schemas._slots_relation_list = slots_relation_list

    def get_slots_relation_list(self):
        return self.schemas._slots_relation_list
