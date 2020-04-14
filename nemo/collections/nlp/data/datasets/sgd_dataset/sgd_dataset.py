"""
This code were adapted from 
https://github.com/google-research/google-research/tree/master/schema_guided_dst
"""

import numpy as np
from torch.utils.data import Dataset

from nemo.collections.nlp.data.datasets.sgd_dataset.schema_embedding_dataset import SchemaEmbeddingDataset

__all__ = ['SGDDataset']


class SGDDataset(Dataset):
    """ 
    TODO
    """

    def __init__(self, dataset_split, schema_emb_processor, dialogues_processor):
        self.features = dialogues_processor.get_dialog_examples(dataset_split)
        self.trainable_schema_emb = schema_emb_processor.is_trainable
        self.schema_embeddings = schema_emb_processor.get_schema_embeddings()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        ex = self.features[idx]
        service_id = ex.service_schema.service_id
    
        example = [
            np.array(ex.example_id_num),
            np.array(service_id),
            np.array(ex.is_real_example, dtype=int),
            np.array(ex.utterance_ids),
            np.array(ex.utterance_segment),
            np.array(ex.utterance_mask, dtype=np.long),
            np.array(ex.num_categorical_slots),
            np.array(ex.categorical_slot_status),
            np.array(ex.num_categorical_slot_values),
            np.array(ex.categorical_slot_values),
            np.array(ex.num_noncategorical_slots),
            np.array(ex.noncategorical_slot_status),
            np.array(ex.noncategorical_slot_value_start),
            np.array(ex.noncategorical_slot_value_end),
            np.array(ex.start_char_idx),  # noncat_alignment_start
            np.array(ex.end_char_idx),  # noncat_alignment_end
            np.array(ex.num_slots),  # num_requested_slots
            np.array(ex.requested_slot_status, dtype=np.float32),
            np.array(ex.num_intents),
            np.array(ex.intent_status),
        ]

        if not self.trainable_schema_emb:
            return example + [
                np.array(self.schema_embeddings['cat_slot_emb'][service_id], dtype=np.float32),
                np.array(self.schema_embeddings['cat_slot_value_emb'][service_id], dtype=np.float32),
                np.array(self.schema_embeddings['noncat_slot_emb'][service_id], dtype=np.float32),
                np.array(self.schema_embeddings['req_slot_emb'][service_id], dtype=np.float32),
                np.array(self.schema_embeddings['intent_emb'][service_id], dtype=np.float32),
            ]

        return example
