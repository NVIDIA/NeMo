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
import numpy as np
from torch.utils.data import Dataset

__all__ = ['SGDDataset']


class SGDDataset(Dataset):
    """ 
    Processes SGD dataset
    Args:
        dataset_split (str): train/dev/test
        dialogues_processor (obj): Data generator for SGD dialogues
    """

    def __init__(self, dataset_split, dialogues_processor):
        self.features = dialogues_processor.get_dialog_examples(dataset_split)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        ex = self.features[idx]
        service_id = ex.service_schema.service_id

        return (
            np.array(ex.example_id_num),
            np.array(service_id),
            np.array(ex.is_real_example, dtype=int),
            np.array(ex.utterance_ids),
            np.array(ex.utterance_segment),
            np.array(ex.utterance_mask, dtype=np.long),
            np.array(ex.categorical_slot_status),
            np.array(ex.cat_slot_status_mask),
            np.array(ex.categorical_slot_values),
            np.array(ex.cat_slot_values_mask),
            np.array(ex.noncategorical_slot_status),
            np.array(ex.noncat_slot_status_mask),
            np.array(ex.noncategorical_slot_value_start),
            np.array(ex.noncategorical_slot_value_end),
            np.array(ex.start_char_idx),  # noncat_alignment_start
            np.array(ex.end_char_idx),  # noncat_alignment_end
            np.array(ex.num_slots),  # num_requested_slots
            np.array(ex.requested_slot_status, dtype=np.float32),
            np.array(ex.requested_slot_mask),
            np.array(ex.intent_status_mask),
            np.array(ex.intent_status_labels),
        )
