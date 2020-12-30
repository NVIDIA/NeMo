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
https://github.com/google-research/google-research/blob/master/schema_guided_dst
"""
from typing import Dict, Optional

import numpy as np

from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, LabelsType, NeuralType

__all__ = ['SGDDataset']


class SGDDataset(Dataset):
    """Processes SGD dataset"""

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        return {
            "example_id_num": NeuralType(('B', 'T'), ChannelType()),
            "service_id": NeuralType(('B'), ChannelType()),
            "utterance_ids": NeuralType(('B', 'T'), ChannelType()),
            "token_type_ids": NeuralType(('B', 'T'), ChannelType()),  # utterance segment
            "attention_mask": NeuralType(('B', 'T'), ChannelType()),  # utterance mask
            "intent_status": NeuralType(('B'), LabelsType()),
            "requested_slot_status": NeuralType(('B'), LabelsType()),
            "categorical_slot_status": NeuralType(('B'), LabelsType()),
            "categorical_slot_value_status": NeuralType(('B'), LabelsType()),
            "noncategorical_slot_status": NeuralType(('B'), LabelsType()),
            "noncategorical_slot_value_start": NeuralType(('B'), LabelsType()),
            "noncategorical_slot_value_end": NeuralType(('B'), LabelsType()),
            "start_char_idx": NeuralType(('B', 'T'), LabelsType()),
            "end_char_idx": NeuralType(('B', 'T'), LabelsType()),
            "task_mask": NeuralType(('B', 'T'), ChannelType()),
        }

    def __init__(self, dataset_split: str, dialogues_processor: object):
        """ Constructor
        Args:
            dataset_split: dataset split
            dialogues_processor: Data generator for SGD dialogues
        """
        self.features = dialogues_processor.get_dialog_examples(dataset_split)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        ex = self.features[idx]
        service_id = ex.service_id

        return (
            np.array(ex.example_id_num),
            np.array(service_id),
            np.array(ex.utterance_ids),
            np.array(ex.utterance_segment),
            np.array(ex.utterance_mask, dtype=np.long),
            np.array(ex.intent_status, dtype=np.float32),
            np.array(ex.requested_slot_status, dtype=np.float32),
            np.array(ex.categorical_slot_status),
            np.array(ex.categorical_slot_value_status, dtype=np.float32),
            np.array(ex.noncategorical_slot_status),
            np.array(ex.noncategorical_slot_value_start),
            np.array(ex.noncategorical_slot_value_end),
            np.array(ex.start_char_idx),  # noncat_alignment_start
            np.array(ex.end_char_idx),  # noncat_alignment_end
            np.array(ex.task_mask),  # noncat_alignment_end
        )
