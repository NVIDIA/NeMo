# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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

import numpy as np

from nemo import logging
from nemo.collections.nlp.utils import get_vocab

__all__ = ['read_intent_slot_outputs']


def read_intent_slot_outputs(
    queries, intent_file, slot_file, intent_logits, slot_logits, slot_masks, intents=None, slots=None
):
    intent_dict = get_vocab(intent_file)
    slot_dict = get_vocab(slot_file)
    pred_intents = np.argmax(intent_logits, 1)
    pred_slots = np.argmax(slot_logits, axis=2)
    slot_masks = slot_masks > 0.5
    for i, query in enumerate(queries):
        logging.info(f'Query: {query}')
        pred = pred_intents[i]
        logging.info(f'Predicted intent:\t{pred}\t{intent_dict[pred]}')
        if intents is not None:
            logging.info(f'True intent:\t{intents[i]}\t{intent_dict[intents[i]]}')

        pred_slot = pred_slots[i][slot_masks[i]]
        tokens = query.strip().split()

        if len(pred_slot) != len(tokens):
            raise ValueError('Pred_slot and tokens must be of the same length')

        for j, token in enumerate(tokens):
            output = f'{token}\t{slot_dict[pred_slot[j]]}'
            if slots is not None:
                output = f'{output}\t{slot_dict[slots[i][j]]}'
            logging.info(output)
