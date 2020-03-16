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

import argparse

import numpy as np

import nemo
import nemo.collections.nlp as nemo_nlp
from nemo.collections.nlp.data.datasets.joint_intent_slot_dataset import (
    JointIntentSlotDataDesc,
    read_intent_slot_outputs,
)
from nemo.collections.nlp.nm.data_layers import BertJointIntentSlotInferDataLayer
from nemo.collections.nlp.nm.trainables import JointIntentSlotClassifier

# Parsing arguments
parser = argparse.ArgumentParser(description='Joint-intent BERT')
parser.add_argument("--query", required=True, type=str)
parser.add_argument("--data_dir", default='data/atis', type=str)
parser.add_argument("--checkpoint_dir", required=True, help="path to your checkpoint folder", type=str)
parser.add_argument("--pretrained_model_name", default="bert-base-uncased", type=str)
parser.add_argument("--bert_config", default=None, type=str)
parser.add_argument("--do_lower_case", action='store_false')
parser.add_argument("--max_seq_length", default=64, type=int)

args = parser.parse_args()

nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch)

pretrained_bert_model = nemo_nlp.nm.trainables.get_huggingface_model(
    bert_config=args.bert_config, pretrained_model_name=args.pretrained_model_name
)
tokenizer = nemo_nlp.data.NemoBertTokenizer(pretrained_model=args.pretrained_model_name)

hidden_size = pretrained_bert_model.hidden_size

data_desc = JointIntentSlotDataDesc(data_dir=args.data_dir)

query = args.query
if args.do_lower_case:
    query = query.lower()

data_layer = BertJointIntentSlotInferDataLayer(
    queries=[query], tokenizer=tokenizer, max_seq_length=args.max_seq_length, batch_size=1
)

# Create sentence classification loss on top
classifier = JointIntentSlotClassifier(
    hidden_size=hidden_size, num_intents=data_desc.num_intents, num_slots=data_desc.num_slots, dropout=0.0
)

input_data = data_layer()

hidden_states = pretrained_bert_model(
    input_ids=input_data.input_ids, token_type_ids=input_data.input_type_ids, attention_mask=input_data.input_mask
)

intent_logits, slot_logits = classifier(hidden_states=hidden_states)

###########################################################################


evaluated_tensors = nf.infer(
    tensors=[intent_logits, slot_logits, input_data.subtokens_mask], checkpoint_dir=args.checkpoint_dir
)


def concatenate(lists):
    return np.concatenate([t.cpu() for t in lists])


intent_logits, slot_logits, subtokens_mask = [concatenate(tensors) for tensors in evaluated_tensors]

read_intent_slot_outputs(
    [query], data_desc.intent_dict_file, data_desc.slot_dict_file, intent_logits, slot_logits, subtokens_mask
)
