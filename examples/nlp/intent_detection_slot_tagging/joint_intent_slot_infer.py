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
import os

import numpy as np
from sklearn.metrics import classification_report

import nemo
import nemo.collections.nlp as nemo_nlp
from nemo import logging
from nemo.collections.nlp.data.datasets.joint_intent_slot_dataset import JointIntentSlotDataDesc
from nemo.collections.nlp.nm.data_layers import BertJointIntentSlotDataLayer
from nemo.collections.nlp.nm.trainables.joint_intent_slot import JointIntentSlotClassifier

# Parsing arguments
parser = argparse.ArgumentParser(description='Batch inference for intent detection/slot tagging with BERT')
parser.add_argument("--checkpoint_dir", required=True, help="your checkpoint folder", type=str)
parser.add_argument("--data_dir", default='data/atis', type=str)
parser.add_argument("--eval_file_prefix", default='test', type=str)
parser.add_argument("--pretrained_model_name", default="bert-base-uncased", type=str)
parser.add_argument("--bert_config", default=None, type=str)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--do_lower_case", action='store_false')
parser.add_argument("--max_seq_length", default=64, type=int)
parser.add_argument("--local_rank", default=None, type=int)

args = parser.parse_args()

if not os.path.exists(args.data_dir):
    raise ValueError(f'Data not found at {args.data_dir}')

nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch, local_rank=args.local_rank)

pretrained_bert_model = nemo_nlp.nm.trainables.get_huggingface_model(
    bert_config=args.bert_config, pretrained_model_name=args.pretrained_model_name
)
tokenizer = nemo_nlp.data.NemoBertTokenizer(pretrained_model=args.pretrained_model_name)

hidden_size = pretrained_bert_model.hidden_size

data_desc = JointIntentSlotDataDesc(data_dir=args.data_dir)

# Evaluation pipeline
logging.info("Loading eval data...")
data_layer = BertJointIntentSlotDataLayer(
    input_file=f'{data_desc.data_dir}/{args.eval_file_prefix}.tsv',
    slot_file=f'{data_desc.data_dir}/{args.eval_file_prefix}_slots.tsv',
    pad_label=data_desc.pad_label,
    tokenizer=tokenizer,
    max_seq_length=args.max_seq_length,
    shuffle=False,
    batch_size=args.batch_size,
)

classifier = JointIntentSlotClassifier(
    hidden_size=hidden_size, num_intents=data_desc.num_intents, num_slots=data_desc.num_slots
)

input_data = data_layer()

hidden_states = pretrained_bert_model(
    input_ids=input_data.input_ids, token_type_ids=input_data.input_type_ids, attention_mask=input_data.input_mask
)
intent_logits, slot_logits = classifier(hidden_states=hidden_states)

###########################################################################


# Instantiate an optimizer to perform `infer` action
evaluated_tensors = nf.infer(
    tensors=[
        intent_logits,
        slot_logits,
        input_data.loss_mask,
        input_data.subtokens_mask,
        input_data.intents,
        input_data.slots,
    ],
    checkpoint_dir=args.checkpoint_dir,
)


def concatenate(lists):
    return np.concatenate([t.cpu() for t in lists])


def get_preds(logits):
    return np.argmax(logits, 1)


intent_logits, slot_logits, loss_mask, subtokens_mask, intents, slot_labels = [
    concatenate(tensors) for tensors in evaluated_tensors
]

pred_intents = np.argmax(intent_logits, 1)
logging.info('Intent prediction results')

intents = np.asarray(intents)
pred_intents = np.asarray(pred_intents)
intent_accuracy = sum(intents == pred_intents) / len(pred_intents)
logging.info(f'Intent accuracy: {intent_accuracy}')
logging.info(classification_report(intents, pred_intents))

slot_preds = np.argmax(slot_logits, axis=2)
slot_preds_list, slot_labels_list = [], []
subtokens_mask = subtokens_mask > 0.5
for i, sp in enumerate(slot_preds):
    slot_preds_list.extend(list(slot_preds[i][subtokens_mask[i]]))
    slot_labels_list.extend(list(slot_labels[i][subtokens_mask[i]]))

logging.info('Slot prediction results')
slot_labels_list = np.asarray(slot_labels_list)
slot_preds_list = np.asarray(slot_preds_list)
slot_accuracy = sum(slot_labels_list == slot_preds_list) / len(slot_labels_list)
logging.info(f'Slot accuracy: {slot_accuracy}')
logging.info(classification_report(slot_labels_list, slot_preds_list))
