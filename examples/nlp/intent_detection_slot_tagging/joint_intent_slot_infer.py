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
from sklearn.metrics import confusion_matrix

import nemo
import nemo.collections.nlp as nemo_nlp
from nemo import logging
from nemo.collections.nlp.data.datasets.joint_intent_slot_dataset import JointIntentSlotDataDesc
from nemo.collections.nlp.nm.data_layers import BertJointIntentSlotDataLayer
from nemo.collections.nlp.nm.trainables.joint_intent_slot import JointIntentSlotClassifier
from nemo.collections.nlp.utils.callback_utils import get_classification_report, get_f1_scores, plot_confusion_matrix

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


def log_misclassified_queries(intent_labels, intent_preds, queries, intent_dict, limit=50):
    logging.info(f'*** Misclassified intent queries (limit {limit}) ***')
    cnt = 0
    for i in range(len(intent_preds)):
        if intent_labels[i] != intent_preds[i]:
            query = queries[i].split('\t')[0]
            logging.info(f'{query} - intent: {intent_dict[intent_labels[i]]} - predicted: {intent_dict[intent_preds[i]]}')
            cnt = cnt + 1
            if cnt >= limit:
                break


def log_misclassified_slots(intent_labels, intent_preds, slot_labels, slot_preds, subtokens_mask, queries, intent_dict, slot_dict, limit=50):
    logging.info('');
    logging.info(f'*** Misclassified slots queries (limit {limit}) ***')
    # print slot dictionary
    str = ''
    for i, slot in enumerate(slot_dict):
        str += f'{i} - {slot}, '
        if i % 5 == 4 or i == len(slot_dict)-1:
            logging.info(str)
            str = ''

    logging.info('----------------');
    cnt = 0
    for i in range(len(intent_preds)):
        cur_slot_pred = slot_preds[i][subtokens_mask[i]]
        cur_slot_label = slot_labels[i][subtokens_mask[i]]
        if not np.all(cur_slot_pred == cur_slot_label):
            query = queries[i].split('\t')[0]
            logging.info(f'{query} - intent: {intent_dict[intent_labels[i]]} - predicted: {intent_dict[intent_preds[i]]}')
            logging.info(cur_slot_pred)
            logging.info(cur_slot_label)
            cnt = cnt + 1
            if cnt >= limit:
                break


# check non compliance of B- and I- slots,
def check_problematic_slots(slot_preds_list, slot_dict):
    slot_dict = open(slot_dict, 'r').readlines()
    cnt = 0

    # for sentence in slot_preds:
        # slots = sentence.split(" ")
    sentence = slot_preds_list
    for i in range(len(sentence)):
        slot_name = slot_dict[int(sentence[i])]
        if slot_name.startswith("I-"):
            prev_slot_name = slot_dict[int(sentence[i - 1])]
            if slot_name[2:] != prev_slot_name[2:]:
                print("Problem: " + slot_name + " - " + prev_slot_name)
                cnt += 1
    print("Total problematic slots: " + str(cnt))


def errors_per_class(cm, dict):
    size = cm.shape[0]
    confused_per_class = {}
    total_errors = 0
    for class_num in range(size):
        sum = 0;
        for i in range(size):
            if i != class_num:
                sum += cm[class_num][i]
                sum += cm[i][class_num]
        confused_per_class[dict[class_num]] = sum
        total_errors += sum
        # logging.info(f'{dict[class_num]} - {sum}')

    logging.info(f'Total errors (multiplied by 2): {total_errors}')
    sorted_confused_per_class = sorted(confused_per_class.items(), key=lambda x: x[1], reverse=True)
    for conf_str in sorted_confused_per_class:
        logging.info(conf_str)


def analyze_confusion_matrix(cm, dict, max_pairs=10):
    threshold = 5
    confused_pairs = {}
    size = cm.shape[0]
    for i in range(size):
        res = cm[i].argsort()
        for j in range(size):
            pos = res[size-j-1]
            # no confusion - same row and column
            if pos == i:
                continue
            elif cm[i][pos] >= threshold:
                str = f'{dict[i]} -> {dict[pos]}'
                confused_pairs[str] = cm[i][pos]
            else:
                break

    # sort my max confusions and print first max_pairs
    sorted_confused_pairs = sorted(confused_pairs.items(), key=lambda x: x[1], reverse=True)
    for i, pair_str in enumerate(sorted_confused_pairs):
        if i>= max_pairs:
            break;
        logging.info(pair_str)


intent_logits, slot_logits, loss_mask, subtokens_mask, intent_labels, slot_labels_unmasked = [
    concatenate(tensors) for tensors in evaluated_tensors
]

# --- analyse of the results ---
# slot accuracies
logging.info('Slot Prediction Results:')
slot_preds_unmasked = np.argmax(slot_logits, axis=2)
subtokens_mask = subtokens_mask > 0.5
slot_labels = slot_labels_unmasked[subtokens_mask]
slot_preds = slot_preds_unmasked[subtokens_mask]
slot_accuracy = np.mean(slot_labels == slot_preds)
logging.info(f'Slot Accuracy: {slot_accuracy}')
f1_scores = get_f1_scores(slot_labels, slot_preds, average_modes=['weighted', 'macro', 'micro'])
for k, v in f1_scores.items():
    logging.info(f'{k}: {v}')

logging.info(f'\n {get_classification_report(slot_labels, slot_preds, label_ids=data_desc.slots_label_ids)}')

# intent accuracies
logging.info('Intent Prediction Results:')
intent_preds = np.asarray(np.argmax(intent_logits, 1))
intent_labels = np.asarray(intent_labels)
intent_accuracy = np.mean(intent_labels == intent_preds)
logging.info(f'Intent Accuracy: {intent_accuracy}')
f1_scores = get_f1_scores(intent_labels, intent_preds, average_modes=['weighted', 'macro', 'micro'])
for k, v in f1_scores.items():
    logging.info(f'{k}: {v}')

logging.info(f'\n {get_classification_report(intent_labels, intent_preds, label_ids=data_desc.intents_label_ids)}')

# print queries with wrong intent:
queries = open(f'{data_desc.data_dir}/{args.eval_file_prefix}.tsv', 'r').readlines()[1:]
intent_dict = open(data_desc.intent_dict_file, 'r').read().splitlines()
log_misclassified_queries(intent_labels, intent_preds, queries, intent_dict, limit=30)

# print queries with wrong slots:
slot_dict = open(data_desc.slot_dict_file, 'r').read().splitlines()
log_misclassified_slots(intent_labels, intent_preds, slot_labels_unmasked, slot_preds_unmasked, subtokens_mask, queries, intent_dict, slot_dict, limit=30)

# analyze confusion matrices
print('')
logging.info(f'*** Most Confused Intents ***')
cm = confusion_matrix(intent_labels, intent_preds)
analyze_confusion_matrix(cm, intent_dict, max_pairs=20)

logging.info(f'*** Intent errors per class (in both directions) ***')
errors_per_class(cm, intent_dict)

print('')
logging.info(f'*** Most Confused Slots ***')
cm = confusion_matrix(slot_labels, slot_preds, np.arange(len(slot_dict)))
analyze_confusion_matrix(cm, slot_dict, max_pairs=20)


# does not work well for large matrices
# print(f'Intent Confusion matrix:\n{cm}')

# check potentially problematic slots - when I- label comes after different B- label
# slot_dict_file = f'{data_desc.data_dir}/dict.slots.csv'
# check_problematic_slots(slot_labels, slot_dict_file)