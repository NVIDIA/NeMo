import argparse
import os

import numpy as np
from pytorch_transformers import BertTokenizer
from sklearn.metrics import confusion_matrix, classification_report

import nemo
import nemo_nlp
from nemo_nlp.data.datasets.utils import JointIntentSlotDataDesc


# Parsing arguments
parser = argparse.ArgumentParser(description='Joint-intent BERT')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--max_seq_length", default=50, type=int)
parser.add_argument("--pretrained_bert_model",
                    default="bert-base-uncased",
                    type=str)
parser.add_argument("--dataset_name", default='snips-all', type=str)
parser.add_argument("--data_dir", default='data/nlu/snips', type=str)
parser.add_argument("--work_dir",
                    default='outputs/SNIPS-ALL/20191014-104316/checkpoints',
                    type=str)
parser.add_argument("--eval_file_prefix", default='test', type=str)
parser.add_argument("--amp_opt_level", default="O0",
                    type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--do_lower_case", action='store_false')

args = parser.parse_args()

if not os.path.exists(args.data_dir):
    raise ValueError(f'Data not found at {args.data_dir}')

nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                   local_rank=args.local_rank,
                                   optimization_level=args.amp_opt_level,
                                   log_dir=None)

""" Load the pretrained BERT parameters
See the list of pretrained models, call:
nemo_nlp.huggingface.BERT.list_pretrained_models()
"""
pretrained_bert_model = nemo_nlp.huggingface.BERT(
    pretrained_model_name=args.pretrained_bert_model)
hidden_size = pretrained_bert_model.local_parameters["hidden_size"]
tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)


data_desc = JointIntentSlotDataDesc(args.data_dir,
                                    args.do_lower_case,
                                    args.dataset_name)

# Evaluation pipeline
nf.logger.info("Loading eval data...")
data_layer = nemo_nlp.BertJointIntentSlotDataLayer(
    input_file=f'{data_desc.data_dir}/{args.eval_file_prefix}.tsv',
    slot_file=f'{data_desc.data_dir}/{args.eval_file_prefix}_slots.tsv',
    pad_label=data_desc.pad_label,
    tokenizer=tokenizer,
    max_seq_length=args.max_seq_length,
    shuffle=False,
    batch_size=args.batch_size,
    num_workers=0,
    local_rank=args.local_rank)

classifier = nemo_nlp.JointIntentSlotClassifier(
    hidden_size=hidden_size,
    num_intents=data_desc.num_intents,
    num_slots=data_desc.num_slots)

ids, type_ids, input_mask, loss_mask, subtokens_mask, intents, slots = data_layer()

hidden_states = pretrained_bert_model(input_ids=ids,
                                      token_type_ids=type_ids,
                                      attention_mask=input_mask)
intent_logits, slot_logits = classifier(hidden_states=hidden_states)

###########################################################################


# Instantiate an optimizer to perform `infer` action
evaluated_tensors = nf.infer(
    tensors=[intent_logits, slot_logits,
             loss_mask, subtokens_mask,
             intents, slots],
    checkpoint_dir=args.work_dir,
)


def concatenate(lists):
    return np.concatenate([t.cpu() for t in lists])


def get_preds(logits):
    return np.argmax(logits, 1)


intent_logits, slot_logits, loss_mask, subtokens_mask, intents, slot_labels =\
    [concatenate(tensors) for tensors in evaluated_tensors]


pred_intents = np.argmax(intent_logits, 1)
nf.logger.info('Intent prediction results')

intents = np.asarray(intents)
pred_intents = np.asarray(pred_intents)
intent_accuracy = sum(intents == pred_intents) / len(pred_intents)
nf.logger.info(f'Intent accuracy: {intent_accuracy}')
nf.logger.info(classification_report(intents, pred_intents))


slot_preds = np.argmax(slot_logits, axis=2)
slot_preds_list, slot_labels_list = [], []
for i, sp in enumerate(slot_preds):
    slot_preds_list.extend(list(slot_preds[i][subtokens_mask[i]]))
    slot_labels_list.extend(list(slot_labels[i][subtokens_mask[i]]))

nf.logger.info('Slot prediction results')
slot_labels_list = np.asarray(slot_labels_list)
slot_preds_list = np.asarray(slot_preds_list)
slot_accuracy = sum(slot_labels_list == slot_preds_list) / \
                len(slot_labels_list)
nf.logger.info(f'Slot accuracy: {slot_accuracy}')
nf.logger.info(classification_report(slot_labels_list, slot_preds_list))
