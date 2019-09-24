import argparse
import os

import numpy as np
from pytorch_transformers import BertTokenizer
from sklearn.metrics import confusion_matrix, classification_report

import nemo
import nemo_nlp
from nemo_nlp.text_data_utils import JointIntentSlotDataDesc


# Parsing arguments
parser = argparse.ArgumentParser(description='Joint-intent BERT')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_seq_length", default=50, type=int)
parser.add_argument("--fc_dropout", default=0.1, type=float)
parser.add_argument("--pretrained_bert_model",
                    default="bert-base-uncased",
                    type=str)
parser.add_argument("--dataset_name", default='atis', type=str)
parser.add_argument("--data_dir",
                    default='data/nlu/ATIS',
                    type=str)
parser.add_argument("--work_dir",
                    default='outputs/ATIS/20190814-152523/checkpoints',
                    type=str)
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
    pretrained_model_name=args.pretrained_bert_model, factory=nf)
hidden_size = pretrained_bert_model.local_parameters["hidden_size"]
tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)


data_desc = JointIntentSlotDataDesc(
    args.dataset_name, args.data_dir, args.do_lower_case)

dataset = nemo_nlp.BertJointIntentSlotDataset(
    input_file=data_desc.eval_file,
    slot_file=data_desc.eval_slot_file,
    pad_label=data_desc.pad_label,
    tokenizer=tokenizer,
    max_seq_length=args.max_seq_length,
    shuffle=False)

classifier = nemo_nlp.JointIntentSlotClassifier(
    hidden_size=hidden_size,
    num_intents=data_desc.num_intents,
    num_slots=data_desc.num_slots,
    dropout=args.fc_dropout)

loss_fn = nemo_nlp.JointIntentSlotLoss(num_slots=data_desc.num_slots)

# Evaluation pipeline
nf.logger.info("Loading eval data...")
data_layer = nemo_nlp.BertJointIntentSlotDataLayer(
    dataset,
    batch_size=args.batch_size,
    num_workers=0,
    local_rank=args.local_rank)

ids, type_ids, input_mask, slot_mask, intents, slots = data_layer()

hidden_states = pretrained_bert_model(input_ids=ids,
                                      token_type_ids=type_ids,
                                      attention_mask=input_mask)

intent_logits, slot_logits = classifier(hidden_states=hidden_states)

###########################################################################


# Instantiate an optimizer to perform `infer` action
evaluated_tensors = nf.infer(
    tensors=[intent_logits, slot_logits, slot_mask, intents, slots],
    checkpoint_dir=args.work_dir,
)


def concatenate(lists):
    return np.concatenate([t.cpu() for t in lists])


def get_preds(logits):
    return np.argmax(logits, 1)


intent_logits = concatenate(evaluated_tensors[0])
slot_logits = concatenate(evaluated_tensors[1])
slot_masks = concatenate(evaluated_tensors[2])
intents = concatenate(evaluated_tensors[3])
slots = concatenate(evaluated_tensors[4])

pred_intents = np.argmax(intent_logits, 1)
nf.logger.info('Intent prediction results')
nf.logger.info(classification_report(intents, pred_intents))

pred_slots = np.argmax(slot_logits, axis=2)
pred_slot_list, slot_list = [], []
for i, pred_slot in enumerate(pred_slots):
    pred_slot_list.extend(list(pred_slot[slot_masks[i]][1:-1]))
    slot_list.extend(list(slots[i][slot_masks[i]][1:-1]))
nf.logger.info('Slot prediction results')
nf.logger.info(classification_report(slot_list, pred_slot_list))
