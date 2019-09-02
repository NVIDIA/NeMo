import argparse
import os

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from pytorch_transformers import BertTokenizer
import nemo
import nemo_nlp
from nemo_nlp.callbacks.joint_intent_slot import \
    eval_iter_callback, eval_epochs_done_callback
from nemo_nlp.text_data_utils import \
    process_snips, process_atis, merge
from nemo_nlp import read_intent_slot_outputs


# Parsing arguments
parser = argparse.ArgumentParser(description='Joint-intent BERT')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_seq_length", default=50, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--fc_dropout", default=0.1, type=float)
parser.add_argument("--pretrained_bert_model",
                    default="bert-base-uncased",
                    type=str)
parser.add_argument("--dataset_name", default='snips-atis', type=str)
parser.add_argument("--data_dir",
                    default='data/nlu',
                    type=str)
parser.add_argument("--work_dir",
                    # default='outputs/ATIS/20190814-152523/checkpoints',
                    # default='outputs/SNIPS-ALL/20190821-154734/checkpoints',
                    default='outputs/SNIPS-ATIS/20190829-140622/checkpoints',
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
logger = nf.logger
# Load the pretrained BERT parameters
# pretrained_model can be one of:
# bert-base-uncased, bert-large-uncased, bert-base-cased,
# bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.

pretrained_bert_model = nf.get_module(
    name="huggingface.BERT",
    params={"pretrained_model_name": args.pretrained_bert_model,
            "local_rank": args.local_rank},
    collection="nemo_nlp",
    pretrained=True)

if args.dataset_name == 'atis':
    num_intents = 26
    num_slots = 129
    data_dir = process_atis(args.data_dir, args.do_lower_case)
    pad_label = num_slots - 1
elif args.dataset_name == 'snips-atis':
    data_dir, pad_label = merge(args.data_dir,
                                ['ATIS/nemo-processed-uncased',
                                 'snips/nemo-processed-uncased/all'],
                                args.dataset_name)
    num_intents = 41
    num_slots = 140
elif args.dataset_name.startswith('snips'):
    data_dir = process_snips(args.data_dir, args.do_lower_case)
    if args.dataset_name.endswith('light'):
        data_dir = f'{data_dir}/light'
        num_intents = 6
        num_slots = 4
    elif args.dataset_name.endswith('speak'):
        data_dir = f'{data_dir}/speak'
        num_intents = 9
        num_slots = 9
    elif args.dataset_name.endswith('all'):
        data_dir = f'{data_dir}/all'
        num_intents = 15
        num_slots = 12
    pad_label = num_slots - 1
else:
    nf.logger.info("Looks like you pass in the name of dataset that isn't "
                   "already supported by NeMo. Please make sure that you "
                   "build the preprocessing method for it.")
tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)
hidden_size = pretrained_bert_model.local_parameters["hidden_size"]

classifier = nemo_nlp.JointIntentSlotClassifier(hidden_size=hidden_size,
                                                num_intents=num_intents,
                                                num_slots=num_slots,
                                                dropout=args.fc_dropout)

loss_fn = nemo_nlp.JointIntentSlotLoss(num_slots=num_slots)


# Evaluation pipeline
logger.info("Loading eval data...")
data_layer = nemo_nlp.BertJointIntentSlotDataLayer(
    path_to_data=data_dir + '/test.tsv',
    path_to_slot=data_dir + '/test_slots.tsv',
    pad_label=num_slots-1,
    tokenizer=tokenizer,
    mode='eval',
    max_seq_length=args.max_seq_length,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    local_rank=args.local_rank
)

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
logger.info('Intent prediction results')
logger.info(classification_report(intents, pred_intents))

pred_slots = np.argmax(slot_logits, axis=2)
pred_slot_list, slot_list = [], []
for i, pred_slot in enumerate(pred_slots):
    pred_slot_list.extend(list(pred_slot[slot_masks[i]][1:-1]))
    slot_list.extend(list(slots[i][slot_masks[i]][1:-1]))
logger.info('Slot prediction results')
logger.info(classification_report(slot_list, pred_slot_list))
