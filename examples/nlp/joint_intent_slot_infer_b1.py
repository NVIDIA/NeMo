import argparse

import numpy as np
from transformers import BertTokenizer

import nemo
import nemo.collections.nlp as nemo_nlp
from nemo.collections.nlp.data.datasets.utils import JointIntentSlotDataDesc
from nemo.collections.nlp.utils.nlp_utils import read_intent_slot_outputs

# Parsing arguments
parser = argparse.ArgumentParser(description='Joint-intent BERT')
parser.add_argument("--max_seq_length", default=50, type=int)
parser.add_argument("--fc_dropout", default=0.1, type=float)
parser.add_argument("--pretrained_bert_model", default="bert-base-uncased", type=str)
parser.add_argument("--dataset_name", default='snips-all', type=str)
parser.add_argument("--data_dir", default='data/nlu/snips', type=str)
parser.add_argument("--query", default='please turn on the light', type=str)
parser.add_argument("--work_dir", required=True, help="your checkpoint folder", type=str)
parser.add_argument("--amp_opt_level", default="O0", type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--do_lower_case", action='store_false')

args = parser.parse_args()

nf = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch, optimization_level=args.amp_opt_level, log_dir=None,
)

""" Load the pretrained BERT parameters
See the list of pretrained models, call:
nemo_nlp.huggingface.BERT.list_pretrained_models()
"""
pretrained_bert_model = nemo_nlp.huggingface.BERT(pretrained_model_name=args.pretrained_bert_model, factory=nf)
tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)
hidden_size = pretrained_bert_model.local_parameters["hidden_size"]

data_desc = JointIntentSlotDataDesc(args.data_dir, args.do_lower_case, args.dataset_name)

query = args.query
if args.do_lower_case:
    query = query.lower()

data_layer = nemo_nlp.BertJointIntentSlotInferDataLayer(
    queries=[query], tokenizer=tokenizer, max_seq_length=args.max_seq_length, batch_size=1,
)

# Create sentence classification loss on top
classifier = nemo_nlp.JointIntentSlotClassifier(
    hidden_size=hidden_size, num_intents=data_desc.num_intents, num_slots=data_desc.num_slots, dropout=args.fc_dropout,
)

ids, type_ids, input_mask, loss_mask, subtokens_mask = data_layer()

hidden_states = pretrained_bert_model(input_ids=ids, token_type_ids=type_ids, attention_mask=input_mask)

intent_logits, slot_logits = classifier(hidden_states=hidden_states)

###########################################################################


evaluated_tensors = nf.infer(tensors=[intent_logits, slot_logits, subtokens_mask], checkpoint_dir=args.work_dir,)


def concatenate(lists):
    return np.concatenate([t.cpu() for t in lists])


intent_logits, slot_logits, subtokens_mask = [concatenate(tensors) for tensors in evaluated_tensors]

read_intent_slot_outputs(
    [query], data_desc.intent_dict_file, data_desc.slot_dict_file, intent_logits, slot_logits, subtokens_mask,
)
