import argparse
import os

import numpy as np
from sklearn.metrics import classification_report

import nemo
import nemo.collections.nlp as nemo_nlp
from nemo.collections.nlp import NemoBertTokenizer
from nemo.collections.nlp.utils.nlp_utils import get_vocab

# Parsing arguments
parser = argparse.ArgumentParser(description='NER with pretrained BERT')
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--fc_dropout", default=0, type=float)
parser.add_argument("--punct_num_fc_layers", default=3, type=int)
parser.add_argument("--pretrained_bert_model", default="bert-base-uncased", type=str)
parser.add_argument("--none_label", default='O', type=str)
parser.add_argument(
    "--queries",
    action='append',
    default=[
        'we bought four shirts from the ' + 'nvidia gear store in santa clara',
        'nvidia is a company',
        'can i help you',
        'how are you',
        'how\'s the weather today',
        'okay',
        'we bought four shirts one mug and ten '
        + 'thousand titan rtx graphics cards the more '
        + 'you buy the more you save',
    ],
    help="Example: --queries 'san francisco' --queries 'la'",
)
parser.add_argument(
    "--add_brackets",
    action='store_false',
    help="Whether to take predicted label in brackets or \
                    just append to word in the output",
)
parser.add_argument("--checkpoints_dir", default='output/checkpoints', type=str)
parser.add_argument(
    "--punct_labels_dict",
    default='punct_label_ids.csv',
    type=str,
    help='This file is generated during training \
                    when the datalayer is created',
)
parser.add_argument(
    "--capit_labels_dict",
    default='capit_label_ids.csv',
    type=str,
    help='This file is generated during training \
                    when the datalayer is created',
)
parser.add_argument("--amp_opt_level", default="O0", type=str, choices=["O0", "O1", "O2"])

args = parser.parse_args()

if not os.path.exists(args.checkpoints_dir):
    raise ValueError(f'Checkpoints folder not found at {args.checkpoints_dir}')
if not (os.path.exists(args.punct_labels_dict) and os.path.exists(args.capit_labels_dict)):
    raise ValueError(
        f'Dictionary with ids to labels not found at {args.punct_labels_dict} \
         or {args.punct_labels_dict}'
    )

nf = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch, optimization_level=args.amp_opt_level, log_dir=None
)

punct_labels_dict = get_vocab(args.punct_labels_dict)

capit_labels_dict = get_vocab(args.capit_labels_dict)

""" Load the pretrained BERT parameters
See the list of pretrained models, call:
nemo_nlp.BERT.list_pretrained_models()
"""
pretrained_bert_model = nemo_nlp.nm.trainables.huggingface.BERT(pretrained_model_name=args.pretrained_bert_model)
hidden_size = pretrained_bert_model.local_parameters["hidden_size"]
tokenizer = NemoBertTokenizer(args.pretrained_bert_model)

data_layer = nemo_nlp.nm.data_layers.BertTokenClassificationInferDataLayer(
    queries=args.queries, tokenizer=tokenizer, max_seq_length=args.max_seq_length, batch_size=1
)

punct_classifier = nemo_nlp.nm.trainables.TokenClassifier(
    hidden_size=hidden_size,
    num_classes=len(punct_labels_dict),
    dropout=args.fc_dropout,
    num_layers=args.punct_num_fc_layers,
    name='Punctuation',
)

capit_classifier = nemo_nlp.nm.trainables.TokenClassifier(
    hidden_size=hidden_size, num_classes=len(capit_labels_dict), dropout=args.fc_dropout, name='Capitalization'
)

input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask = data_layer()

hidden_states = pretrained_bert_model(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

punct_logits = punct_classifier(hidden_states=hidden_states)
capit_logits = capit_classifier(hidden_states=hidden_states)

###########################################################################

# Instantiate an optimizer to perform `infer` action
evaluated_tensors = nf.infer(tensors=[punct_logits, capit_logits, subtokens_mask], checkpoint_dir=args.checkpoints_dir)


def concatenate(lists):
    return np.concatenate([t.cpu() for t in lists])


def get_preds(logits):
    return np.argmax(logits, 1)


punct_logits, capit_logits, subtokens_mask = [concatenate(tensors) for tensors in evaluated_tensors]

punct_preds = np.argmax(punct_logits, axis=2)
capit_preds = np.argmax(capit_logits, axis=2)

for i, query in enumerate(args.queries):
    nemo.logging.info(f'Query: {query}')

    punct_pred = punct_preds[i][subtokens_mask[i] > 0.5]
    capit_pred = capit_preds[i][subtokens_mask[i] > 0.5]
    words = query.strip().split()
    if len(punct_pred) != len(words) or len(capit_pred) != len(words):
        raise ValueError('Pred and words must be of the same length')

    output = ''
    for j, w in enumerate(words):
        punct_label = punct_labels_dict[punct_pred[j]]
        capit_label = capit_labels_dict[capit_pred[j]]

        if capit_label != args.none_label:
            w = w.capitalize()
        output += w
        if punct_label != args.none_label:
            output += punct_label
        output += ' '
    nemo.logging.info(f'Combined: {output.strip()}\n')
