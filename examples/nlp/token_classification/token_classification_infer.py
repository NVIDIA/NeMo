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

import nemo
import nemo.collections.nlp as nemo_nlp
from nemo import logging
from nemo.collections.nlp.nm.trainables import TokenClassifier
from nemo.collections.nlp.utils.data_utils import get_vocab

# Parsing arguments
parser = argparse.ArgumentParser(description='NER with pretrained BERT')
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument(
    "--pretrained_model_name",
    default="bert-base-uncased",
    type=str,
    help="Name of the pre-trained model",
    choices=nemo_nlp.nm.trainables.get_pretrained_lm_models_list(),
)
parser.add_argument("--bert_config", default=None, type=str, help="Path to bert config file in json format")
parser.add_argument(
    "--tokenizer",
    default="nemobert",
    type=str,
    choices=["nemobert", "sentencepiece"],
    help="tokenizer to use, only relevant when using custom pretrained checkpoint.",
)
parser.add_argument(
    "--tokenizer_model",
    default=None,
    type=str,
    help="Path to pretrained tokenizer model, only used if --tokenizer is sentencepiece",
)
parser.add_argument(
    "--vocab_file", default=None, type=str, help="Path to the vocab file. Required for pretrained Megatron models"
)
parser.add_argument(
    "--do_lower_case",
    action='store_true',
    help="Whether to lower case the input text. True for uncased models, False for cased models. "
    + "Only applicable when tokenizer is build with vocab file",
)
parser.add_argument("--none_label", default='O', type=str)
parser.add_argument(
    "--queries",
    action='append',
    default=[
        'we bought four shirts from the nvidia gear ' + 'store in santa clara',
        'Nvidia is a company',
        'The Adventures of Tom Sawyer by Mark Twain '
        + 'is an 1876 novel about a young boy growing '
        + 'up along the Mississippi River',
    ],
    help="Example: --queries 'San Francisco' --queries 'LA'",
)
parser.add_argument(
    "--add_brackets",
    action='store_false',
    help="Whether to take predicted label in brackets or \
                    just append to word in the output",
)
parser.add_argument("--checkpoint_dir", default='output/checkpoints', type=str)
parser.add_argument("--labels_dict", default='label_ids.csv', type=str)

args = parser.parse_args()
logging.info(args)

if not os.path.exists(args.checkpoint_dir):
    raise ValueError(f'Checkpoint directory not found at {args.checkpoint_dir}')
if not os.path.exists(args.labels_dict):
    raise ValueError(f'Dictionary with ids to labels not found at {args.labels_dict}')

nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch, log_dir=None)

labels_dict = get_vocab(args.labels_dict)

model = nemo_nlp.nm.trainables.get_pretrained_lm_model(
    pretrained_model_name=args.pretrained_model_name, config=args.bert_config, vocab=args.vocab_file
)

tokenizer = nemo.collections.nlp.data.tokenizers.get_tokenizer(
    tokenizer_name=args.tokenizer,
    pretrained_model_name=args.pretrained_model_name,
    tokenizer_model=args.tokenizer_model,
    vocab_file=args.vocab_file,
    do_lower_case=args.do_lower_case,
)
hidden_size = model.hidden_size


data_layer = nemo_nlp.nm.data_layers.BertTokenClassificationInferDataLayer(
    queries=args.queries, tokenizer=tokenizer, max_seq_length=args.max_seq_length, batch_size=1
)

classifier = TokenClassifier(hidden_size=hidden_size, num_classes=len(labels_dict))

input_ids, input_type_ids, input_mask, _, subtokens_mask = data_layer()

hidden_states = model(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
logits = classifier(hidden_states=hidden_states)

###########################################################################

# Instantiate an optimizer to perform `infer` action
evaluated_tensors = nf.infer(tensors=[logits, subtokens_mask], checkpoint_dir=args.checkpoint_dir)


def concatenate(lists):
    return np.concatenate([t.cpu() for t in lists])


def add_brackets(text, add=args.add_brackets):
    return '[' + text + ']' if add else text


logits, subtokens_mask = [concatenate(tensors) for tensors in evaluated_tensors]

preds = np.argmax(logits, axis=2)

for i, query in enumerate(args.queries):
    logging.info(f'Query: {query}')

    pred = preds[i][subtokens_mask[i] > 0.5]
    words = query.strip().split()
    if len(pred) != len(words):
        raise ValueError('Pred and words must be of the same length')

    output = ''
    for j, w in enumerate(words):
        output += w
        label = labels_dict[pred[j]]
        if label != args.none_label:
            label = add_brackets(label)
            output += label
        output += ' '
    logging.info(f'Combined: {output.strip()}')
