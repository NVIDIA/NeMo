# pylint: disable=invalid-name
import argparse
import math
import numpy as np
import os
import torch

import nemo
from nemo.utils.lr_policies import SquareAnnealing, CosineAnnealing, \
    WarmupAnnealing

import nemo_nlp
from nemo_nlp import NemoBertTokenizer, SentencePieceTokenizer
from nemo_nlp.utils.callbacks.punctuation import \
    eval_iter_callback, eval_epochs_done_callback

# Parsing arguments
parser = argparse.ArgumentParser(description="Punctuation_with_pretrainedBERT")
parser.add_argument("--interactive", action='store_true'),
parser.add_argument("--pretrained_bert_model", default="bert-base-uncased",
                    type=str)
parser.add_argument("--infer_file", default="", type=str,
                    help="File to use for inference")
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--num_classes", default=4, type=int)
parser.add_argument("--dataset_type", default="BertPunctuationDataset",
                    type=str)
parser.add_argument("--bert_checkpoint", default='BERT-EPOCH-1.pt', type=str)
parser.add_argument("--classifier_checkpoint",
                    default='TokenClassifier-EPOCH-1.pt', type=str)
parser.add_argument("--bert_config", default=None, type=str)
parser.add_argument("--work_dir", default='output_glue', type=str,
                    help="The output directory where the model predictions \
                    and checkpoints will be written.")

args = parser.parse_args()

# Instantiate Neural Factory with supported backend
nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                   log_dir=args.work_dir)

output_file = f'{nf.work_dir}/output.txt'

tokenizer = NemoBertTokenizer(args.pretrained_bert_model)
bert_model = nemo_nlp.huggingface.BERT(
            pretrained_model_name=args.pretrained_bert_model)

tag_ids = {'O': 0, ',': 3, '.': 2, '?': 1}
ids_to_tags = {tag_ids[k]: k for k in tag_ids}

num_labels = len(tag_ids)
hidden_size = bert_model.local_parameters["hidden_size"]
classifier = nemo_nlp.TokenClassifier(hidden_size=hidden_size,
                                      num_classes=args.num_classes,
                                      dropout=0)

if args.bert_checkpoint:
    bert_model.restore_from(args.bert_checkpoint)
    classifier.restore_from(args.classifier_checkpoint)

bert_model.eval()
classifier.eval()

if not args.interactive and os.path.isfile(args.infer_file):
    eval_data_layer = nemo_nlp.BertTokenClassificationDataLayer(
        tokenizer=tokenizer,
        input_file=os.path.join(args.infer_file),
        max_seq_length=args.max_seq_length,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_workers=0)

    input_ids, input_type_ids, input_mask, labels, seq_ids = eval_data_layer()
    hidden_states = bert_model(input_ids=input_ids,
                               token_type_ids=input_type_ids,
                               attention_mask=input_mask)

    logits = classifier(hidden_states=hidden_states)

    callback_eval = nemo.core.EvaluatorCallback(
        eval_tensors=[logits, seq_ids],
        user_iter_callback=lambda x, y: eval_iter_callback(
            x, y, eval_data_layer, tag_ids),
        user_epochs_done_callback=lambda x: eval_epochs_done_callback(
            x, tag_ids, output_file))

    nf.eval(callbacks=[callback_eval])


if args.interactive:
    # set all modules into evaluation mode (turn off dropout)
    bert_model.eval()
    classifier.eval()

    def get_punctuation(text):
        ids = tokenizer.text_to_ids(text)
        tokens = tokenizer.ids_to_tokens(ids)
        input_ids = torch.Tensor(ids).long()\
                                     .to(bert_model._device)\
                                     .unsqueeze(0)
        input_type_ids = torch.zeros_like(input_ids)
        input_mask = torch.ones_like(input_ids)
        hidden_states = bert_model.forward(input_ids=input_ids,
                                           token_type_ids=input_type_ids,
                                           attention_mask=input_mask)

        logits = classifier.forward(hidden_states=hidden_states)
        logits = logits.squeeze(0).detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        preds = [ids_to_tags[i] for i in preds]

        output = ''
        tokens_no_punct = []
        punct_tokens = []
        punct_mark = None
        for i in range(len(preds)):
            if preds[i] == 'O':
                if punct_mark:
                    output += tokenizer.tokens_to_text(tokens_no_punct) + ' '
                    output += tokenizer.tokens_to_text(punct_tokens)
                    output += punct_mark + ' '
                    tokens_no_punct = []
                    punct_tokens = []
                    punct_mark = None
                tokens_no_punct.append(tokens[i])
            else:
                punct_tokens.append(tokens[i])
                if punct_mark is None:
                    punct_mark = preds[i]

        if len(tokens_no_punct) > 0:
            output += tokenizer.tokens_to_text(tokens_no_punct)
        if punct_mark:
            output += ' ' + tokenizer.tokens_to_text(punct_tokens)
            output += punct_mark + ' '
        return (output)

    print()
    print("========== Interactive translation mode ==========")
    while True:
        print("Type text to add punctuation, type STOP to exit.", "\n")
        input_text = input()
        if input_text == "STOP":
            print("============ Exiting translation mode ============")
            break
        print(get_punctuation(input_text), "\n")
