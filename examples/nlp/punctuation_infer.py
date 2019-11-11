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
parser.add_argument("--infer_file", default="dev.txt", type=str,
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
parser.add_argument("--work_dir", default='output_punct', type=str,
                    help="The output directory where the model predictions \
                    and checkpoints will be written.")

args = parser.parse_args()

args.interactive = True
args.bert_checkpoint = '/home/ebakhturina/output/punct/dataset_33_dr0.2_lr0.0001/checkpoints/BERT-EPOCH-9.pt'
args.classifier_checkpoint = '/home/ebakhturina/output/punct/dataset_33_dr0.2_lr0.0001/checkpoints/TokenClassifier-EPOCH-9.pt'

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
        text = '[CLS] ' + text
        
        ids = tokenizer.text_to_ids(text)
        tokens = tokenizer.ids_to_tokens(ids)
        print (tokens)
        input_ids = torch.Tensor(ids).long()\
                                     .to(bert_model._device)\
                                     .unsqueeze(0)
        input_type_ids = torch.zeros_like(input_ids)
        input_mask = torch.ones_like(input_ids)
        hidden_states = bert_model.forward(input_ids=input_ids,
                                           token_type_ids=input_type_ids,
                                           attention_mask=input_mask)
        
        '''
        ids = tokenizer.text_to_ids(text)
        tokens = tokenizer.ids_to_tokens(ids)
        input_ids_orig = torch.Tensor(ids).long()\
                                     .to(bert_model._device)\
                                     .unsqueeze(0) 

        input_ids = torch.zeros(1, args.max_seq_length)
        input_ids = input_ids.long().to(bert_model._device)
        input_ids[:, :input_ids_orig.shape[1]] = input_ids_orig
        
        input_type_ids = torch.zeros_like(input_ids)
        input_mask = torch.zeros_like(input_ids)
        input_mask[:,:input_ids_orig.shape[1]] = 1
        
        hidden_states = bert_model.forward(input_ids=input_ids,
                                           token_type_ids=input_type_ids,
                                           attention_mask=input_mask)
        import pdb; pdb.set_trace()
        ''' 
        
        logits = classifier.forward(hidden_states=hidden_states)
        logits = logits.squeeze(0).detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        preds = [ids_to_tags[i] for i in preds]
        print (preds)

        # drop [CLS] token and its prediction
        preds = preds[1:]
        tokens = tokens[1:]
        
        output = ''
        combined_tokens = ''
        prev_token = ''
        first_subtoken_punct_mark = ''
        use_prev_token = True
        delimiter = '##'
        
        for t in range(len(tokens)):
            if delimiter not in tokens[t]:
                if len(combined_tokens) > 0:
                    if first_subtoken_punct_mark != 'O':
                        combined_tokens += first_subtoken_punct_mark
                    output += combined_tokens + ' '
                    combined_tokens = ''
                    first_subtoken_punct_mark = ''
                output += tokens[t] + ' '
                use_prev_token = True
            else:
                if use_prev_token:
                    combined_tokens = combined_tokens.strip() + prev_token
                    output = output[:output.find(prev_token)]
                    first_subtoken_punct_mark = punct_mark
                combined_tokens += tokens[t][len(delimiter):]
                use_prev_token = False
            prev_token = tokens[t]
            
            punct_mark = preds[t]
            if use_prev_token and punct_mark != 'O':
                output = output.strip() + punct_mark
       
        if len(combined_tokens) > 0:
            if first_subtoken_punct_mark != 'O':
                combined_tokens += first_subtoken_punct_mark
            output += combined_tokens
        
        output = output.replace(" ' ", "'").strip().capitalize()

        return output

    print()
    print("========== Interactive translation mode ==========")
    while True:
        print("Type text to add punctuation, type STOP to exit.", "\n")
        input_text = input()
        if input_text == "STOP":
            print("============ Exiting translation mode ============")
            break
        print(get_punctuation(input_text), "\n")
