"""
Copyright 2018 The Google AI Language Team Authors and
The HuggingFace Inc. team.
Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Some transformer of this code were adapted from the HuggingFace library at
https://github.com/huggingface/transformers

Download the Squad data by running the script:
examples/nlp/scripts/download_squad.py

To run this example on 1 GPU:
python squad.py  \
--data_dir /path_to_data_dir/squad/v1.1 \
--work_dir /path_to_output_folder
"""

import argparse
import os
import sys

import nemo
from nemo.utils.lr_policies import get_lr_policy
import json
import nemo_nlp
from nemo_nlp import BertSquadDataLayer
from nemo_nlp import NemoBertTokenizer, SentencePieceTokenizer
from nemo_nlp.utils.callbacks.glue import \
    eval_iter_callback, eval_epochs_done_callback
from nemo_nlp import QuestionAnsweringLoss

parser = argparse.ArgumentParser(description="Squad_with_pretrained_BERT")

# Parsing arguments
parser.add_argument("--data_dir", type=str, required=True,
                    help="The input data dir. Should contain train.json, dev.json    \
                    files (or other data files) for the task.")
parser.add_argument("--dataset_type", type=str,
                    help="The input data type.")
parser.add_argument("--pretrained_bert_model", default="bert-base-uncased",
                    type=str, help="Name of the pre-trained model")
parser.add_argument("--bert_checkpoint", default=None, type=str,
                    help="Path to model checkpoint")
parser.add_argument("--bert_config", default=None, type=str,
                    help="Path to bert config file in json format")
parser.add_argument("--tokenizer_model", default="tokenizer.model", type=str,
                    help="Path to pretrained tokenizer model, \
                    only used if --tokenizer is sentencepiece")
parser.add_argument("--tokenizer", default="nemobert", type=str,
                    choices=["nemobert", "sentencepiece"],
                    help="tokenizer to use, \
                    only relevant when using custom pretrained checkpoint.")
parser.add_argument("--optimizer_kind", default="adam", type=str,
                    help="Optimizer kind")
parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
parser.add_argument("--lr", default=5e-5, type=float,
                    help="The initial learning rate.")
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--num_epochs", default=3, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for training/evaluation.")
parser.add_argument("--doc_stride", default=128, type=int,
                    help="When splitting up a long document into chunks, how much stride to take between chunks.")
parser.add_argument("--max_query_length", default=64, type=int,
                    help="The maximum number of tokens for the question. Questions longer than this will "
                            "be truncated to this length.")
parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
parser.add_argument("--num_gpus", default=1, type=int,
                    help="Number of GPUs")
parser.add_argument("--amp_opt_level", default="O0", type=str,
                    choices=["O0", "O1", "O2"],
                    help="01/02 to enable mixed precision")
parser.add_argument("--local_rank", type=int, default=None,
                    help="For distributed training: local_rank")
parser.add_argument("--work_dir", default='output_squad', type=str,
                    help="The output directory where the model predictions \
                    and checkpoints will be written.")
parser.add_argument("--save_epoch_freq", default=1, type=int,
                    help="Frequency of saving checkpoint \
                    '-1' - epoch checkpoint won't be saved")
parser.add_argument("--save_step_freq", default=-1, type=int,
                    help="Frequency of saving checkpoint \
                    '-1' - step checkpoint won't be saved")
parser.add_argument("--loss_step_freq", default=25, type=int,
                    help="Frequency of printing loss")

args = parser.parse_args()

if not os.path.exists(args.data_dir):
    raise FileNotFoundError("GLUE datasets not found. Datasets can be "
                            "obtained at https://gist.github.com/W4ngatang/ \
                            60c2bdb54d156a41194446737ce03e2e")

args.work_dir = f'{args.work_dir}/squad1.1'

label_list = ""
num_labels = len(label_list)

# Instantiate neural factory with supported backend
nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                   local_rank=args.local_rank,
                                   optimization_level=args.amp_opt_level,
                                   log_dir=args.work_dir,
                                   create_tb_writer=True,
                                   files_to_copy=[__file__],
                                   add_time_to_log_dir=True)

if args.bert_checkpoint is None:
    """ Use this if you're using a standard BERT model.
    To see the list of pretrained models, call:
    nemo_nlp.huggingface.BERT.list_pretrained_models()
    """
    tokenizer = NemoBertTokenizer(args.pretrained_bert_model)
    model = nemo_nlp.huggingface.BERT(
        pretrained_model_name=args.pretrained_bert_model)
else:
    """ Use this if you're using a BERT model that you pre-trained yourself.
    Replace BERT-STEP-150000.pt with the path to your checkpoint.
    """
    if args.tokenizer == "sentencepiece":
        tokenizer = SentencePieceTokenizer(model_path=args.tokenizer_model)
        tokenizer.add_special_tokens(["[MASK]", "[CLS]", "[SEP]"])
    elif args.tokenizer == "nemobert":
        tokenizer = NemoBertTokenizer(args.pretrained_bert_model)
    else:
        raise ValueError(f"received unexpected tokenizer '{args.tokenizer}'")
    if args.bert_config is not None:
        with open(args.bert_config) as json_file:
            config = json.load(json_file)
        model = nemo_nlp.huggingface.BERT(**config)
    else:
        model = nemo_nlp.huggingface.BERT(
            pretrained_model_name=args.pretrained_bert_model)

    model.restore_from(args.bert_checkpoint)

hidden_size = model.local_parameters["hidden_size"]

# uses [CLS] token for classification (the first token)

qa_head = nemo_nlp.TokenClassifier(hidden_size=hidden_size,
                                        num_classes=2,
                                        num_layers=1,
                                        log_softmax=False)
squad_loss = QuestionAnsweringLoss()

# token_params = {'bos_token': None,
#                 'eos_token': '[SEP]',
#                 'pad_token': '[PAD]',
#                 'cls_token': '[CLS]'}
token_params={}
def create_pipeline(max_query_length=args.max_query_length, max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    batch_size=args.batch_size,
                    local_rank=args.local_rank,
                    num_gpus=args.num_gpus,
                    evaluate=False):

    data_layer = 'BertSquadDataLayer'
    data_layer = getattr(sys.modules[__name__], data_layer)

    data_layer = data_layer(
                    evaluate=evaluate,
                    task_name="SquadV1",
                    batch_size=batch_size,
                    num_workers=0,
                    local_rank=local_rank,
                    tokenizer=tokenizer,
                    data_dir=args.data_dir,
                    max_query_length=max_query_length,
                    max_seq_length=max_seq_length,
                    doc_stride=doc_stride,
                    token_params=token_params)

    input_ids, input_type_ids, input_mask, start_positions, end_positions = data_layer()

    hidden_states = model(input_ids=input_ids,
                          token_type_ids=input_type_ids,
                          attention_mask=input_mask)


    qa_output = qa_head(hidden_states=hidden_states)
    loss = squad_loss(logits=qa_output, start_positions=start_positions, end_positions=end_positions)

    steps_per_epoch = len(data_layer) // (batch_size * num_gpus)
    return loss, steps_per_epoch, data_layer


train_loss, steps_per_epoch, _= create_pipeline()
# _, _, eval_data_layer, eval_tensors = create_pipeline(evaluate=True)

# callbacks_eval = [nemo.core.EvaluatorCallback(
#     eval_tensors=eval_tensors,
#     user_iter_callback=lambda x, y: eval_iter_callback(x, y),
#     user_epochs_done_callback=lambda x:
#         eval_epochs_done_callback(x, args.work_dir, eval_task_names[0]),
#     tb_writer=nf.tb_writer,
#     eval_step=steps_per_epoch)]


nf.logger.info(f"steps_per_epoch = {steps_per_epoch}")
callback_train = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    print_func=lambda x: print("Loss: {:.3f}".format(x[0].item())),
    get_tb_values=lambda x: [["loss", x[0]]],
    step_freq=args.loss_step_freq,
    tb_writer=nf.tb_writer)

ckpt_callback = nemo.core.CheckpointCallback(
    folder=nf.checkpoint_dir,
    epoch_freq=args.save_epoch_freq,
    step_freq=args.save_step_freq)

lr_policy_fn = get_lr_policy(args.lr_policy,
                             total_steps=args.num_epochs * steps_per_epoch,
                             warmup_ratio=args.lr_warmup_proportion)

nf.train(tensors_to_optimize=[train_loss],
         callbacks=[callback_train, ckpt_callback],
         lr_policy=lr_policy_fn,
         optimizer=args.optimizer_kind,
         optimization_params={"num_epochs": args.num_epochs,
                              "lr": args.lr})
