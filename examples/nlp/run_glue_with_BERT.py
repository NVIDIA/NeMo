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
"""

import argparse
import os


import nemo
from nemo.backends.pytorch.common import CrossEntropyLoss, MSELoss
from nemo.utils.lr_policies import get_lr_policy

import nemo_nlp
from nemo_nlp import NemoBertTokenizer, SentencePieceTokenizer
from nemo_nlp.utils.callbacks.glue import \
    eval_iter_callback, eval_epochs_done_callback

from nemo_nlp.data.datasets.utils import processors, output_modes

parser = argparse.ArgumentParser(description="GLUE_with_pretrained_BERT")

# Parsing arguments
parser.add_argument("--data_dir", default='COLA', type=str, required=True,
                    help="The input data dir. Should contain the .tsv    \
                    files (or other data files) for the task.")
parser.add_argument("--task_name", default="CoLA", type=str, required=True,
                    help="Supported tasks: CoLA, SST-2, MRPC, STS-B, QQP, \
                    MNLI (matched and mismatched), QNLI, RTE, WNLI")
parser.add_argument("--pretrained_bert_model", default="bert-base-cased",
                    type=str, help="Name of the pre-trained model")
parser.add_argument("--bert_checkpoint", default=None, type=str,
                    help="Path to model checkpoint")
parser.add_argument("--bert_config", default=None, type=str,
                    help="Path to bert config file")
parser.add_argument("--tokenizer_model", default="tokenizer.model", type=str,
                    help="Path to pretrained tokenizer model")
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after   \
                    tokenization.Sequences longer than this will be       \
                    truncated, sequences shorter will be padded.")
parser.add_argument("--optimizer_kind", default="adam", type=str,
                    help="Optimizer kind")
parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
parser.add_argument("--lr", default=5e-5, type=float,
                    help="The initial learning rate.")
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--num_epochs", default=2, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for training/evaluation.")
parser.add_argument("--num_gpus", default=2, type=int,
                    help="Number of GPUs")
parser.add_argument("--amp_opt_level", default="O0", type=str,
                    choices=["O0", "O1", "O2"],
                    help="01/02 to enable mixed precision")
parser.add_argument("--local_rank", type=int, default=None,
                    help="For distributed training: local_rank")
parser.add_argument("--work_dir", default='output_glue', type=str,
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

args.work_dir = f'{args.work_dir}/{args.task_name.upper()}'
args.task_name = args.task_name.lower()

"""
Prepare GLUE task
MNLI task has two separate dev sets: matched and mismatched
"""
if args.task_name == 'mnli':
    eval_task_names = ("mnli", "mnli-mm")
    task_processors = (processors["mnli"](), processors["mnli-mm"]())
else:
    eval_task_names = (args.task_name,)
    task_processors = (processors[args.task_name](),)

label_list = task_processors[0].get_labels()
num_labels = len(label_list)
output_mode = output_modes[args.task_name]

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
        pretrained_model_name=args.pretrained_bert_model, factory=nf)
else:
    """ Use this if you're using a BERT model that you pre-trained yourself.
    Replace BERT-STEP-150000.pt with the path to your checkpoint.
    """
    tokenizer = SentencePieceTokenizer(model_path=args.tokenizer_model)
    tokenizer.add_special_tokens(["[MASK]", "[CLS]", "[SEP]"])

    model = nemo_nlp.huggingface.BERT(
                                  config_filename=args.bert_config, factory=nf)
    model.restore_from(args.bert_checkpoint)

hidden_size = model.local_parameters["hidden_size"]
data_layer_params = {'bos_token': None,
                     'eos_token': '[SEP]',
                     'pad_token': '[PAD]',
                     'cls_token': '[CLS]'}

nf.logger.info("Loading training data...")
train_dataset = nemo_nlp.GLUEDataset(
                                    tokenizer=tokenizer,
                                    data_dir=args.data_dir,
                                    max_seq_length=args.max_seq_length,
                                    processor=task_processors[0],
                                    output_mode=output_mode,
                                    evaluate=False,
                                    **data_layer_params)

nf.logger.info("Loading eval data...")
eval_datasets = []

# 2 task_processors for MNLI task: matched and mismatched dev set
for task_processor in task_processors:
    eval_datasets.append(nemo_nlp.GLUEDataset(
                                            tokenizer=tokenizer,
                                            data_dir=args.data_dir,
                                            max_seq_length=args.max_seq_length,
                                            processor=task_processor,
                                            output_mode=output_mode,
                                            evaluate=True,
                                            **data_layer_params))

# uses [CLS] token for classification (the first token)
classifier = nemo_nlp.SequenceClassifier(hidden_size=hidden_size,
                                         num_classes=num_labels,
                                         log_softmax=False)
if args.task_name == 'sts-b':
    glue_loss = MSELoss()
else:
    glue_loss = CrossEntropyLoss()


def create_pipeline(dataset, batch_size=args.batch_size,
                    local_rank=args.local_rank, num_gpus=args.num_gpus):

    data_layer = nemo_nlp.BertSentenceClassificationDataLayer(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
            factory=nf)

    input_ids, input_type_ids, input_mask, labels = data_layer()

    hidden_states = model(input_ids=input_ids,
                          token_type_ids=input_type_ids,
                          attention_mask=input_mask)
    logits = classifier(hidden_states=hidden_states)

    loss = glue_loss(logits=logits, labels=labels)

    steps_per_epoch = len(data_layer) // (batch_size * num_gpus)
    return loss, steps_per_epoch, data_layer, [logits, labels]


train_loss, steps_per_epoch, _, _ = create_pipeline(train_dataset)
_, _, eval_data_layer, eval_tensors = \
                               create_pipeline(eval_datasets[0])

callbacks_eval = [nemo.core.EvaluatorCallback(
    eval_tensors=eval_tensors,
    user_iter_callback=lambda x, y: eval_iter_callback(
        x, y, eval_data_layer, output_mode),
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(
        x, args.work_dir, eval_task_names[0]),
    tb_writer=nf.tb_writer,
    eval_step=steps_per_epoch)]

# create additional callback and data layer for MNLI mismatched dev set
if args.task_name == 'mnli':
    _, _, eval_data_layer_mm, eval_tensors_mm = \
                               create_pipeline(eval_datasets[1])
    callbacks_eval.append(nemo.core.EvaluatorCallback(
        eval_tensors=eval_tensors_mm,
        user_iter_callback=lambda x, y: eval_iter_callback(
            x, y, eval_data_layer_mm, output_mode),
        user_epochs_done_callback=lambda x: eval_epochs_done_callback(
            x, args.work_dir, eval_task_names[1]),
        tb_writer=nf.tb_writer,
        eval_step=steps_per_epoch))

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
         callbacks=[callback_train, ckpt_callback] + callbacks_eval,
         lr_policy=lr_policy_fn,
         optimizer=args.optimizer_kind,
         optimization_params={"num_epochs": args.num_epochs,
                              "lr": args.lr})
