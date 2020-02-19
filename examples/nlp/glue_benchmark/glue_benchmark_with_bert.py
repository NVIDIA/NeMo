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

Example of running a pretrained BERT model on the 9 GLUE tasks, read more
about GLUE benchmark here: https://gluebenchmark.com

Download the GLUE data by running the script:
https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e

To run this example on 1 GPU:
python glue_benchmark_with_bert.py  \
--data_dir /path_to_data_dir/MRPC \
--task_name mrpc \
--work_dir /path_to_output_folder \

To run this example on 4 GPUs with mixed precision:
python -m torch.distributed.launch \
--nproc_per_node=4 glue_benchmark_with_bert.py \
--data_dir=/path_to_data/MNLI \
--task_name mnli \
--work_dir /path_to_output_folder \
--num_gpus=4 \
--amp_opt_level=O1 \

The generated predictions and associated labels will be stored in the
word_dir in {task_name}.txt along with the checkpoints and tensorboard files.

Some of these tasks have a small dataset and training can lead to high variance
in the results between different runs. Below is the median on 5 runs
(with different seeds) for each of the metrics on the dev set of the benchmark
with an uncased BERT base model (the checkpoint bert-base-uncased)
(source https://github.com/huggingface/transformers/tree/master/examples#glue).

Task	Metric	                        Result
CoLA	Matthew's corr	                48.87
SST-2	Accuracy	                    91.74
MRPC	F1/Accuracy	                 90.70/86.27
STS-B	Person/Spearman corr.	     91.39/91.04
QQP	    Accuracy/F1	                 90.79/87.66
MNLI	Matched acc./Mismatched acc. 83.70/84.83
QNLI	Accuracy	                    89.31
RTE	    Accuracy	                    71.43
WNLI	Accuracy	                    43.66

"""

import argparse
import os

from transformers import BertConfig

import nemo.collections.nlp as nemo_nlp
import nemo.core as nemo_core
from nemo import logging
from nemo.backends.pytorch.common import CrossEntropyLossNM, MSELoss
from nemo.collections.nlp.callbacks.glue_benchmark_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.collections.nlp.data import NemoBertTokenizer, SentencePieceTokenizer
from nemo.collections.nlp.data.datasets.glue_benchmark_dataset.glue_benchmark_dataset import output_modes, processors
from nemo.collections.nlp.nm.data_layers import GlueClassificationDataLayer, GlueRegressionDataLayer
from nemo.collections.nlp.nm.trainables import SequenceClassifier, SequenceRegression
from nemo.utils.lr_policies import get_lr_policy

parser = argparse.ArgumentParser(description="GLUE_with_pretrained_BERT")

# Parsing arguments
parser.add_argument(
    "--data_dir",
    default='COLA',
    type=str,
    required=True,
    help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
)
parser.add_argument(
    "--task_name",
    default="CoLA",
    type=str,
    required=True,
    choices=['cola', 'sst-2', 'mrpc', 'sts-b', 'qqp', 'mnli', 'qnli', 'rte', 'wnli'],
    help="GLUE task name, MNLI includes both matched and mismatched tasks",
)
parser.add_argument(
    "--pretrained_bert_model", default="bert-base-cased", type=str, help="Name of the pre-trained model"
)
parser.add_argument("--bert_checkpoint", default=None, type=str, help="Path to model checkpoint")
parser.add_argument("--bert_config", default=None, type=str, help="Path to bert config file in json format")
parser.add_argument(
    "--tokenizer_model",
    default="tokenizer.model",
    type=str,
    help="Path to pretrained tokenizer model, only used if --tokenizer is sentencepiece",
)
parser.add_argument(
    "--tokenizer",
    default="nemobert",
    type=str,
    choices=["nemobert", "sentencepiece"],
    help="tokenizer to use, only relevant when using custom pretrained checkpoint.",
)
parser.add_argument(
    "--max_seq_length",
    default=128,
    type=int,
    choices=range(1, 513),
    help="The maximum total input sequence length after tokenization.Sequences longer than this will be \
                    truncated, sequences shorter will be padded.",
)
parser.add_argument("--optimizer_kind", default="adam", type=str, help="Optimizer kind")
parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate.")
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
parser.add_argument("--num_epochs", default=3, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training/evaluation.")
parser.add_argument("--num_gpus", default=1, type=int, help="Number of GPUs")
parser.add_argument(
    "--amp_opt_level", default="O0", type=str, choices=["O0", "O1", "O2"], help="01/02 to enable mixed precision"
)
parser.add_argument("--local_rank", type=int, default=None, help="For distributed training: local_rank")
parser.add_argument(
    "--work_dir",
    default='output_glue',
    type=str,
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument(
    "--save_epoch_freq",
    default=1,
    type=int,
    help="Frequency of saving checkpoint '-1' - epoch checkpoint won't be saved",
)
parser.add_argument(
    "--save_step_freq",
    default=-1,
    type=int,
    help="Frequency of saving checkpoint '-1' - step checkpoint won't be saved",
)
parser.add_argument("--loss_step_freq", default=25, type=int, help="Frequency of printing loss")

args = parser.parse_args()

if not os.path.exists(args.data_dir):
    raise FileNotFoundError(
        "GLUE datasets not found. Datasets can be "
        "obtained at https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e"
    )

args.work_dir = f'{args.work_dir}/{args.task_name.upper()}'

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
nf = nemo_core.NeuralModuleFactory(
    backend=nemo_core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=args.work_dir,
    create_tb_writer=True,
    files_to_copy=[__file__],
    add_time_to_log_dir=True,
)


if args.bert_checkpoint is None:
    """ Use this if you're using a standard BERT model.
    To see the list of pretrained models, call:
    nemo_nlp.nm.trainables.huggingface.BERT.list_pretrained_models()
    """
    tokenizer = NemoBertTokenizer(args.pretrained_bert_model)
    model = nemo_nlp.nm.trainables.huggingface.BERT(pretrained_model_name=args.pretrained_bert_model)
else:
    """ Use this if you're using a BERT model that you pre-trained yourself.
    Replace BERT-STEP-150000.pt with the path to your checkpoint.
    """
    if args.tokenizer == "sentencepiece":
        special_tokens = nemo_nlp.utils.MODEL_SPECIAL_TOKENS['bert']
        tokenizer = SentencePieceTokenizer(model_path=args.tokenizer_model, special_tokens=special_tokens)
    elif args.tokenizer == "nemobert":
        tokenizer = NemoBertTokenizer(args.pretrained_bert_model)
    else:
        raise ValueError(f"received unexpected tokenizer '{args.tokenizer}'")

    if args.bert_config is not None:
        config = BertConfig.from_json_file(args.bert_config).to_dict()
        args.vocab_size = config['vocab_size']
        args.hidden_size = config['hidden_size']
        args.num_hidden_layers = config['num_hidden_layers']
        args.num_attention_heads = config['num_attention_heads']
        args.intermediate_size = config['intermediate_size']
        args.hidden_act = config['hidden_act']
        args.max_seq_length = config['max_position_embeddings']

        model = nemo_nlp.nm.trainables.huggingface.BERT(
            vocab_size=args.vocab_size,
            num_hidden_layers=args.num_hidden_layers,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            max_position_embeddings=args.max_seq_length,
            hidden_act=args.hidden_act,
        )
        logging.info(f"using {args.bert_config}")
    else:
        model = nemo_nlp.nm.trainables.huggingface.BERT(pretrained_model_name=args.pretrained_bert_model)
    model.restore_from(args.bert_checkpoint)
    logging.info(f"model restored from {args.bert_checkpoint}")

hidden_size = model.hidden_size

# uses [CLS] token for classification (the first token)
if args.task_name == 'sts-b':
    pooler = SequenceRegression(hidden_size=hidden_size)
    glue_loss = MSELoss()
else:
    pooler = SequenceClassifier(hidden_size=hidden_size, num_classes=num_labels, log_softmax=False)
    glue_loss = CrossEntropyLossNM()


def create_pipeline(
    max_seq_length=args.max_seq_length,
    batch_size=args.batch_size,
    local_rank=args.local_rank,
    num_gpus=args.num_gpus,
    evaluate=False,
    processor=task_processors[0],
):
    data_layer = GlueClassificationDataLayer
    if output_mode == 'regression':
        data_layer = GlueRegressionDataLayer

    data_layer = data_layer(
        processor=processor,
        evaluate=evaluate,
        batch_size=batch_size,
        # num_workers=0,
        # local_rank=local_rank,
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        max_seq_length=max_seq_length,
        token_params=token_params,
    )

    input_ids, input_type_ids, input_mask, labels = data_layer()

    hidden_states = model(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

    """
    For STS-B (regressiont tast), the pooler_output represents a is single
    number prediction for each sequence.
    The rest of GLUE tasts are classification tasks; the pooler_output
    represents logits.
    """
    pooler_output = pooler(hidden_states=hidden_states)
    if args.task_name == 'sts-b':
        loss = glue_loss(preds=pooler_output, labels=labels)
    else:
        loss = glue_loss(logits=pooler_output, labels=labels)

    steps_per_epoch = len(data_layer) // (batch_size * num_gpus)
    return loss, steps_per_epoch, data_layer, [pooler_output, labels]


token_params = {'bos_token': None, 'eos_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]'}

train_loss, steps_per_epoch, _, _ = create_pipeline()
_, _, eval_data_layer, eval_tensors = create_pipeline(evaluate=True)

callbacks_eval = [
    nemo_core.EvaluatorCallback(
        eval_tensors=eval_tensors,
        user_iter_callback=lambda x, y: eval_iter_callback(x, y),
        user_epochs_done_callback=lambda x: eval_epochs_done_callback(x, args.work_dir, eval_task_names[0]),
        tb_writer=nf.tb_writer,
        eval_step=steps_per_epoch,
    )
]

"""
MNLI task has two dev sets: matched and mismatched
Create additional callback and data layer for MNLI mismatched dev set
"""
if args.task_name == 'mnli':
    _, _, eval_data_layer_mm, eval_tensors_mm = create_pipeline(evaluate=True, processor=task_processors[1])
    callbacks_eval.append(
        nemo_core.EvaluatorCallback(
            eval_tensors=eval_tensors_mm,
            user_iter_callback=lambda x, y: eval_iter_callback(x, y),
            user_epochs_done_callback=lambda x: eval_epochs_done_callback(x, args.work_dir, eval_task_names[1]),
            tb_writer=nf.tb_writer,
            eval_step=steps_per_epoch,
        )
    )

logging.info(f"steps_per_epoch = {steps_per_epoch}")
callback_train = nemo_core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    print_func=lambda x: print("Loss: {:.3f}".format(x[0].item())),
    get_tb_values=lambda x: [["loss", x[0]]],
    step_freq=args.loss_step_freq,
    tb_writer=nf.tb_writer,
)

ckpt_callback = nemo_core.CheckpointCallback(
    folder=nf.checkpoint_dir, epoch_freq=args.save_epoch_freq, step_freq=args.save_step_freq
)

lr_policy_fn = get_lr_policy(
    args.lr_policy, total_steps=args.num_epochs * steps_per_epoch, warmup_ratio=args.lr_warmup_proportion
)

nf.train(
    tensors_to_optimize=[train_loss],
    callbacks=[callback_train, ckpt_callback] + callbacks_eval,
    lr_policy=lr_policy_fn,
    optimizer=args.optimizer_kind,
    optimization_params={"num_epochs": args.num_epochs, "lr": args.lr},
)
