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
import math

import numpy as np

import nemo
import nemo.collections.nlp as nemo_nlp
from nemo import logging
from nemo.collections.nlp.callbacks.text_classification_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.collections.nlp.data.datasets import TextClassificationDataDesc
from nemo.utils.lr_policies import get_lr_policy

# Parsing arguments
parser = argparse.ArgumentParser(description='Sentence classification with pretrained BERT')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_seq_length", default=36, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--num_train_samples", default=-1, type=int)
parser.add_argument("--num_eval_samples", default=-1, type=int)
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--fc_dropout", default=0.1, type=float)
parser.add_argument("--pretrained_model_name", default="bert-base-uncased", type=str)
parser.add_argument("--bert_checkpoint", default=None, type=str)
parser.add_argument("--bert_config", default=None, type=str)
parser.add_argument("--data_dir", required=True, type=str)
parser.add_argument(
    "--dataset_name",
    required=True,
    type=str,
    choices=["sst-2", "imdb", "thucnews", "jarvis", "nlu-ubuntu", "nlu-web", "nlu-chat"],
)
parser.add_argument("--use_cache", action='store_true')
parser.add_argument("--train_file_prefix", default='train', type=str)
parser.add_argument("--eval_file_prefix", default='test', type=str)
parser.add_argument("--work_dir", default='outputs', type=str)
parser.add_argument("--save_epoch_freq", default=1, type=int)
parser.add_argument("--save_step_freq", default=-1, type=int)
parser.add_argument("--optimizer_kind", default="adam", type=str)
parser.add_argument("--amp_opt_level", default="O0", type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--do_lower_case", action='store_true')
parser.add_argument("--no_shuffle_data", action='store_false', dest="shuffle_data")
parser.add_argument("--class_balancing", default="None", type=str, choices=["None", "weighted_loss"])

args = parser.parse_args()

work_dir = f'{args.work_dir}/{args.dataset_name.upper()}'
nf = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=work_dir,
    create_tb_writer=True,
    files_to_copy=[__file__],
    add_time_to_log_dir=True,
)

model_type = args.pretrained_model_name.split('-')[0]
model_cls = nemo_nlp.utils.DEFAULT_MODELS[model_type]['class']

if args.bert_config is not None:
    model = model_cls(config_filename=args.bert_config)
else:
    """ Use this if you're using a standard BERT model.
    To see the list of pretrained models, call:
    nemo_nlp.nm.trainables.huggingface.BERT.list_pretrained_models()
    nemo_nlp.nm.trainables.huggingface.Albert.list_pretrained_models()
    nemo_nlp.nm.trainables.huggingface.Roberta.list_pretrained_models()
    """
    model = model_cls(pretrained_model_name=args.pretrained_model_name)


if args.bert_checkpoint is not None:
    model.restore_from(args.bert_checkpoint)
    logging.info(f"model restored from {args.bert_checkpoint}")

hidden_size = model.hidden_size

tokenizer_cls = nemo_nlp.data.NemoBertTokenizer
tokenizer_special_tokens = nemo_nlp.utils.MODEL_SPECIAL_TOKENS[model_type]
tokenizer = tokenizer_cls(
    pretrained_model=args.pretrained_model_name, special_tokens=tokenizer_special_tokens, bert_derivate=model_type,
)

data_desc = TextClassificationDataDesc(args.dataset_name, args.data_dir, args.do_lower_case, args.eval_file_prefix)

# Create sentence classification loss on top
classifier = nemo_nlp.nm.trainables.SequenceClassifier(
    hidden_size=hidden_size, num_classes=data_desc.num_labels, dropout=args.fc_dropout
)

if args.class_balancing == 'weighted_loss':
    # You may need to increase the number of epochs for convergence.
    loss_fn = nemo.backends.pytorch.common.CrossEntropyLossNM(weight=data_desc.class_weights)
else:
    loss_fn = nemo.backends.pytorch.common.CrossEntropyLossNM()


def create_pipeline(num_samples=-1, batch_size=32, num_gpus=1, local_rank=0, mode='train'):
    logging.info(f"Loading {mode} data...")
    data_file = f'{data_desc.data_dir}/{mode}.tsv'
    shuffle = args.shuffle_data if mode == 'train' else False

    data_layer = nemo_nlp.nm.data_layers.BertTextClassificationDataLayer(
        input_file=data_file,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        num_samples=num_samples,
        shuffle=shuffle,
        batch_size=batch_size,
        use_cache=args.use_cache,
        model_name=args.pretrained_model_name,
    )

    ids, type_ids, input_mask, labels = data_layer()
    data_size = len(data_layer)

    if data_size < batch_size:
        logging.warning("Batch_size is larger than the dataset size")
        logging.warning("Reducing batch_size to dataset size")
        batch_size = data_size

    steps_per_epoch = math.ceil(data_size / (batch_size * num_gpus))
    logging.info(f"Steps_per_epoch = {steps_per_epoch}")

    hidden_states = model(input_ids=ids, token_type_ids=type_ids, attention_mask=input_mask)

    logits = classifier(hidden_states=hidden_states)
    loss = loss_fn(logits=logits, labels=labels)

    if mode == 'train':
        tensors_to_evaluate = [loss, logits]
    else:
        tensors_to_evaluate = [logits, labels]

    return tensors_to_evaluate, loss, steps_per_epoch, data_layer


train_tensors, train_loss, steps_per_epoch, _ = create_pipeline(
    num_samples=args.num_train_samples,
    batch_size=args.batch_size,
    num_gpus=args.num_gpus,
    local_rank=args.local_rank,
    mode=args.train_file_prefix,
)
eval_tensors, _, _, data_layer = create_pipeline(
    num_samples=args.num_eval_samples,
    batch_size=args.batch_size,
    num_gpus=args.num_gpus,
    local_rank=args.local_rank,
    mode=args.eval_file_prefix,
)

# Create callbacks for train and eval modes
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=train_tensors,
    print_func=lambda x: print("Loss: {:.3f}".format(x[0].item())),
    tb_writer=nf.tb_writer,
    get_tb_values=lambda x: [["loss", x[0]]],
)

eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=eval_tensors,
    user_iter_callback=lambda x, y: eval_iter_callback(x, y, data_layer),
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(x, f'{nf.work_dir}/graphs'),
    tb_writer=nf.tb_writer,
    eval_step=steps_per_epoch,
)

# Create callback to save checkpoints
ckpt_callback = nemo.core.CheckpointCallback(
    folder=nf.checkpoint_dir, epoch_freq=args.save_epoch_freq, step_freq=args.save_step_freq
)

lr_policy_fn = get_lr_policy(
    args.lr_policy, total_steps=args.num_epochs * steps_per_epoch, warmup_ratio=args.lr_warmup_proportion
)

nf.train(
    tensors_to_optimize=[train_loss],
    callbacks=[train_callback, eval_callback, ckpt_callback],
    lr_policy=lr_policy_fn,
    optimizer=args.optimizer_kind,
    optimization_params={"num_epochs": args.num_epochs, "lr": args.lr, "weight_decay": args.weight_decay},
)
