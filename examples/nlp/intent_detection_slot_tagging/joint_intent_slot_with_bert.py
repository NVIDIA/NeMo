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
import os

import numpy as np
from transformers import BertTokenizer

import nemo
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM, LossAggregatorNM
from nemo.collections.nlp.callbacks.joint_intent_slot_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.collections.nlp.data.datasets.joint_intent_slot_dataset.data_descriptor import JointIntentSlotDataDesc
from nemo.collections.nlp.nm.data_layers import BertJointIntentSlotDataLayer
from nemo.collections.nlp.nm.trainables import JointIntentSlotClassifier
from nemo.collections.nlp.nm.trainables.common.huggingface import BERT
from nemo.core import CheckpointCallback, SimpleLossLoggerCallback
from nemo.utils.lr_policies import get_lr_policy

# Parsing arguments
parser = argparse.ArgumentParser(description='Joint intent detection and slot filling with pre-trained BERT')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--max_seq_length", default=50, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--num_train_samples", default=-1, type=int)
parser.add_argument("--num_eval_samples", default=-1, type=int)
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--fc_dropout", default=0.1, type=float)
parser.add_argument("--ignore_start_end", action='store_false')
parser.add_argument("--ignore_extra_tokens", action='store_false')
parser.add_argument("--pretrained_bert_model", default="bert-base-uncased", type=str)
parser.add_argument("--bert_checkpoint", default="", type=str)
parser.add_argument("--bert_config", default="", type=str)
parser.add_argument("--data_dir", default='data/nlu/atis', type=str)
parser.add_argument("--dataset_name", default='atis', type=str)
parser.add_argument("--train_file_prefix", default='train', type=str)
parser.add_argument("--eval_file_prefix", default='test', type=str)
parser.add_argument("--none_slot_label", default='O', type=str)
parser.add_argument("--pad_label", default=-1, type=int)
parser.add_argument("--work_dir", default='outputs', type=str)
parser.add_argument("--save_epoch_freq", default=1, type=int)
parser.add_argument("--save_step_freq", default=-1, type=int)
parser.add_argument("--optimizer_kind", default="adam", type=str)
parser.add_argument("--amp_opt_level", default="O0", type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--do_lower_case", action='store_true')
parser.add_argument("--shuffle_data", action='store_true')
parser.add_argument("--intent_loss_weight", default=0.6, type=float)
parser.add_argument("--class_balancing", default="regular", type=str, choices=["regular", "weighted_loss"])

args = parser.parse_args()

if not os.path.exists(args.data_dir):
    raise ValueError(f'Data not found at {args.data_dir}')

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

tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)

""" Load the pretrained BERT parameters
See the list of pretrained models, call:
nemo_nlp.huggingface.BERT.list_pretrained_models()
"""
if args.bert_checkpoint and args.bert_config:
    pretrained_bert_model = BERT(config_filename=args.bert_config)
    pretrained_bert_model.restore_from(args.bert_checkpoint)
else:
    pretrained_bert_model = BERT(pretrained_model_name=args.pretrained_bert_model)

hidden_size = pretrained_bert_model.hidden_size

data_desc = JointIntentSlotDataDesc(
    args.data_dir, args.do_lower_case, args.dataset_name, args.none_slot_label, args.pad_label
)

# Create sentence classification loss on top
classifier = JointIntentSlotClassifier(
    hidden_size=hidden_size, num_intents=data_desc.num_intents, num_slots=data_desc.num_slots, dropout=args.fc_dropout
)

if args.class_balancing == 'weighted_loss':
    # To tackle imbalanced classes, you may use weighted loss
    intent_loss_fn = CrossEntropyLossNM(logits_dim=2, weight=data_desc.intent_weights)
    slot_loss_fn = CrossEntropyLossNM(logits_dim=3, weight=data_desc.intent_weights)

else:
    intent_loss_fn = CrossEntropyLossNM(logits_dim=2)
    slot_loss_fn = CrossEntropyLossNM(logits_dim=3)

total_loss_fn = LossAggregatorNM(num_inputs=2, weights=[args.intent_loss_weight, 1.0 - args.intent_loss_weight])


def create_pipeline(num_samples=-1, batch_size=32, num_gpus=1, mode='train'):
    logging.info(f"Loading {mode} data...")
    data_file = f'{data_desc.data_dir}/{mode}.tsv'
    slot_file = f'{data_desc.data_dir}/{mode}_slots.tsv'
    shuffle = args.shuffle_data if mode == 'train' else False

    data_layer = BertJointIntentSlotDataLayer(
        input_file=data_file,
        slot_file=slot_file,
        pad_label=data_desc.pad_label,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        num_samples=num_samples,
        shuffle=shuffle,
        batch_size=batch_size,
        ignore_extra_tokens=args.ignore_extra_tokens,
        ignore_start_end=args.ignore_start_end,
    )

    input_data = data_layer()
    data_size = len(data_layer)

    logging.info(f'The length of data layer is {data_size}')

    if data_size < batch_size:
        logging.warning("Batch_size is larger than the dataset size")
        logging.warning("Reducing batch_size to dataset size")
        batch_size = data_size

    steps_per_epoch = math.ceil(data_size / (batch_size * num_gpus))
    logging.info(f"Steps_per_epoch = {steps_per_epoch}")

    hidden_states = pretrained_bert_model(
        input_ids=input_data.input_ids, token_type_ids=input_data.input_type_ids, attention_mask=input_data.input_mask
    )

    intent_logits, slot_logits = classifier(hidden_states=hidden_states)

    intent_loss = intent_loss_fn(logits=intent_logits, labels=input_data.intents)
    slot_loss = slot_loss_fn(logits=slot_logits, labels=input_data.slots, loss_mask=input_data.loss_mask)
    total_loss = total_loss_fn(loss_1=intent_loss, loss_2=slot_loss)

    if mode == 'train':
        tensors_to_evaluate = [total_loss, intent_logits, slot_logits]
    else:
        tensors_to_evaluate = [
            intent_logits,
            slot_logits,
            input_data.intents,
            input_data.slots,
            input_data.subtokens_mask,
        ]

    return tensors_to_evaluate, total_loss, steps_per_epoch, data_layer


train_tensors, train_loss, steps_per_epoch, _ = create_pipeline(
    args.num_train_samples, batch_size=args.batch_size, num_gpus=args.num_gpus, mode=args.train_file_prefix,
)
eval_tensors, _, _, data_layer = create_pipeline(
    args.num_eval_samples, batch_size=args.batch_size, num_gpus=args.num_gpus, mode=args.eval_file_prefix,
)

# Create callbacks for train and eval modes
train_callback = SimpleLossLoggerCallback(
    tensors=train_tensors,
    print_func=lambda x: str(np.round(x[0].item(), 3)),
    tb_writer=nf.tb_writer,
    get_tb_values=lambda x: [["loss", x[0]]],
    step_freq=steps_per_epoch,
)

eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=eval_tensors,
    user_iter_callback=lambda x, y: eval_iter_callback(x, y, data_layer),
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(x, f'{nf.work_dir}/graphs'),
    tb_writer=nf.tb_writer,
    eval_step=steps_per_epoch,
)

# Create callback to save checkpoints
ckpt_callback = CheckpointCallback(
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
