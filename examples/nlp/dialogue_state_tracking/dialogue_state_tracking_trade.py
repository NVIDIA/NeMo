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

""" An implementation of the paper "Transferable Multi-Domain State Generator
for Task-Oriented Dialogue Systems" (Wu et al., 2019 - ACL 2019)
Adopted from: https://github.com/jasonwu0731/trade-dst
"""

import argparse
import math
import os

import numpy as np

import nemo.backends.pytorch as nemo_backend
import nemo.backends.pytorch.common.losses
import nemo.collections.nlp as nemo_nlp
import nemo.core as nemo_core
from nemo import logging
from nemo.backends.pytorch.common import EncoderRNN
from nemo.collections.nlp.callbacks.state_tracking_trade_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.collections.nlp.data.datasets.multiwoz_dataset import MultiWOZDataDesc
from nemo.utils.lr_policies import get_lr_policy

parser = argparse.ArgumentParser(description='Dialog state tracking with TRADE model on MultiWOZ dataset')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--eval_batch_size", default=16, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--lr_warmup_proportion", default=0.0, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--lr_policy", default=None, type=str)
parser.add_argument("--min_lr", default=1e-4, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--emb_dim", default=400, type=int)
parser.add_argument("--hid_dim", default=400, type=int)
parser.add_argument("--n_layers", default=1, type=int)
parser.add_argument("--dropout", default=0.2, type=float)
parser.add_argument("--input_dropout", default=0.2, type=float)
parser.add_argument("--data_dir", default='data/statetracking/multiwoz2.1', type=str)
parser.add_argument("--train_file_prefix", default='train', type=str)
parser.add_argument("--eval_file_prefix", default='test', type=str)
parser.add_argument("--work_dir", default='outputs', type=str)
parser.add_argument("--save_epoch_freq", default=-1, type=int)
parser.add_argument("--save_step_freq", default=-1, type=int)
parser.add_argument("--optimizer_kind", default="adam", type=str)
parser.add_argument("--amp_opt_level", default="O0", type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--shuffle_data", action='store_true')
parser.add_argument("--num_train_samples", default=-1, type=int)
parser.add_argument("--num_eval_samples", default=-1, type=int)
parser.add_argument("--grad_norm_clip", type=float, default=10, help="gradient clipping")
parser.add_argument("--teacher_forcing", default=0.5, type=float)
args = parser.parse_args()

# List of the domains to be considered
domains = {"attraction": 0, "restaurant": 1, "taxi": 2, "train": 3, "hotel": 4}

if not os.path.exists(args.data_dir):
    raise ValueError(f'Data not found at {args.data_dir}')

work_dir = f'{args.work_dir}/DST_TRADE'

data_desc = MultiWOZDataDesc(args.data_dir, domains)

nf = nemo_core.NeuralModuleFactory(
    backend=nemo_core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=work_dir,
    create_tb_writer=True,
    files_to_copy=[__file__],
    add_time_to_log_dir=True,
)

vocab_size = len(data_desc.vocab)
encoder = EncoderRNN(vocab_size, args.emb_dim, args.hid_dim, args.dropout, args.n_layers)

decoder = nemo_nlp.nm.trainables.TRADEGenerator(
    data_desc.vocab,
    encoder.embedding,
    args.hid_dim,
    args.dropout,
    data_desc.slots,
    len(data_desc.gating_dict),
    teacher_forcing=args.teacher_forcing,
)

gate_loss_fn = nemo_backend.losses.CrossEntropyLossNM(logits_dim=3)
ptr_loss_fn = nemo_nlp.nm.losses.MaskedLogLoss()
total_loss_fn = nemo.backends.pytorch.common.losses.LossAggregatorNM(num_inputs=2)


def create_pipeline(num_samples, batch_size, num_gpus, input_dropout, data_prefix, is_training):
    logging.info(f"Loading {data_prefix} data...")
    shuffle = args.shuffle_data if is_training else False

    data_layer = nemo_nlp.nm.data_layers.MultiWOZDataLayer(
        args.data_dir,
        data_desc.domains,
        all_domains=data_desc.all_domains,
        vocab=data_desc.vocab,
        slots=data_desc.slots,
        gating_dict=data_desc.gating_dict,
        num_samples=num_samples,
        shuffle=shuffle,
        num_workers=0,
        batch_size=batch_size,
        mode=data_prefix,
        is_training=is_training,
        input_dropout=input_dropout,
    )

    src_ids, src_lens, tgt_ids, tgt_lens, gate_labels, turn_domain = data_layer()

    data_size = len(data_layer)
    logging.info(f'The length of data layer is {data_size}')

    if data_size < batch_size:
        logging.warning("Batch_size is larger than the dataset size")
        logging.warning("Reducing batch_size to dataset size")
        batch_size = data_size

    steps_per_epoch = math.ceil(data_size / (batch_size * num_gpus))
    logging.info(f"Steps_per_epoch = {steps_per_epoch}")

    outputs, hidden = encoder(inputs=src_ids, input_lens=src_lens)

    point_outputs, gate_outputs = decoder(
        encoder_hidden=hidden, encoder_outputs=outputs, input_lens=src_lens, src_ids=src_ids, targets=tgt_ids
    )

    gate_loss = gate_loss_fn(logits=gate_outputs, labels=gate_labels)
    ptr_loss = ptr_loss_fn(logits=point_outputs, labels=tgt_ids, length_mask=tgt_lens)
    total_loss = total_loss_fn(loss_1=gate_loss, loss_2=ptr_loss)

    if is_training:
        tensors_to_evaluate = [total_loss, gate_loss, ptr_loss]
    else:
        tensors_to_evaluate = [total_loss, point_outputs, gate_outputs, gate_labels, turn_domain, tgt_ids, tgt_lens]

    return tensors_to_evaluate, total_loss, ptr_loss, gate_loss, steps_per_epoch, data_layer


(
    tensors_train,
    total_loss_train,
    ptr_loss_train,
    gate_loss_train,
    steps_per_epoch_train,
    data_layer_train,
) = create_pipeline(
    args.num_train_samples,
    batch_size=args.batch_size,
    num_gpus=args.num_gpus,
    input_dropout=args.input_dropout,
    data_prefix=args.train_file_prefix,
    is_training=True,
)

tensors_eval, total_loss_eval, ptr_loss_eval, gate_loss_eval, steps_per_epoch_eval, data_layer_eval = create_pipeline(
    args.num_eval_samples,
    batch_size=args.eval_batch_size,
    num_gpus=args.num_gpus,
    input_dropout=0.0,
    data_prefix=args.eval_file_prefix,
    is_training=False,
)

# Create callbacks for train and eval modes
train_callback = nemo_core.SimpleLossLoggerCallback(
    tensors=[total_loss_train, gate_loss_train, ptr_loss_train],
    print_func=lambda x: logging.info(
        f'Loss:{str(np.round(x[0].item(), 3))}, '
        f'Gate Loss:{str(np.round(x[1].item(), 3))}, '
        f'Pointer Loss:{str(np.round(x[2].item(), 3))}'
    ),
    tb_writer=nf.tb_writer,
    get_tb_values=lambda x: [["loss", x[0]], ["gate_loss", x[1]], ["pointer_loss", x[2]]],
    step_freq=steps_per_epoch_train,
)

eval_callback = nemo_core.EvaluatorCallback(
    eval_tensors=tensors_eval,
    user_iter_callback=lambda x, y: eval_iter_callback(x, y, data_desc),
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(x, data_desc),
    tb_writer=nf.tb_writer,
    eval_step=steps_per_epoch_train,
)

ckpt_callback = nemo_core.CheckpointCallback(
    folder=nf.checkpoint_dir, epoch_freq=args.save_epoch_freq, step_freq=args.save_step_freq
)

if args.lr_policy is not None:
    total_steps = args.num_epochs * steps_per_epoch_train
    lr_policy_fn = get_lr_policy(
        args.lr_policy, total_steps=total_steps, warmup_ratio=args.lr_warmup_proportion, min_lr=args.min_lr
    )
else:
    lr_policy_fn = None

grad_norm_clip = args.grad_norm_clip if args.grad_norm_clip > 0 else None
nf.train(
    tensors_to_optimize=[total_loss_train],
    callbacks=[eval_callback, train_callback, ckpt_callback],
    lr_policy=lr_policy_fn,
    optimizer=args.optimizer_kind,
    optimization_params={
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "grad_norm_clip": grad_norm_clip,
        "weight_decay": args.weight_decay,
    },
)
