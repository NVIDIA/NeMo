""" An implementation of the paper "Transferable Multi-Domain State Generator
for Task-Oriented Dialogue Systems" (Wu et al., 2019)
Adopted from: https://github.com/jasonwu0731/trade-dst
"""

import argparse
import os
import math

import numpy as np
from tqdm import tqdm

from nemo.core import DeviceType
import nemo
from nemo.backends.pytorch.common import EncoderRNN
from nemo.utils.lr_policies import get_lr_policy
import nemo_nlp
from nemo_nlp.utils.callbacks.state_tracking_trade import \
    eval_iter_callback, eval_epochs_done_callback

parser = argparse.ArgumentParser(
    description='TRADE for MultiWOZ 2.1 dialog state tracking')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--eval_batch_size", default=16, type=int)
parser.add_argument("--max_seq_length", default=50, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--emb_dim", default=400, type=int)
parser.add_argument("--hid_dim", default=400, type=int)
parser.add_argument("--n_layers", default=1, type=int)
parser.add_argument("--dropout", default=0.2, type=float)
parser.add_argument("--input_dropout", default=0.2, type=float)
parser.add_argument("--data_dir", default='data/statetracking/multiwoz', type=str)
parser.add_argument("--dataset_name", default='multiwoz', type=str)
parser.add_argument("--train_file_prefix", default='train', type=str)
parser.add_argument("--eval_file_prefix", default='test', type=str)
parser.add_argument("--work_dir", default='outputs', type=str)
parser.add_argument("--save_epoch_freq", default=-1, type=int)
parser.add_argument("--save_step_freq", default=-1, type=int)
parser.add_argument("--optimizer_kind", default="adam", type=str)
parser.add_argument("--amp_opt_level", default="O0",
                    type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--do_lower_case", action='store_true')
parser.add_argument("--shuffle_data", action='store_true')
parser.add_argument("--num_train_samples", default=-1, type=int)
parser.add_argument("--num_eval_samples", default=-1, type=int)
parser.add_argument("--grad_norm_clip", type=float, default=-1,
                    help="gradient clipping")
parser.add_argument("--progress_bar", action='store_true')
parser.add_argument("--teacher_forcing", default=0.0, type=float)
args = parser.parse_args()

DOMAINS = {"attraction": 0, "restaurant": 1, "taxi": 2, "train": 3, "hotel": 4}

if not os.path.exists(args.data_dir):
    raise ValueError(f'Data not found at {args.data_dir}')

work_dir = f'{args.work_dir}/{args.dataset_name.upper()}'

# TODO
# import torch
# torch.backends.cudnn.deterministic = True
#
# torch.manual_seed(999)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(999)
# import random
# random.seed(30)


nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                   local_rank=args.local_rank,
                                   optimization_level=args.amp_opt_level,
                                   log_dir=work_dir,
                                   create_tb_writer=True,
                                   files_to_copy=[__file__],
                                   add_time_to_log_dir=True,
                                   #placement=CPU
                                   )

total_cpus = os.cpu_count()
num_workers = 0  # max(int(total_cpus / nf.world_size), 1)

data_layer_train = nemo_nlp.WOZDSTDataLayer(args.data_dir,
                                      DOMAINS,
                                      num_samples=args.num_train_samples,
                                      shuffle=args.shuffle_data,
                                      num_workers=num_workers,
                                      local_rank=args.local_rank,
                                      batch_size=args.batch_size,
                                      mode='train',
                                      is_training=True,
                                      input_dropout=args.input_dropout)
src_ids_train, src_lens_train, tgt_ids_train, \
    tgt_lens_train, gate_labels_train, turn_domain_train =  data_layer_train()
vocab_size = len(data_layer_train._dataset.vocab)

train_data_size = len(data_layer_train)
print(f'The length of train data layer is {train_data_size}')

batch_size = args.batch_size
if train_data_size < batch_size:
    nf.logger.warning("Batch_size is larger than the train dataset size")
    nf.logger.warning("Reducing batch_size to dataset size")
    batch_size = train_data_size

steps_per_epoch = math.ceil(train_data_size / (batch_size * args.num_gpus))
nf.logger.info(f"Steps_per_epoch Train= {steps_per_epoch}")


encoder = EncoderRNN(vocab_size,
                     args.emb_dim,
                     args.hid_dim,
                     args.dropout,
                     args.n_layers)

outputs_train, hidden_train = encoder(inputs=src_ids_train,
                                      input_lens=src_lens_train)

decoder = nemo_nlp.DSTGenerator(data_layer_train._dataset.vocab,
                                encoder.embedding,
                                args.hid_dim,
                                args.dropout,
                                data_layer_train._dataset.slots,
                                len(data_layer_train._dataset.gating_dict),
                                # TODO
                                teacher_forcing=args.teacher_forcing)

point_outputs_train, gate_outputs_train = decoder(encoder_hidden=hidden_train,
                                                  encoder_outputs=outputs_train,
                                                  input_lens=src_lens_train,
                                                  src_ids=src_ids_train,
                                                  targets=tgt_ids_train)

gate_loss_fn = \
    nemo_nlp.CrossEntropyLoss3D(num_classes=len(data_layer_train.gating_dict))
ptr_loss_fn = nemo_nlp.DSTMaskedCrossEntropy()

total_loss = nemo_nlp.LossAggregatorNM(num_inputs=2)

gate_loss_train = gate_loss_fn(logits=gate_outputs_train,
                                 labels=gate_labels_train)
ptr_loss_train = ptr_loss_fn(logits=point_outputs_train,
                                targets=tgt_ids_train,
                                mask=tgt_lens_train)

loss_train = total_loss(loss_1=gate_loss_train, loss_2=ptr_loss_train)

data_layer_eval = nemo_nlp.WOZDSTDataLayer(args.data_dir,
                                           DOMAINS,
                                           num_samples=args.num_eval_samples,
                                           shuffle=False,
                                           num_workers=num_workers,
                                           local_rank=args.local_rank,
                                           batch_size=args.batch_size,
                                           mode=args.eval_file_prefix,
                                           is_training=False,
                                           input_dropout=args.input_dropout)

(src_ids_eval, src_lens_eval, tgt_ids_eval,
 tgt_lens_eval, gate_labels_eval, turn_domain_eval) = data_layer_eval()
outputs, hidden = encoder(inputs=src_ids_eval, input_lens=src_lens_eval)
point_outputs_eval, gate_outputs_eval = decoder(encoder_hidden=hidden,
                                                encoder_outputs=outputs,
                                                input_lens=src_lens_eval,
                                                src_ids=src_ids_eval,
                                                targets=tgt_ids_eval)

gate_loss_eval = gate_loss_fn(logits=gate_outputs_eval,
                                labels=gate_labels_eval)

ptr_loss_eval = ptr_loss_fn(logits=point_outputs_eval,
                            targets=tgt_ids_eval,
                            mask=tgt_lens_eval)

loss_eval = total_loss(loss_1=gate_loss_eval, loss_2=ptr_loss_eval)

eval_tensors = [loss_eval, point_outputs_eval, gate_outputs_eval,
                gate_labels_eval, turn_domain_eval, tgt_ids_eval, tgt_lens_eval]

# Create progress bars
if args.progress_bar:
    iter_num_eval = math.ceil(len(data_layer_eval) /
                              args.batch_size / nf.world_size)
    progress_bar_eval = tqdm(total=iter_num_eval, position=0, leave=False)

    iter_num_train = math.ceil(len(data_layer_train) /
                               args.batch_size / nf.world_size)
    progress_bar_train = tqdm(total=iter_num_train, position=0, leave=False)
else:
    progress_bar_eval = None
    progress_bar_train = None

# Create callbacks for train and eval modes
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[loss_train, gate_loss_train, ptr_loss_train],
    print_func=lambda x: print(f'Loss:{str(np.round(x[0].item(), 3))}, '
                               f'Gate Loss:{str(np.round(x[1].item(), 3))}, '
                               f'Pointer Loss:{str(np.round(x[2].item(), 3))}'),
    tb_writer=nf.tb_writer,
    get_tb_values=lambda x: [["loss", x[0]],
                             ["gate_loss", x[1]],
                             ["pointer_loss", x[2]]],
    step_freq=steps_per_epoch,
    progress_bar=progress_bar_train)

eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=eval_tensors,
    user_iter_callback=lambda x, y: eval_iter_callback(
        x, y, data_layer_eval, progress_bar_eval),
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(
        x, data_layer_eval, progress_bar_eval),
    tb_writer=nf.tb_writer,
    eval_step=steps_per_epoch)

# Create callback to save checkpoints
ckpt_callback = nemo.core.CheckpointCallback(
    folder=nf.checkpoint_dir,
    epoch_freq=args.save_epoch_freq,
    step_freq=args.save_step_freq)

lr_policy_fn = get_lr_policy(args.lr_policy,
                             total_steps=args.num_epochs * steps_per_epoch,
                             warmup_ratio=args.lr_warmup_proportion)

grad_norm_clip = args.grad_norm_clip if args.grad_norm_clip > 0 else None
# TODO
nf.train(tensors_to_optimize=[loss_train],
         callbacks=[eval_callback, train_callback, ckpt_callback],
         #callbacks=[train_callback, ckpt_callback],
         #lr_policy=lr_policy_fn,
         optimizer=args.optimizer_kind,
         optimization_params={"num_epochs": args.num_epochs,
                              "lr": args.lr,
                              "grad_norm_clip": grad_norm_clip,
                              "weight_decay": args.weight_decay
                              })
