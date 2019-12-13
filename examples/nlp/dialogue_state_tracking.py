""" An implentation of the paper "Transferable Multi-Domain State Generator
for Task-Oriented Dialogue Systems" (Wu et al., 2019)


"""
import argparse
import os

import numpy as np

import nemo
from nemo.backends.pytorch.common import EncoderRNN
from nemo.utils.lr_policies import get_lr_policy
import nemo_nlp
from nemo_nlp.utils.callbacks.trade_dst import \
    eval_iter_callback, eval_epochs_done_callback


parser = argparse.ArgumentParser(
    description='TRADE for MultiWOZ 2.1 dialog state tracking')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--eval_batch_size", default=24, type=int)
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
parser.add_argument("--data_dir", default='data/dialog/multiwoz', type=str)
parser.add_argument("--dataset_name", default='multiwoz', type=str)
parser.add_argument("--train_file_prefix", default='train', type=str)
parser.add_argument("--eval_file_prefix", default='test', type=str)
parser.add_argument("--none_slot_label", default='O', type=str)
parser.add_argument("--pad_label", default=-1, type=int)
parser.add_argument("--work_dir", default='outputs', type=str)
parser.add_argument("--save_epoch_freq", default=1, type=int)
parser.add_argument("--save_step_freq", default=-1, type=int)
parser.add_argument("--optimizer_kind", default="adam", type=str)
parser.add_argument("--amp_opt_level", default="O0",
                    type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--do_lower_case", action='store_false')
parser.add_argument("--shuffle_data", action='store_false')


# parser.add_argument('--vocab_size', default=1, type=int)

# # Testing Setting
# parser.add_argument('-rundev', '--run_dev_testing', help='',
#                     default=0, type=int)
# parser.add_argument('-viz', '--vizualization',
#                     help='vizualization', type=int, default=0)
# parser.add_argument('-gs', '--genSample', help='Generate Sample',
#                     type=int, default=0)
# parser.add_argument('-evalp', '--evalp',
#                     help='evaluation period', default=1)
# parser.add_argument('-an', '--addName',
#                     help='An add name for the save folder', default='')
# parser.add_argument('-eb', '--eval_batch', help='Evaluation Batch_size',
#                     type=int, default=0)

args = parser.parse_args()
DOMAINS = {"attraction": 0, "restaurant": 1, "taxi": 2, "train": 3, "hotel": 4}

if not os.path.exists(args.data_dir):
    raise ValueError(f'Data not found at {args.data_dir}')

work_dir = f'{args.work_dir}/{args.dataset_name.upper()}'

nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                   local_rank=args.local_rank,
                                   optimization_level=args.amp_opt_level,
                                   log_dir=work_dir,
                                   create_tb_writer=True,
                                   files_to_copy=[__file__],
                                   add_time_to_log_dir=True)

data_layer = nemo_nlp.WOZDSTDataLayer(args.data_dir,
                                      DOMAINS,
                                      batch_size=args.batch_size,
                                      mode='train')
src_ids, src_lens, tgt_ids, tgt_lens, gate_labels, turn_domain = data_layer()
vocab_size = len(data_layer._dataset.vocab)
steps_per_epoch = len(data_layer) // args.batch_size

encoder = EncoderRNN(vocab_size,
                     args.emb_dim,
                     args.hid_dim,
                     args.dropout,
                     args.n_layers)

outputs, hidden = encoder(inputs=src_ids, input_lens=src_lens)

decoder = nemo_nlp.DSTGenerator(data_layer._dataset.vocab,
                                encoder.embedding,
                                args.hid_dim,
                                args.dropout,
                                data_layer._dataset.slots,
                                len(data_layer._dataset.gating_dict),
                                teacher_forcing=0.5)

point_outputs, gate_outputs = decoder(encoder_hidden=hidden,
                                      encoder_outputs=outputs,
                                      input_lens=src_lens,
                                      src_ids=src_ids,
                                      targets=tgt_ids)

eval_data_layer = nemo_nlp.WOZDSTDataLayer(args.data_dir,
                                           DOMAINS,
                                           batch_size=args.batch_size,
                                           mode='val')
eval_data_layer()
# gate_loss_fn = nemo.backends.pytorch.common.CrossEntropyLoss()
ptr_loss_fn = nemo_nlp.DSTMaskedCrossEntropy()

# gate_loss = gate_loss_fn()
train_loss = ptr_loss_fn(logits=point_outputs,
                         targets=tgt_ids,
                         mask=tgt_lens)

eval_data_layer = nemo_nlp.WOZDSTDataLayer(args.data_dir,
                                           DOMAINS,
                                           batch_size=args.batch_size,
                                           mode='val')
(eval_src_ids, eval_src_lens, eval_tgt_ids,
 eval_tgt_lens, eval_gate_labels, eval_turn_domain) = eval_data_layer()
outputs, hidden = encoder(inputs=eval_src_ids, input_lens=eval_src_lens)
eval_point_outputs, eval_gate_outputs = decoder(encoder_hidden=hidden,
                                                encoder_outputs=outputs,
                                                input_lens=eval_src_lens,
                                                src_ids=eval_src_ids,
                                                targets=eval_tgt_ids)
eval_loss = ptr_loss_fn(logits=eval_point_outputs,
                        targets=eval_tgt_ids,
                        mask=eval_tgt_lens)
eval_tensors = [eval_loss, eval_point_outputs, eval_gate_outputs,
                eval_gate_labels, eval_turn_domain]

# Create callbacks for train and eval modes
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    print_func=lambda x: print('Loss:', str(np.round(x[0].item(), 3))),
    tb_writer=nf.tb_writer,
    get_tb_values=lambda x: [["loss", x[0]]],
    step_freq=steps_per_epoch)


eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=eval_tensors,
    user_iter_callback=lambda x, y: eval_iter_callback(
        x, y, data_layer),
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(
        x, f'{nf.work_dir}/graphs'),
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

nf.train(tensors_to_optimize=[train_loss],
         callbacks=[train_callback, eval_callback, ckpt_callback],
         lr_policy=lr_policy_fn,
         optimizer=args.optimizer_kind,
         optimization_params={"num_epochs": args.num_epochs,
                              "lr": args.lr,
                              "weight_decay": args.weight_decay})
