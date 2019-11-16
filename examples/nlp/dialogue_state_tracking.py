""" An implentation of the paper "Transferable Multi-Domain State Generator
for Task-Oriented Dialogue Systems" (Wu et al., 2019)


"""
import argparse
import os

import nemo
import nemo_nlp

parser = argparse.ArgumentParser(
    description='Multi-domain dialogue state tracking')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--eval_batch_size", default=24, type=int)
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
parser.add_argument("--pretrained_bert_model",
                    default="bert-base-uncased", type=str)
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
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

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

dataset = nemo_nlp.data.datasets.WOZDSTDataset(args.data_dir,
                                               EXPERIMENT_DOMAINS,
                                               training=True)
