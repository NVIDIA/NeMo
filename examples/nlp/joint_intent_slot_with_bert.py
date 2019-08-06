import argparse
import logging
import math
import os
import sys
import time

import nemo
from nemo_nlp.callbacks.joint_intent_slot import \
    eval_iter_callback, eval_epochs_done_callback

from nemo_nlp import NemoBertTokenizer

import numpy as np
from tensorboardX import SummaryWriter


# from nemo_nlp.data.datasets.sentence_classification \
#   import sentence_classification_eval_iter_callback, \
#   sentence_classification_eval_epochs_done_callback

from nemo_nlp.text_data_utils import \
    process_imdb, process_nlu, process_nvidia_car, process_atis

# Parsing arguments
parser = argparse.ArgumentParser(description='Joint-intent BERT')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--max_seq_length", default=50, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--num_train_samples", default=-1, type=int)
parser.add_argument("--num_dev_samples", default=-1, type=int)
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--lr", default=5e-5, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--pretrained_bert_model",
                    default="bert-base-uncased",
                    type=str)
parser.add_argument("--data_dir", default='', type=str)
parser.add_argument("--dataset_name", default='atis', type=str)
parser.add_argument("--work_dir", default='outputs', type=str)
parser.add_argument("--optimizer_kind", default="adam", type=str)
parser.add_argument("--mixed_precision", action='store_true')
parser.add_argument("--do_lower_case", action='store_true')
parser.add_argument("--lr_policy", default="lr_warmup", type=str)
parser.add_argument("--intent_loss_weight", default=0.5, type=float)

args = parser.parse_args()


if not os.path.exists(args.data_dir):
    print(f'Training Data Not Found. Downloading to {args.data_dir}')

work_dir = f'{args.work_dir}/{args.dataset_name.upper()}'

work_dir = os.path.join(work_dir, time.strftime('%Y%m%d-%H%M%S'))
os.makedirs(work_dir, exist_ok=True)

logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'{work_dir}/log.txt')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

params = vars(args)
for key in params:
    logger.info(f'{key}\t{params[key]}')


tb_fold = os.path.join(work_dir, 'tensorboard')
os.makedirs(tb_fold, exist_ok=True)
try:
    import tensorflow as tf
    tb_writer = SummaryWriter(tb_fold)
except ImportError:
    tb_writer = None
    logger.info('Install TensorFlow to use TensorBoard')

""" TODO: write a utils function to figure out all the setups
"""

local_rank = args.local_rank
lr_policy = args.lr_policy
num_epochs = args.num_epochs

if local_rank is not None:
    device = nemo.core.DeviceType.AllGpu
else:
    device = nemo.core.DeviceType.GPU

optimization_level = nemo.core.Optimization.nothing

if args.mixed_precision:
    optimization_level = nemo.core.Optimization.mxprO1

# instantiate Neural Factory with supported backend
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=local_rank,
    optimization_level=optimization_level,
    placement=device)

# Load the pretrained BERT parameters
# pretrained_model can be one of:
# bert-base-uncased, bert-large-uncased, bert-base-cased,
# bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
pretrained_model_name = args.pretrained_bert_model

pretrained_bert_model = neural_factory.get_module(
    name="BERT",
    params={"pretrained_model_name": pretrained_model_name,
            "local_rank": local_rank},
    collection="nemo_nlp",
    pretrained=True)

if args.dataset_name == 'sst-2':
    num_labels = 2
    data_dir = args.data_dir
    devfile = data_dir + '/dev.tsv'
elif args.dataset_name == 'imdb':
    num_labels = 2
    data_dir = process_imdb(args.data_dir, args.do_lower_case)
    devfile = data_dir + '/test.tsv'
elif args.dataset_name.startswith('nlu-'):
    if args.dataset_name.endswith('chat'):
        args.data_dir = f'{args.data_dir}/ChatbotCorpus.json'
        num_labels = 2
    elif args.dataset_name.endswith('ubuntu'):
        args.data_dir = f'{args.data_dir}/AskUbuntuCorpus.json'
        num_labels = 5
    elif args.dataset_name.endswith('web'):
        args.data_dir = f'{args.data_dir}/WebApplicationsCorpus.json'
        num_labels = 8
    data_dir = process_nlu(args.data_dir,
                           args.do_lower_case,
                           dataset_name=args.dataset_name)
    devfile = data_dir + '/test.tsv'
elif args.dataset_name == 'nvidia-car':
    data_dir, labels = process_nvidia_car(args.data_dir, args.do_lower_case)
    for intent in labels:
        idx = labels[intent]
        logger.info(f'{intent}: {idx}')
    num_labels = len(labels)
    devfile = data_dir + '/test.tsv'
elif args.dataset_name == 'atis':
    num_intents = 26
    num_slots = 129
    data_dir = process_atis(f'{args.data_dir}', args.do_lower_case)
    devfile = data_dir + '/test.tsv'

# Create sentence classification loss on top
d_model = pretrained_bert_model.local_parameters["d_model"]
classification_dropout = \
    pretrained_bert_model.local_parameters["fully_connected_dropout"]

joint_intent_slot_loss = neural_factory.get_module(
    name="JointIntentSlotLoss",
    params={"d_model": d_model,
            "num_intents": num_intents,
            "num_slots": num_slots,
            "dropout": classification_dropout},
    collection="nemo_nlp"
)

# Data layer with SST training data
path_to_vocab_file = \
    pretrained_bert_model.local_parameters["path_to_vocab_file"]
vocab_positional_map = \
    pretrained_bert_model.local_parameters[
        "vocab_positional_embedding_size_map"]


# define tokenizer, in this example we use WordPiece BertTokenizer
# we also increase the vocabulary size to make it multiple of 8 to accelerate
# training in fp16 mode with the use of Tensor Cores
tokenizer = NemoBertTokenizer(
    vocab_file=path_to_vocab_file,
    do_lower_case=True,
    max_len=vocab_positional_map
)

# Training pipeline
print("Loading training data...")
train_data_layer = neural_factory.get_module(
    name="BertJointIntentSlotDataLayer",
    params={
        "path_to_data": data_dir + '/train.tsv',
        "path_to_slot": data_dir + '/train_slots.tsv',
        "pad_label": str(num_slots-1),
        "tokenizer": tokenizer,
        "mode": 'train',
        "max_seq_length": args.max_seq_length,
        "num_samples": args.num_train_samples,
        "path_to_vocab_file": path_to_vocab_file,
        "vocab_positional_embedding_size_map": vocab_positional_map,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 0,
        "local_rank": local_rank
    },
    collection="nemo_nlp"
)


input_ids, input_type_ids, input_mask, intents, slots = train_data_layer()

print('input_ids', input_ids)
print('input_type_ids', input_type_ids)
print('input_mask', input_mask)
print('intents', intents)
print('slots', slots)

train_data_size = len(train_data_layer)
steps_per_epoch = int(train_data_size / (args.batch_size*args.num_gpus))

logger.info(f"Steps_per_epoch = {steps_per_epoch}")

hidden_states = pretrained_bert_model(
    input_ids=input_ids, input_type_ids=input_type_ids, input_mask=input_mask)

train_loss, train_intent_logits, train_slot_logits = \
    joint_intent_slot_loss(hidden_states=hidden_states,
                           intents=intents,
                           slots=slots,
                           input_mask=input_mask,
                           intent_loss_weight=args.intent_loss_weight)

# Evaluation pipeline
logger.info("Loading eval data...")
eval_data_layer = neural_factory.get_module(
    name="BertJointIntentSlotDataLayer",
    params={
        "path_to_data": data_dir + '/test.tsv',
        "path_to_slot": data_dir + '/test_slots.tsv',
        "pad_label": str(num_slots-1),
        "tokenizer": tokenizer,
        "mode": "eval",
        "max_seq_length": args.max_seq_length,
        "path_to_vocab_file": path_to_vocab_file,
        "vocab_positional_embedding_size_map": vocab_positional_map,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_samples": args.num_dev_samples,
        "num_workers": 0,
        "local_rank": local_rank
    },
    collection="nemo_nlp"
)

input_ids, input_type_ids, eval_input_mask, eval_intents, eval_slots = \
                                                            eval_data_layer()

hidden_states = pretrained_bert_model(
    input_ids=input_ids, input_type_ids=input_type_ids,
    input_mask=eval_input_mask)

eval_loss, eval_intent_logits, eval_slot_logits = \
    joint_intent_slot_loss(hidden_states=hidden_states,
                           intents=eval_intents,
                           slots=eval_slots,
                           input_mask=eval_input_mask,
                           intent_loss_weight=args.intent_loss_weight)

###############################################################################


def get_loss(loss):
    str_ = str(np.round(loss, 3))
    return str_


# Create trainer and execute training action
callback_train = nemo.core.SimpleLossLoggerCallback(
    tensor_list2string=lambda x: get_loss(x[0].item()),
    tensorboard_writer=tb_writer,
    step_frequency=100)

# Instantiate an optimizer to perform `train` action
optimizer = neural_factory.get_trainer(
    params={
        "optimizer_kind": args.optimizer_kind,
        "optimization_params": {"num_epochs": args.num_epochs,
                                "lr": args.lr,
                                "weight_decay": args.weight_decay},
    }
)

callback_eval = nemo.core.EvaluatorCallback(
    eval_tensors=[eval_intent_logits, eval_slot_logits,
                  eval_intents, eval_slots],
    user_iter_callback=lambda x, y: eval_iter_callback(x, y, eval_data_layer),
    user_epochs_done_callback=lambda x:eval_epochs_done_callback(x, f'{work_dir}/graphs'),
    tensorboard_writer=tb_writer,
    eval_step=steps_per_epoch)


def lr_policy_cosine_decorator(num_steps):

    def lr_policy_cosine(initial_lr, step, e):
        progress = float(step/num_steps)
        out_lr = initial_lr * 0.5 * (1.+math.cos(math.pi * progress))
        return out_lr

    return lr_policy_cosine


def lr_policy_poly_decorator(num_steps):

    def lr_policy_poly(initial_lr, step, e):
        min_lr = 0.00001
        res = initial_lr * ((num_steps - step + 1) / num_steps) ** 2
        return max(res, min_lr)

    return lr_policy_poly


def lr_policy_warmup_decorator(num_steps, warmup_proportion):

    def lr_policy_warmup(initial_lr, step, e):
        progress = float(step/num_steps)
        if progress < warmup_proportion:
            out_lr = initial_lr*progress/warmup_proportion
        else:
            out_lr = initial_lr*max((progress-1.)/(warmup_proportion-1.), 0.)
        return out_lr

    return lr_policy_warmup


if lr_policy == "lr_warmup":
    lr_policy_func = lr_policy_warmup_decorator(num_epochs*steps_per_epoch,
                                                args.lr_warmup_proportion)
elif lr_policy == "lr_poly":
    lr_policy_func = lr_policy_poly_decorator(num_epochs*steps_per_epoch)
elif lr_policy == "lr_cosine":
    lr_policy_func = lr_policy_cosine_decorator(num_epochs*steps_per_epoch)
else:
    raise ValueError("Invalid lr_policy, must be lr_warmup or lr_poly")

optimizer.train(
    tensors_to_optimize=[train_loss],
    callbacks=[callback_train, callback_eval],
    lr_policy=lr_policy_func)
