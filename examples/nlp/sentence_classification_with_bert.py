import argparse
import math
import os

import numpy as np
from pytorch_transformers import BertTokenizer

import nemo
from nemo_nlp.callbacks.sentence_classification import \
    eval_iter_callback, eval_epochs_done_callback
from nemo_nlp.text_data_utils import \
    process_sst_2, process_imdb, process_nlu, process_nvidia_car
from nemo.utils import ExpManager

# Parsing arguments
parser = argparse.ArgumentParser(
    description='Sentiment analysis with pretrained BERT')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_seq_length", default=36, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--num_train_samples", default=1000, type=int)
parser.add_argument("--num_dev_samples", default=100, type=int)
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--lr_policy", default="lr_warmup", type=str)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--fc_dropout", default=0.1, type=float)
parser.add_argument("--pretrained_bert_model",
                    default="bert-base-uncased",
                    type=str)
parser.add_argument("--data_dir", default='data/sc/aclImdb', type=str)
parser.add_argument("--dataset_name", default='imdb', type=str)
parser.add_argument("--work_dir", default='outputs', type=str)
parser.add_argument("--save_epoch_freq", default=1, type=int)
parser.add_argument("--save_step_freq", default=-1, type=int)
parser.add_argument("--optimizer_kind", default="adam", type=str)
parser.add_argument("--mixed_precision", action='store_true')
parser.add_argument("--do_lower_case", action='store_false')

args = parser.parse_args()

work_dir = f'{args.work_dir}/{args.dataset_name.upper()}'
exp = ExpManager(work_dir, local_rank=args.local_rank)
exp.create_logger()
exp.log_exp_info(vars(args))

local_rank = args.local_rank
lr_policy = args.lr_policy
num_epochs = args.num_epochs

device = nemo.utils.get_device(local_rank)
optimization_level = nemo.utils.get_opt_level(args.mixed_precision)

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
pretrained_bert_model = neural_factory.get_module(
    name="huggingface.BERT",
    params={"pretrained_model_name": args.pretrained_bert_model,
            "local_rank": local_rank},
    collection="nemo_nlp",
    pretrained=True)

if args.dataset_name == 'sst-2':
    data_dir = process_sst_2(args.data_dir)
    num_labels = 2
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
else:
    exp.logger.info("Looks like you pass in the name of dataset that isn't "
                    "already supported by NeMo. Please make sure that you "
                    "build the preprocessing method for it.")

# Create sentence classification loss on top
d_model = pretrained_bert_model.local_parameters["d_model"]

sequence_classifier = neural_factory.get_module(
    name="SequenceClassifier",
    params={"d_model": d_model,
            "num_classes": num_labels,
            "dropout": args.fc_dropout},
    collection="nemo_nlp"
)

sequence_loss = nemo.backends.pytorch.common.CrossEntropyLoss(
    factory=neural_factory)

tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)

# Training pipeline
exp.logger.info("Loading training data...")

train_data_layer = neural_factory.get_module(
    name="BertSentenceClassificationDataLayer",
    params={
        "path_to_data": data_dir + '/train.tsv',
        "tokenizer": tokenizer,
        "mode": 'train',
        "max_seq_length": args.max_seq_length,
        "num_samples": args.num_train_samples,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 0,
        "local_rank": local_rank
    },
    collection="nemo_nlp"
)


input_ids, input_type_ids, input_mask, labels = train_data_layer()

train_data_size = len(train_data_layer)

if train_data_size < args.batch_size:
    exp.logger.warning("Batch_size is larger than the dataset size")
    exp.logger.warning("Reducing batch_size to dataset size")
    args.batch_size = train_data_size

steps_per_epoch = int(train_data_size / (args.batch_size * args.num_gpus))

exp.logger.info(f"Steps_per_epoch = {steps_per_epoch}")

hidden_states = pretrained_bert_model(input_ids=input_ids,
                                      token_type_ids=input_type_ids,
                                      attention_mask=input_mask)

train_logits = sequence_classifier(hidden_states=hidden_states)
train_loss = sequence_loss(logits=train_logits, labels=labels)

# Evaluation pipeline
exp.logger.info("Loading eval data...")
eval_data_layer = neural_factory.get_module(
    name="BertSentenceClassificationDataLayer",
    params={
        "path_to_data": devfile,
        "tokenizer": tokenizer,
        "mode": "eval",
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_samples": args.num_dev_samples,
        "num_workers": 0,
        "local_rank": local_rank
    },
    collection="nemo_nlp"
)

eval_input_ids, eval_input_type_ids, eval_input_mask, eval_labels =\
    eval_data_layer()

hidden_states = pretrained_bert_model(input_ids=eval_input_ids,
                                      token_type_ids=eval_input_type_ids,
                                      attention_mask=eval_input_mask)

eval_logits = sequence_classifier(hidden_states=hidden_states)
eval_loss = sequence_loss(logits=eval_logits, labels=eval_labels)

###############################################################################


def get_loss(loss):
    return str(np.round(loss, 3))


# Create callback to save checkpoints
ckpt_callback = nemo.core.CheckpointCallback(
    folder=exp.ckpt_dir,
    epoch_freq=args.save_epoch_freq,
    step_freq=args.save_step_freq)

# Create trainer and execute training action
callback_train = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss, train_logits],
    print_func=lambda x: get_loss(x[0].item()),
    tb_writer=exp.tb_writer,
    get_tb_values=lambda x: [["loss", x[0]]],
    step_freq=100)

# Instantiate an optimizer to perform `train` action
optimizer = neural_factory.get_trainer()

callback_eval = nemo.core.EvaluatorCallback(
    eval_tensors=[eval_logits, eval_labels],
    user_iter_callback=lambda x,
    y: eval_iter_callback(
        x,
        y,
        eval_data_layer),
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(
        x,
        f'{exp.work_dir}/graphs'),
    tb_writer=exp.tb_writer,
    eval_step=steps_per_epoch)


def lr_policy_cosine_decorator(num_steps):

    def lr_policy_cosine(initial_lr, step, e):
        progress = float(step / num_steps)
        out_lr = initial_lr * 0.5 * (1. + math.cos(math.pi * progress))
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
        progress = float(step / num_steps)
        if progress < warmup_proportion:
            out_lr = initial_lr * progress / warmup_proportion
        else:
            out_lr = initial_lr * \
                max((progress - 1.) / (warmup_proportion - 1.), 0.)
        return out_lr

    return lr_policy_warmup


if lr_policy == "lr_warmup":
    lr_policy_func = lr_policy_warmup_decorator(num_epochs * steps_per_epoch,
                                                args.lr_warmup_proportion)
elif lr_policy == "lr_poly":
    lr_policy_func = lr_policy_poly_decorator(num_epochs * steps_per_epoch)
elif lr_policy == "lr_cosine":
    lr_policy_func = lr_policy_cosine_decorator(num_epochs * steps_per_epoch)
else:
    raise ValueError("Invalid lr_policy, must be lr_warmup or lr_poly")

optimizer.train(
    tensors_to_optimize=[train_loss],
    callbacks=[callback_train, callback_eval, ckpt_callback],
    lr_policy=lr_policy_func,
    optimizer=args.optimizer_kind,
    optimization_params={"num_epochs": args.num_epochs,
                         "lr": args.lr,
                         "weight_decay": args.weight_decay})
