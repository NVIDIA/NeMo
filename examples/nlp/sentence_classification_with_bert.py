import argparse
import math
import os

import numpy as np
from pytorch_transformers import BertTokenizer

import nemo
from nemo.utils.lr_policies import get_lr_policy

import nemo_nlp
from nemo_nlp.callbacks.sentence_classification import \
    eval_iter_callback, eval_epochs_done_callback
from nemo_nlp.text_data_utils import \
    process_sst_2, process_imdb, process_nlu, process_nvidia_car

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
parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
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
parser.add_argument("--amp_opt_level", default="O0",
                    type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--do_lower_case", action='store_false')
parser.add_argument("--shuffle_data", action='store_false')

args = parser.parse_args()

work_dir = f'{args.work_dir}/{args.dataset_name.upper()}'
nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                   local_rank=args.local_rank,
                                   optimization_level=args.amp_opt_level,
                                   log_dir=work_dir,
                                   create_tb_writer=True,
                                   files_to_copy=[__file__],
                                   add_time_to_log_dir=True)

# Load the pretrained BERT parameters
# See the list of pretrained models here:
# https://huggingface.co/pytorch-transformers/pretrained_models.html
pretrained_bert_model = nemo_nlp.huggingface.BERT(
    pretrained_model_name=args.pretrained_bert_model, factory=nf)

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
        nf.logger.info(f'{intent}: {idx}')
    num_labels = len(labels)
    devfile = data_dir + '/test.tsv'
else:
    nf.logger.info("Looks like you pass in the name of dataset that isn't "
                   "already supported by NeMo. Please make sure that you "
                   "build the preprocessing method for it.")

# Create sentence classification loss on top
hidden_size = pretrained_bert_model.local_parameters["hidden_size"]
classifier = nemo_nlp.SequenceClassifier(hidden_size=hidden_size,
                                         num_classes=num_labels,
                                         dropout=args.fc_dropout)

loss_fn = nemo.backends.pytorch.common.CrossEntropyLoss(
    factory=nf)

tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)


def create_pipeline(data_file,
                    max_seq_length,
                    batch_size=32,
                    num_samples=-1,
                    shuffle=True,
                    num_gpus=1,
                    local_rank=0,
                    mode='train'):
    nf.logger.info(f"Loading {mode} data...")
    data_layer = nemo_nlp.BertSentenceClassificationDataLayer(
        path_to_data=data_file,
        tokenizer=tokenizer,
        mode=mode,
        max_seq_length=max_seq_length,
        num_samples=num_samples,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        local_rank=local_rank
    )

    ids, type_ids, input_mask, labels = data_layer()
    data_size = len(data_layer)

    if data_size < batch_size:
        nf.logger.warning("Batch_size is larger than the dataset size")
        nf.logger.warning("Reducing batch_size to dataset size")
        batch_size = data_size

    steps_per_epoch = int(data_size / (batch_size * num_gpus))
    nf.logger.info(f"Steps_per_epoch = {steps_per_epoch}")

    hidden_states = pretrained_bert_model(input_ids=ids,
                                          token_type_ids=type_ids,
                                          attention_mask=input_mask)

    logits = classifier(hidden_states=hidden_states)
    loss = loss_fn(logits=logits, labels=labels)

    # Create trainer and execute training action
    if mode == 'train':
        callback_fn = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss, logits],
            print_func=lambda x: str(np.round(x[0].item(), 3)),
            tb_writer=nf.tb_writer,
            get_tb_values=lambda x: [["loss", x[0]]],
            step_freq=100)
    elif mode == 'eval':
        callback_fn = nemo.core.EvaluatorCallback(
            eval_tensors=[logits, labels],
            user_iter_callback=lambda x, y: eval_iter_callback(
                x, y, data_layer),
            user_epochs_done_callback=lambda x: eval_epochs_done_callback(
                x, f'{nf.work_dir}/graphs'),
            tb_writer=nf.tb_writer,
            eval_step=steps_per_epoch)

    return loss, callback_fn, steps_per_epoch


train_loss, callback_train, steps_per_epoch =\
    create_pipeline(data_dir + '/train.tsv',
                    max_seq_length=args.max_seq_length,
                    batch_size=args.batch_size,
                    num_samples=args.num_train_samples,
                    shuffle=args.shuffle_data,
                    num_gpus=args.num_gpus,
                    local_rank=args.local_rank,
                    mode='train')
_, callback_eval, _ =\
    create_pipeline(data_dir + '/test.tsv',
                    max_seq_length=args.max_seq_length,
                    batch_size=args.batch_size,
                    num_samples=args.num_train_samples,
                    shuffle=False,
                    num_gpus=args.num_gpus,
                    local_rank=args.local_rank,
                    mode='eval')


# Create callback to save checkpoints
ckpt_callback = nemo.core.CheckpointCallback(
    folder=nf.checkpoint_dir,
    epoch_freq=args.save_epoch_freq,
    step_freq=args.save_step_freq)

lr_policy_fn = get_lr_policy(args.lr_policy,
                             total_steps=args.num_epochs * steps_per_epoch,
                             warmup_ratio=args.lr_warmup_proportion)
nf.train(tensors_to_optimize=[train_loss],
         callbacks=[callback_train, callback_eval, ckpt_callback],
         lr_policy=lr_policy_fn,
         optimizer=args.optimizer_kind,
         optimization_params={"num_epochs": args.num_epochs,
                              "lr": args.lr,
                              "weight_decay": args.weight_decay})
