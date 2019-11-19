#!/usr/bin/env python3
# Copyright (c) 2019 NVIDIA Corporation
import argparse
import os

import nemo
from nemo.utils.lr_policies import get_lr_policy

import nemo_nlp
from nemo_nlp.data.datasets.utils import BERTPretrainingDataDesc
from nemo_nlp.utils.callbacks.bert_pretraining import \
    eval_iter_callback, eval_epochs_done_callback

<<<<<<< HEAD

console = logging.StreamHandler()
console.setLevel(logging.INFO)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
=======
>>>>>>> upstream/master

parser = argparse.ArgumentParser(description='BERT pretraining')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--eval_batch_size", default=16, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--lr_policy", default="CosineAnnealing", type=str)
parser.add_argument("--lr_warmup_proportion", default=0.05, type=float)
parser.add_argument("--optimizer", default="novograd", type=str)
parser.add_argument("--beta1", default=0.95, type=float)
parser.add_argument("--beta2", default=0.25, type=float)
parser.add_argument("--amp_opt_level",
                    default="O0",
                    type=str,
                    choices=["O0", "O1", "O2"])
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--vocab_size", default=3200, type=int)
parser.add_argument("--tokenizer",
                    default="sentence-piece",
                    type=str,
                    choices=["sentence-piece", "nemo-bert"])
parser.add_argument("--sample_size", default=1e7, type=int)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--mask_probability", default=0.15, type=float)
parser.add_argument("--short_seq_prob", default=0.1, type=float)
parser.add_argument("--d_model", default=768, type=int)
parser.add_argument("--d_inner", default=3072, type=int)
parser.add_argument("--num_layers", default=12, type=int)
parser.add_argument("--num_heads", default=12, type=int)
<<<<<<< HEAD
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--embedding_dropout", default=0.1, type=float)
parser.add_argument("--ffn_dropout", default=0.1, type=float)
parser.add_argument("--attn_score_dropout", default=0.1, type=float)
parser.add_argument("--attn_layer_dropout", default=0.1, type=float)
parser.add_argument("--conv_kernel_size", default=7, type=int)
parser.add_argument("--conv_weight_dropout", default=0.1, type=float)
parser.add_argument("--conv_layer_dropout", default=0.1, type=float)
parser.add_argument("--tokenizer_model", default="tokenizer.model",
                    type=str)
parser.add_argument("--dataset_dir", default="./pubmed-corpus", type=str)
parser.add_argument(
    "--dev_dataset_dir", default="./pubmed-corpus-test", type=str)
parser.add_argument("--train_sentence_indices_filename",
                    default="train_sentence_indices.pkl", type=str)
parser.add_argument("--dev_sentence_indices_filename",
                    default="dev_sentence_indices.pkl", type=str)
parser.add_argument("--checkpoint_directory", default="./checkpoint", type=str)
parser.add_argument("--checkpoint_save_frequency", default=25000, type=int)
parser.add_argument("--tensorboard_filename", default=None, type=str)
parser.add_argument("--fp16", default=0, type=int, choices=[0, 1, 2, 3])
parser.add_argument("--batch_per_step", default=1, type=int)
parser.add_argument("--short_seq_prob", default=0.1, type=float)
=======
parser.add_argument("--data_dir", default="data/lm/wikitext-2", type=str)
parser.add_argument("--dataset_name", default="wikitext-2", type=str)
parser.add_argument("--work_dir", default="outputs/bert_lm", type=str)
parser.add_argument("--save_epoch_freq", default=1, type=int)
parser.add_argument("--save_step_freq", default=-1, type=int)
>>>>>>> upstream/master
args = parser.parse_args()

nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                   local_rank=args.local_rank,
                                   optimization_level=args.amp_opt_level,
                                   log_dir=args.work_dir,
                                   create_tb_writer=True,
                                   files_to_copy=[__file__],
                                   add_time_to_log_dir=True)

special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
data_desc = BERTPretrainingDataDesc(args.dataset_name,
                                    args.data_dir,
                                    args.vocab_size,
                                    args.sample_size,
                                    special_tokens,
                                    'train.txt')

if args.tokenizer == "sentence-piece":
    nf.logger.info("To use SentencePieceTokenizer.")
    tokenizer = nemo_nlp.SentencePieceTokenizer(
        model_path=data_desc.tokenizer_model)
    tokenizer.add_special_tokens(special_tokens)
elif args.tokenizer == "nemo-bert":
    nf.logger.info("To use NemoBertTokenizer.")
    vocab_file = os.path.join(args.data_dir, 'vocab.txt')
    # To train on a Chinese dataset, use NemoBertTokenizer
    tokenizer = nemo_nlp.NemoBertTokenizer(vocab_file=vocab_file)
else:
    raise ValueError("Please add your tokenizer"
                     " or use sentence-piece or nemo-bert.")

bert_model = nemo_nlp.huggingface.BERT(
    vocab_size=tokenizer.vocab_size,
    num_layers=args.num_layers,
    d_model=args.d_model,
    num_heads=args.num_heads,
    d_inner=args.d_inner,
    max_seq_length=args.max_seq_length,
    hidden_act="gelu")

""" create necessary modules for the whole translation pipeline, namely
data layers, BERT encoder, and MLM and NSP loss functions
"""
mlm_classifier = nemo_nlp.TokenClassifier(args.d_model,
                                          num_classes=tokenizer.vocab_size,
                                          num_layers=1,
                                          log_softmax=True)
mlm_loss_fn = nemo_nlp.MaskedLanguageModelingLossNM()

nsp_classifier = nemo_nlp.SequenceClassifier(args.d_model,
                                             num_classes=2,
                                             num_layers=2,
                                             log_softmax=True)
nsp_loss_fn = nemo.backends.pytorch.common.CrossEntropyLoss()

bert_loss = nemo_nlp.LossAggregatorNM(num_inputs=2)

# tie weights of MLM softmax layer and embedding layer of the encoder
mlm_classifier.mlp.last_linear_layer.weight = \
    bert_model.bert.embeddings.word_embeddings.weight

<<<<<<< HEAD
train_data_layer = nemo_nlp.BertPretrainingDataLayer(
    tokenizer=tokenizer,
    dataset=args.dataset_dir,
    name="train",
    sentence_indices_filename=args.train_sentence_indices_filename,
    max_seq_length=args.max_sequence_length,
    short_seq_prob=args.short_seq_prob,
    mask_probability=args.mask_probability,
    batch_size=args.batch_size,
    factory=neural_factory)

dev_data_layer = nemo_nlp.BertPretrainingDataLayer(
    tokenizer=tokenizer,
    dataset=args.dev_dataset_dir,
    name="dev",
    sentence_indices_filename=args.dev_sentence_indices_filename,
    max_seq_length=args.max_sequence_length,
    short_seq_prob=args.short_seq_prob,
    mask_probability=args.mask_probability,
    batch_size=args.eval_batch_size,
    factory=neural_factory)

# training pipeline
input_ids, input_type_ids, input_mask, \
    output_ids, output_mask, nsp_labels = train_data_layer()
hidden_states = bert_model(input_ids=input_ids,
                           token_type_ids=input_type_ids,
                           attention_mask=input_mask)
train_mlm_log_probs = mlm_log_softmax(hidden_states=hidden_states)
train_mlm_loss = mlm_loss(log_probs=train_mlm_log_probs,
                      output_ids=output_ids,
                      output_mask=output_mask)
train_nsp_log_probs = nsp_log_softmax(hidden_states=hidden_states)
train_nsp_loss = nsp_loss(log_probs=train_nsp_log_probs, labels=nsp_labels)
train_loss = bert_loss(loss_1=train_mlm_loss, loss_2=train_nsp_loss)

# evaluation pipeline
input_ids_, input_type_ids_, input_mask_, \
    output_ids_, output_mask_, nsp_labels_ = dev_data_layer()
hidden_states_ = bert_model(input_ids=input_ids_,
                            token_type_ids=input_type_ids_,
                            attention_mask=input_mask_)
dev_mlm_log_probs = mlm_log_softmax(hidden_states=hidden_states_)
dev_mlm_loss = mlm_loss(log_probs=dev_mlm_log_probs,
                        output_ids=output_ids_,
                        output_mask=output_mask_)
dev_nsp_log_probs = nsp_log_softmax(hidden_states=hidden_states_)
dev_nsp_loss = nsp_loss(log_probs=dev_nsp_log_probs, labels=nsp_labels_)
=======

def create_pipeline(data_file, max_seq_length, mask_probability,
                    short_seq_prob, batch_size):
    data_layer = nemo_nlp.BertPretrainingDataLayer(tokenizer,
                                                   data_file,
                                                   max_seq_length,
                                                   mask_probability,
                                                   short_seq_prob,
                                                   batch_size=batch_size)
    steps_per_epoch = len(data_layer) // (batch_size * args.num_gpus)

    input_ids, input_type_ids, input_mask, \
        output_ids, output_mask, nsp_labels = data_layer()
    hidden_states = bert_model(input_ids=input_ids,
                               token_type_ids=input_type_ids,
                               attention_mask=input_mask)
    mlm_logits = mlm_classifier(hidden_states=hidden_states)
    mlm_loss = mlm_loss_fn(logits=mlm_logits,
                           output_ids=output_ids,
                           output_mask=output_mask)
    nsp_logits = nsp_classifier(hidden_states=hidden_states)
    nsp_loss = nsp_loss_fn(logits=nsp_logits, labels=nsp_labels)

    loss = bert_loss(loss_1=mlm_loss, loss_2=nsp_loss)
    return loss, [mlm_loss, nsp_loss], steps_per_epoch


train_loss, _, steps_per_epoch = create_pipeline(data_desc.train_file,
                                                 args.max_seq_length,
                                                 args.mask_probability,
                                                 args.short_seq_prob,
                                                 args.batch_size)
eval_loss, eval_tensors, _ = create_pipeline(data_desc.eval_file,
                                             args.max_seq_length,
                                             args.mask_probability,
                                             args.short_seq_prob,
                                             args.eval_batch_size)
>>>>>>> upstream/master

# callback which prints training loss and perplexity once in a while
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    print_func=lambda x: nf.logger.info("Loss: {:.3f}".format(x[0].item())),
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=nf.tb_writer)

<<<<<<< HEAD
callback_dev = nemo.core.EvaluatorCallback(
    eval_tensors=[dev_mlm_loss, dev_nsp_loss],
=======
eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=eval_tensors,
>>>>>>> upstream/master
    user_iter_callback=eval_iter_callback,
    user_epochs_done_callback=eval_epochs_done_callback,
    eval_step=steps_per_epoch,
    tb_writer=nf.tb_writer)

ckpt_callback = nemo.core.CheckpointCallback(folder=nf.checkpoint_dir,
                                             epoch_freq=args.save_epoch_freq,
                                             step_freq=args.save_step_freq)

# define learning rate decay policy
lr_policy_fn = get_lr_policy(args.lr_policy,
                             total_steps=args.num_epochs * steps_per_epoch,
                             warmup_ratio=args.lr_warmup_proportion)

config_path = f'{nf.checkpoint_dir}/bert-config.json'
if not os.path.exists(config_path):
    bert_model.config.to_json_file(config_path)

# define and launch training algorithm (optimizer)
nf.train(tensors_to_optimize=[train_loss],
         lr_policy=lr_policy_fn,
         callbacks=[train_callback, eval_callback, ckpt_callback],
         optimizer=args.optimizer,
         optimization_params={"batch_size": args.batch_size,
                              "num_epochs": args.num_epochs,
                              "lr": args.lr,
                              "weight_decay": args.weight_decay})
