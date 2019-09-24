#!/usr/bin/env python3
# Copyright (c) 2019 NVIDIA Corporation

import argparse
import math
import os

import nemo
from nemo.utils.lr_policies import SquareAnnealing, CosineAnnealing, \
    InverseSquareRootAnnealing

import nemo_nlp
from nemo_nlp.utils.callbacks.bert_pretraining import eval_iter_callback, \
    eval_epochs_done_callback


parser = argparse.ArgumentParser(description='BERT pretraining')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--beta1", default=0.95, type=float)
parser.add_argument("--beta2", default=0.25, type=float)
parser.add_argument("--optimizer", default="novograd", type=str)
parser.add_argument("--amp_opt_level", default="O0",
                    type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--lr_policy", default="CosineAnnealing", type=str)
parser.add_argument("--lr_warmup_proportion", default=0.05, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--eval_batch_size", default=16, type=int)
parser.add_argument("--max_sequence_length", default=128, type=int)
parser.add_argument("--mask_probability", default=0.15, type=float)
parser.add_argument("--d_model", default=768, type=int)
parser.add_argument("--d_inner", default=3072, type=int)
parser.add_argument("--num_layers", default=12, type=int)
parser.add_argument("--num_heads", default=12, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--embedding_dropout", default=0.1, type=float)
parser.add_argument("--tokenizer_model", default="tokenizer.model",
                    type=str)
parser.add_argument("--dataset_dir", default="data/lm/wikitext-2", type=str)
parser.add_argument("--work_dir", default="outputs", type=str)
parser.add_argument("--dataset_name", default="wikitext-2", type=str)
parser.add_argument("--save_epoch_freq", default=1, type=int)
parser.add_argument("--save_step_freq", default=-1, type=int)
args = parser.parse_args()

nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                   local_rank=args.local_rank,
                                   optimization_level=args.amp_opt_level,
                                   log_dir=args.work_dir,
                                   create_tb_writer=True,
                                   files_to_copy=[__file__])

data_desc = BERTLanguageModelDataDesc(
    args.dataset_name, args.data_dir, )


tokenizer = nemo_nlp.SentencePieceTokenizer(model_path=args.tokenizer_model)
tokenizer.add_special_tokens(["[MASK]", "[CLS]", "[SEP]"])
vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)

bert_model = nemo_nlp.huggingface.BERT(
    vocab_size=vocab_size,
    num_layers=args.num_layers,
    d_model=args.d_model,
    num_heads=args.num_heads,
    d_inner=args.d_inner,
    max_seq_length=args.max_sequence_length,
    hidden_act="gelu",
    factory=neural_factory)

# instantiate necessary modules for the whole translation pipeline, namely
# data layers, BERT encoder, and MLM and NSP loss functions
mlm_log_softmax = nemo_nlp.TransformerLogSoftmaxNM(
    vocab_size=vocab_size,
    d_model=args.d_model,
    factory=neural_factory)
mlm_loss = nemo_nlp.MaskedLanguageModelingLossNM(factory=neural_factory)

nsp_log_softmax = nemo_nlp.SentenceClassificationLogSoftmaxNM(
    d_model=args.d_model,
    num_classes=2,
    factory=neural_factory)
nsp_loss = nemo_nlp.NextSentencePredictionLossNM(factory=neural_factory)

bert_loss = nemo_nlp.LossAggregatorNM(
    num_inputs=2,
    factory=neural_factory)

# tie weights of MLM softmax layer and embedding layer of the encoder
mlm_log_softmax.log_softmax.dense.weight = \
    bert_model.bert.embeddings.word_embeddings.weight

train_data_layer = nemo_nlp.BertPretrainingDataLayer(
    tokenizer=tokenizer,
    dataset=args.dataset_dir,
    name="train",
    sentence_indices_filename=args.train_sentence_indices_filename,
    max_seq_length=args.max_sequence_length,
    mask_probability=args.mask_probability,
    batch_size=args.batch_size,
    factory=neural_factory)

dev_data_layer = nemo_nlp.BertPretrainingDataLayer(
    tokenizer=tokenizer,
    dataset=args.dev_dataset_dir,
    name="dev",
    sentence_indices_filename=args.dev_sentence_indices_filename,
    max_seq_length=args.max_sequence_length,
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
train_loss = mlm_loss(log_probs=train_mlm_log_probs,
                      output_ids=output_ids,
                      output_mask=output_mask)
# train_nsp_log_probs = nsp_log_softmax(hidden_states=hidden_states)
# train_nsp_loss = nsp_loss(log_probs=train_nsp_log_probs, labels=nsp_labels)
# train_loss = bert_loss(loss_1=train_mlm_loss, loss_2=train_nsp_loss)

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
# dev_nsp_log_probs = nsp_log_softmax(hidden_states=hidden_states_)
# dev_nsp_loss = nsp_loss(log_probs=dev_nsp_log_probs, labels=nsp_labels_)

# callback which prints training loss and perplexity once in a while
callback_loss = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    print_func=lambda x: print("Loss: {:.3f}".format(x[0].item())),
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=tb_writer)

callback_ckpt = nemo.core.CheckpointCallback(
    folder=args.checkpoint_directory,
    step_freq=args.checkpoint_save_frequency)

train_data_size = len(train_data_layer)
steps_per_epoch = int(train_data_size / (args.batch_size * args.num_gpus))

callback_dev = nemo.core.EvaluatorCallback(
    # eval_tensors=[dev_mlm_loss, dev_nsp_loss],
    eval_tensors=[dev_mlm_loss],
    user_iter_callback=eval_iter_callback,
    user_epochs_done_callback=eval_epochs_done_callback,
    eval_step=steps_per_epoch,
    tb_writer=tb_writer)

# define learning rate decay policy
if args.lr_decay_policy == "poly":
    lr_policy = SquareAnnealing(args.num_epochs * steps_per_epoch,
                                warmup_ratio=args.lr_warmup_proportion)
elif args.lr_decay_policy == "cosine":
    lr_policy = CosineAnnealing(args.num_epochs * steps_per_epoch,
                                warmup_ratio=args.lr_warmup_proportion)
elif args.lr_decay_policy == "noam":
    lr_policy = \
        InverseSquareRootAnnealing(args.num_epochs * steps_per_epoch,
                                   warmup_ratio=args.lr_warmup_proportion)
else:
    raise NotImplementedError

# save config file
if not os.path.exists(args.checkpoint_directory):
    os.makedirs(args.checkpoint_directory)

config_path = os.path.join(args.checkpoint_directory, "bert-config.json")
if not os.path.exists(config_path):
    bert_model.config.to_json_file(config_path)

# define and launch training algorithm (optimizer)
optimizer = neural_factory.get_trainer()
optimizer.train(tensors_to_optimize=[train_loss],
                lr_policy=lr_policy,
                callbacks=[callback_loss, callback_ckpt, callback_dev],
                optimizer=args.optimizer,
                optimization_params={
                    "batch_size": args.batch_size,
                    "num_epochs": args.num_epochs,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "betas": (args.beta1, args.beta2),
                    "grad_norm_clip": None
})
