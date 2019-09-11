# Copyright (c) 2019 NVIDIA Corporation
import nemo
import math
import logging
import argparse
from nemo.utils.lr_policies import SquareAnnealing, CosineAnnealing, \
    InverseSquareRootAnnealing
from nemo_nlp.callbacks.language_modeling import eval_iter_callback, \
    eval_epochs_done_callback
from nemo_nlp import WordTokenizer

console = logging.StreamHandler()
console.setLevel(logging.INFO)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

parser = argparse.ArgumentParser(description='Transformer EN-DE translation')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--lr", default=0.002, type=float)
parser.add_argument("--optimizer", default="novograd", type=str)
parser.add_argument("--lr_decay_policy", default="cosine", type=str)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--eval_batch_size", default=16, type=int)
parser.add_argument("--max_sequence_length", default=128, type=int)
parser.add_argument("--d_model", default=384, type=int)
parser.add_argument("--d_inner", default=1920, type=int)
parser.add_argument("--num_layers", default=10, type=int)
parser.add_argument("--num_heads", default=12, type=int)
parser.add_argument("--embedding_dropout", default=0.3, type=float)
parser.add_argument("--ffn_dropout", default=0.3, type=float)
parser.add_argument("--attn_score_dropout", default=0.3, type=float)
parser.add_argument("--attn_layer_dropout", default=0.3, type=float)
parser.add_argument("--conv_kernel_size", default=7, type=int)
parser.add_argument("--conv_weight_dropout", default=0.1, type=float)
parser.add_argument("--conv_layer_dropout", default=0.1, type=float)
parser.add_argument(
    "--encoder_first_sub_layer", default="self_attention", type=str)
parser.add_argument("--eval_step_frequency", default=500, type=int)
parser.add_argument("--max_num_steps", default=100000, type=int)
parser.add_argument(
    "--dataset_dir",
    default="../../tests/data/wikitext-2/",
    type=str)
parser.add_argument("--fp16", default=0, type=int, choices=[0, 1, 2])
parser.add_argument("--beam_size", default=4, type=int)
parser.add_argument("--len_pen", default=0.0, type=float)
parser.add_argument("--batch_accumulation_steps", default=1, type=int)
parser.add_argument("--max_num_epochs", default=1000, type=int)
parser.add_argument("--train_dataset", default="train", type=str)
parser.add_argument("--eval_datasets", default="test", type=str)
parser.add_argument("--tokenizer_model", default="vocab.txt", type=str)
parser.add_argument("--predict_last_k", default=0, type=int)
args = parser.parse_args()

# create TensorboardX logger to log training statistics
name = "transformer_big-lr{0}-opt{1}-warmup{2}-{3}-bs{4}".format(
    args.lr, args.optimizer, args.warmup_steps,
    args.lr_decay_policy, args.batch_size)
tb_writer = None  # SummaryWriter(name)

# instantiate Neural Factory with supported backend
device = nemo.core.DeviceType.AllGpu if args.local_rank is not None \
    else nemo.core.DeviceType.GPU
if args.fp16 == 1:
    optimization_level = nemo.core.Optimization.mxprO1
elif args.fp16 == 2:
    optimization_level = nemo.core.Optimization.mxprO2
else:
    optimization_level = nemo.core.Optimization.nothing
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=optimization_level,
    placement=device)

# define tokenizer, in this example we use word-level tokenizer
# we also increase the vocabulary size to make it multiple of 8 to accelerate
# training in fp16 mode with the use of Tensor Cores
tokenizer = WordTokenizer(args.dataset_dir + args.tokenizer_model)
vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)

# instantiate necessary modules for the whole language modeling pipeline,
# namely data layers, encoder (which is basically transformer decoder without
# encoder-decoder attention, output log_softmax, greedy language generator
# and loss function
train_dataset = args.dataset_dir + args.train_dataset + ".txt"
eval_datasets = args.dataset_dir + args.eval_datasets + ".txt"

train_data_layer = neural_factory.get_module(
    name="LanguageModelingDataLayer",
    params={
        "tokenizer": tokenizer,
        "dataset": train_dataset,
        "max_seq_length": args.max_sequence_length,
        "batch_size": args.batch_size,
        "batch_step": args.max_sequence_length // 2
    },
    collection="nemo_nlp"
)
eval_data_layer = neural_factory.get_module(
    name="LanguageModelingDataLayer",
    params={
        "tokenizer": tokenizer,
        "dataset": eval_datasets,
        "max_seq_length": args.max_sequence_length,
        "batch_size": args.batch_size,
        "batch_step": args.predict_last_k
    },
    collection="nemo_nlp"
)
generation_data_layer = neural_factory.get_module(
    name="LanguageModelingDataLayer",
    params={
        "tokenizer": tokenizer,
        "dataset": eval_datasets,
        "max_seq_length": args.max_sequence_length - args.predict_last_k,
        "batch_size": args.batch_size,
        "batch_step": args.predict_last_k
    },
    collection="nemo_nlp"
)

t_encoder = neural_factory.get_module(
    name="TransformerEncoderNM",
    params={
        "d_model": args.d_model,
        "d_inner": args.d_inner,
        "num_layers": args.num_layers,
        "num_attn_heads": args.num_heads,
        "ffn_dropout": args.ffn_dropout,
        "vocab_size": vocab_size,
        "max_seq_length": args.max_sequence_length,
        "embedding_dropout": args.embedding_dropout,
        "mask_future": True,
        "first_sub_layer": args.encoder_first_sub_layer,
        "attn_score_dropout": args.attn_score_dropout,
        "attn_layer_dropout": args.attn_layer_dropout,
        "conv_kernel_size": args.conv_kernel_size,
        "conv_weight_dropout": args.conv_weight_dropout,
        "conv_layer_dropout": args.conv_layer_dropout
    },
    collection="nemo_nlp"
)

t_log_softmax = neural_factory.get_module(
    name="TransformerLogSoftmaxNM",
    params={
        "vocab_size": vocab_size,
        "d_model": args.d_model
    },
    collection="nemo_nlp"
)

t_loss = neural_factory.get_module(
    name="PaddedSmoothedCrossEntropyLossNM",
    params={
        "pad_id": tokenizer.pad_id(),
        "label_smoothing": 0
    },
    collection="nemo_nlp"
)
t_loss_eval = neural_factory.get_module(
    name="PaddedSmoothedCrossEntropyLossNM",
    params={
        "pad_id": tokenizer.pad_id(),
        "label_smoothing": 0,
        "predict_last_k": args.predict_last_k
    },
    collection="nemo_nlp"
)
language_generator = neural_factory.get_module(
    name="GreedyLanguageGeneratorNM",
    params={
        "decoder": t_encoder,
        "log_softmax": t_log_softmax,
        "max_seq_length": args.max_sequence_length,
        "pad_token": tokenizer.pad_id(),
        "bos_token": tokenizer.bos_id(),
        "eos_token": tokenizer.eos_id()
    },
    collection="nemo_nlp"
)

t_log_softmax.log_softmax.dense.weight = \
    t_encoder.embedding_layer.token_embedding.weight

# training pipeline
src, src_mask, labels = train_data_layer()
src_hiddens = t_encoder(input_ids=src, input_mask_src=src_mask)
log_softmax = t_log_softmax(hidden_states=src_hiddens)
train_loss = t_loss(log_probs=log_softmax, target_ids=labels)

# evaluation pipeline
src_, src_mask_, labels_ = eval_data_layer()
src_hiddens_ = t_encoder(input_ids=src_, input_mask_src=src_mask_)
log_softmax_ = t_log_softmax(hidden_states=src_hiddens_)
eval_loss = t_loss_eval(log_probs=log_softmax_, target_ids=labels_)

# generation pipeline with greedy language generator on top of the model output
src__, src_mask__, labels__ = generation_data_layer()
generated_text = language_generator(input_ids=src__)

# callback which prints training loss once in a while
callback = nemo.core.SimpleLossLoggerCallback(
    tensor_list2str=lambda x: str(x[0].item()),
    tb_writer=tb_writer,
    step_freq=100)
# callback which calculates evaluation loss without label smoothing
# and BLEU scores between outputs of beam search and reference translations
callback_dev = nemo.core.EvaluatorCallback(
    eval_tensors=[eval_loss],
    user_iter_callback=eval_iter_callback,
    user_epochs_done_callback=eval_epochs_done_callback,
    eval_step=args.eval_step_frequency,
    tb_writer=tb_writer)

# define learning rate decay policy
if args.lr_decay_policy == "poly":
    lr_policy = SquareAnnealing(
        args.max_num_steps, warmup_steps=args.warmup_steps)
elif args.lr_decay_policy == "cosine":
    lr_policy = CosineAnnealing(
        args.max_num_steps, warmup_steps=args.warmup_steps)
elif args.lr_decay_policy == "noam":
    lr_policy = InverseSquareRootAnnealing(
        args.max_num_steps, warmup_steps=args.warmup_steps)
else:
    raise NotImplementedError

# define and launch training algorithm (optimizer)
optimizer = neural_factory.get_trainer(
    params={
        "optimizer_kind": args.optimizer,
        "optimization_params": {
            "num_epochs": args.max_num_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay
        }
    }
)
optimizer.train(
    tensors_to_optimize=[train_loss],
    tensors_to_evaluate=[],
    lr_policy=lr_policy,
    callbacks=[callback, callback_dev],
    batches_per_step=args.batch_accumulation_steps
)
