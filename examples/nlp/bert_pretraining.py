# Copyright (c) 2019 NVIDIA Corporation
import math
import nemo
import numpy as np
import logging
import argparse
from nemo.utils.lr_policies import SquareAnnealing, CosineAnnealing, \
    InverseSquareRootAnnealing
from nemo_nlp.callbacks.bert_pretraining import eval_iter_callback, \
    eval_epochs_done_callback
from nemo_nlp import NemoBertTokenizer
from tensorboardX import SummaryWriter

console = logging.StreamHandler()
console.setLevel(logging.INFO)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

parser = argparse.ArgumentParser(description='BERT pretraining on WikiText-2')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--optimizer", default="novograd", type=str)
parser.add_argument("--lr_decay_policy", default="poly", type=str)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--eval_batch_size", default=16, type=int)
parser.add_argument("--max_sequence_length", default=128, type=int)
parser.add_argument("--mask_probability", default=0.05, type=float)
parser.add_argument("--d_model", default=512, type=int)
parser.add_argument("--d_inner", default=2048, type=int)
parser.add_argument("--num_layers", default=4, type=int)
parser.add_argument("--num_heads", default=8, type=int)
parser.add_argument("--embedding_dropout", default=0.1, type=float)
parser.add_argument("--fully_connected_dropout", default=0.1, type=float)
parser.add_argument("--attn_score_dropout", default=0.1, type=float)
parser.add_argument("--attn_layer_dropout", default=0.1, type=float)
parser.add_argument("--conv_kernel_size", default=7, type=int)
parser.add_argument("--conv_weight_dropout", default=0.1, type=float)
parser.add_argument("--conv_layer_dropout", default=0.1, type=float)
parser.add_argument(
    "--encoder_first_sub_layer", default="self_attention", type=str)
parser.add_argument("--eval_step_frequency", default=1000, type=int)
parser.add_argument("--max_num_steps", default=100000, type=int)
parser.add_argument("--dataset_dir", default="/dataset/", type=str)
parser.add_argument("--fp16", default=0, type=int, choices=[0, 1, 2])
args = parser.parse_args()

name = "bert_base-lr{0}-opt{1}-warmup{2}-{3}-bs{4}".format(
    args.lr, args.optimizer, args.warmup_steps,
    args.lr_decay_policy, args.batch_size)
tb_writer = SummaryWriter(name)

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

# define the parameters for the first sub layer in Transformer block
# currently only self-attention and lightweight-conv are supported
if args.encoder_first_sub_layer == "lightweight_conv":
    enc_first_sublayer_params = {
        "first_sub_layer": args.encoder_first_sub_layer,
        "conv_kernel_size": args.conv_kernel_size,
        "conv_weight_dropout": args.conv_weight_dropout,
        "conv_layer_dropout": args.conv_layer_dropout,
    }
elif args.encoder_first_sub_layer == "self_attention":
    enc_first_sublayer_params = {
        "first_sub_layer": args.encoder_first_sub_layer,
        "attn_score_dropout": args.attn_score_dropout,
        "attn_layer_dropout": args.attn_layer_dropout,
    }
else:
    raise NotImplementedError

# define tokenizer, in this example we use word-level BertTokenizer
tokenizer = NemoBertTokenizer(
    vocab_file="../../tests/data/wikitext-2/vocab.txt",
    do_lower_case=False,
    max_len=args.max_sequence_length)
vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)

# instantiate necessary modules for the whole translation pipeline, namely
# data layers, BERT encoder, and MLM and NSP loss functions
train_data_layer = neural_factory.get_module(
    name="BertPretrainingDataLayer",
    params={
        "tokenizer": tokenizer,
        "dataset": "../../tests/data/wikitext-2/train.txt",
        "max_seq_length": args.max_sequence_length,
        "mask_probability": args.mask_probability,
        "batch_size": args.batch_size,
        "num_workers": 0
    },
    collection="nemo_nlp"
)
dev_data_layer = neural_factory.get_module(
    name="BertPretrainingDataLayer",
    params={
        "tokenizer": tokenizer,
        "dataset": "../../tests/data/wikitext-2/valid.txt",
        "max_seq_length": args.max_sequence_length,
        "mask_probability": args.mask_probability,
        "batch_size": args.eval_batch_size,
        "num_workers": 0
    },
    collection="nemo_nlp"
)
bert_encoder = neural_factory.get_module(
    name="BertEncoderNM",
    params={
        "vocab_size": vocab_size,
        "num_layers": args.num_layers,
        "d_model": args.d_model,
        "num_attn_heads": args.num_heads,
        "d_inner": args.d_inner,
        "max_seq_length": args.max_sequence_length,
        "embedding_dropout": args.embedding_dropout,
        "fully_connected_dropout": args.fully_connected_dropout,
        "hidden_act": "gelu",
        **enc_first_sublayer_params
    },
    collection="nemo_nlp"
)

mlm_log_softmax = neural_factory.get_module(
    name="TransformerLogSoftmaxNM",
    params={"vocab_size": vocab_size, "d_model": args.d_model},
    collection="nemo_nlp"
)
mlm_loss = neural_factory.get_module(
    name="MaskedLanguageModelingLossNM",
    params={},
    collection="nemo_nlp"
)

nsp_log_softmax = neural_factory.get_module(
    name="SentenceClassificationLogSoftmaxNM",
    params={"d_model": args.d_model, "num_classes": 2},
    collection="nemo_nlp"
)
nsp_loss = neural_factory.get_module(
    name="NextSentencePredictionLossNM",
    params={},
    collection="nemo_nlp"
)

bert_loss = neural_factory.get_module(
    name="LossAggregatorNM",
    params={"num_inputs": 2},
    collection="nemo_nlp"
)

# tie weights of MLM softmax layer and embedding layer of the encoder
# mlm_loss.tie_weights_with(
#     bert_model,
#     weight_names=["log_softmax.dense.weight"],
#     name2name_and_transform={
#         "log_softmax.dense.weight": (
#             "embedding_layer.word_embedding.weight", 0)
#     }
# )
mlm_log_softmax.log_softmax.dense.weight = \
    bert_encoder.embedding_layer.token_embedding.weight

# training pipeline
input_ids, input_type_ids, input_mask, \
output_ids, output_mask, nsp_labels = train_data_layer()
hidden_states = bert_encoder(input_ids=input_ids,
                             input_type_ids=input_type_ids,
                             input_mask=input_mask)
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
hidden_states_ = bert_encoder(input_ids=input_ids_,
                              input_type_ids=input_type_ids_,
                              input_mask=input_mask_)
dev_mlm_log_probs = mlm_log_softmax(hidden_states=hidden_states_)
dev_mlm_loss = mlm_loss(log_probs=dev_mlm_log_probs,
                          output_ids=output_ids_,
                          output_mask=output_mask_)
dev_nsp_log_probs = nsp_log_softmax(hidden_states=hidden_states_)
dev_nsp_loss = nsp_loss(log_probs=dev_nsp_log_probs, labels=nsp_labels_)

# callback which prints training loss and perplexity once in a while
def get_loss_and_ppl(loss):
    str_ = str(np.round(loss, 3)) + \
           ", perplexity = " + str(np.round(np.exp(loss), 3))
    return str_
callback_loss = nemo.core.SimpleLossLoggerCallback(
    tensor_list2string=lambda x: get_loss_and_ppl(x[0].item()),
    step_frequency=100)
# callback which calculates evaluation loss on held-out validation data
callback_dev = nemo.core.EvaluatorCallback(
    eval_tensors=[dev_mlm_loss, dev_nsp_loss],
    user_iter_callback=eval_iter_callback,
    user_epochs_done_callback=eval_epochs_done_callback,
    eval_step=500)

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
        "optimization_params":{
            "num_epochs": 40,
            "lr": args.lr,
            "weight_decay": args.weight_decay
        }
    }
)
optimizer.train(tensors_to_optimize=[train_loss],
                local_rank=args.local_rank,
                tensors_to_evaluate=[],
                lr_policy=lr_policy,
                callbacks=[callback_loss, callback_dev])
