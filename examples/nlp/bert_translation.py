# Copyright (c) 2019 NVIDIA Corporation
import nemo
import math
import logging
import argparse
from nemo.utils.lr_policies import SquareAnnealing, CosineAnnealing, \
    InverseSquareRootAnnealing
from nemo_nlp.callbacks.translation import eval_iter_callback, \
    eval_epochs_done_callback
from nemo_nlp import NemoBertTokenizer
#from tensorboardX import SummaryWriter

console = logging.StreamHandler()
console.setLevel(logging.INFO)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

parser = argparse.ArgumentParser(
    description="EN-DE translation with pretrained BERT encoder")
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--optimizer", default="novograd", type=str)
parser.add_argument("--pretrained_model_name",
                    default="bert-base-multilingual-cased", type=str)
parser.add_argument("--lr_decay_policy", default="poly", type=str)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--warmup_steps", default=4000, type=int)
parser.add_argument("--batch_size", default=5120, type=int)
parser.add_argument("--eval_batch_size", default=320, type=int)
parser.add_argument("--max_sequence_length", default=256, type=int)
parser.add_argument("--d_model", default=768, type=int)
parser.add_argument("--d_inner", default=3072, type=int)
parser.add_argument("--num_layers", default=6, type=int)
parser.add_argument("--num_heads", default=12, type=int)
parser.add_argument("--embedding_dropout", default=0.2, type=float)
parser.add_argument("--fully_connected_dropout", default=0.2, type=float)
parser.add_argument("--attn_score_dropout", default=0.2, type=float)
parser.add_argument("--attn_layer_dropout", default=0.2, type=float)
parser.add_argument("--conv_kernel_size", default=7, type=int)
parser.add_argument("--conv_weight_dropout", default=0.2, type=float)
parser.add_argument("--conv_layer_dropout", default=0.2, type=float)
parser.add_argument(
    "--decoder_first_sub_layer", default="self_attention", type=str)
parser.add_argument("--eval_step_frequency", default=10000, type=int)
parser.add_argument("--max_num_steps", default=100000, type=int)
parser.add_argument("--dataset_dir", default="/dataset/", type=str)
parser.add_argument("--src_lang", default="en", type=str)
parser.add_argument("--tgt_lang", default="de", type=str)
parser.add_argument("--fp16", default=0, type=int, choices=[0, 1, 2])
parser.add_argument("--beam_size", default=4, type=int)
parser.add_argument("--len_pen", default=0.0, type=float)
parser.add_argument("--batch_accumulation_steps", default=1, type=int)
parser.add_argument("--max_num_epochs", default=100, type=int)
args = parser.parse_args()

# create TensorboardX logger to log training statistics
name = "translation_with_bert_encoder-lr{0}-opt{1}-warmup{2}-{3}-bs{4}".format(
    args.lr, args.optimizer, args.warmup_steps,
    args.lr_decay_policy, args.batch_size)
tb_writer = None #SummaryWriter(name)

# instantiate Neural Factory with supported backend
device = nemo.core.DeviceType.AllGpu if args.local_rank is not None \
    else nemo.core.DeviceType.GPU
if args.fp16 == 0:
   optimization_level = nemo.core.Optimization.nothing
elif args.fp16 == 1:
   optimization_level = nemo.core.Optimization.mxprO1
elif args.fp16 == 2:
   optimization_level = nemo.core.Optimization.mxprO2
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=optimization_level,
    placement=device)

# define the parameters for the first sub layer in Transformer block
# currently only self-attention and lightweight-conv are supported
if args.decoder_first_sub_layer == "lightweight_conv":
    dec_first_sublayer_params = {
        "first_sub_layer": args.decoder_first_sub_layer,
        "conv_kernel_size": args.conv_kernel_size,
        "conv_weight_dropout": args.conv_weight_dropout,
        "conv_layer_dropout": args.conv_layer_dropout,
    }
elif args.decoder_first_sub_layer == "self_attention":
    dec_first_sublayer_params = {
        "first_sub_layer": args.decoder_first_sub_layer,
        "attn_score_dropout": args.attn_score_dropout,
        "attn_layer_dropout": args.attn_layer_dropout,
    }
else:
    raise NotImplementedError

# define tokenizer, in this example we use WordPiece BertTokenizer
# we also increase the vocabulary size to make it multiple of 8 to accelerate
# training in fp16 mode with the use of Tensor Cores
tokenizer = NemoBertTokenizer(
    pretrained_model=args.pretrained_model_name + "-vocab.txt")
vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)

# instantiate necessary modules for the whole translation pipeline, namely
# data layers, encoder (pre-trained Bert), decoder, output log_softmax,
# beam_search_translator and loss function
train_data_layer = neural_factory.get_module(
    name="TranslationDataLayer",
    params={
        "tokenizer_src": tokenizer,
        "tokenizer_tgt": tokenizer,
        "dataset_src": args.dataset_dir + "train." + args.src_lang,
        "dataset_tgt": args.dataset_dir + "train." + args.tgt_lang,
        "tokens_in_batch": args.batch_size,
        "clean": True
    },
    collection="nemo_nlp"
)
eval_data_layer = neural_factory.get_module(
    name="TranslationDataLayer",
    params={
        "tokenizer_src": tokenizer,
        "tokenizer_tgt": tokenizer,
        "dataset_src": args.dataset_dir + "test." + args.src_lang,
        "dataset_tgt": args.dataset_dir + "test." + args.tgt_lang,
        "tokens_in_batch": args.eval_batch_size,
        "clean": False
    },
    collection="nemo_nlp"
)
zeros_transform = neural_factory.get_module(
    name="ZeroTransformation",
    params={},
    collection="nemo_nlp"
)
encoder = neural_factory.get_module(
  name="BertEncoderNM",
  params={
      "pretrained_model_name": args.pretrained_model_name,
      "local_rank": args.local_rank,
      "pretrained": True
  },
  collection="nemo_nlp"
)
decoder = neural_factory.get_module(
  name="TransformerDecoderNM",
  params={
    "d_model": args.d_model,
    "d_inner": args.d_inner,
    "num_layers": args.num_layers,
    "num_attn_heads": args.num_heads,
    "fully_connected_dropout": args.fully_connected_dropout,
    "vocab_size": vocab_size,
    "max_seq_length": args.max_sequence_length,
    "embedding_dropout": args.embedding_dropout,
    **dec_first_sublayer_params
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
beam_translator = neural_factory.get_module(
    name="BeamSearchTranslatorNM",
    params={
        "decoder": decoder,
        "log_softmax": t_log_softmax,
        "max_seq_length": args.max_sequence_length,
        "beam_size": args.beam_size,
        "length_penalty": args.len_pen,
        "bos_token": tokenizer.bos_id(),
        "pad_token": tokenizer.pad_id(),
        "eos_token": tokenizer.eos_id()
    },
    collection="nemo_nlp"
)
loss = neural_factory.get_module(
    name="PaddedSmoothedCrossEntropyLossNM",
    params={
        "pad_id": tokenizer.pad_id(),
        "label_smoothing": 0.1
    },
    collection="nemo_nlp"
)
loss_eval = neural_factory.get_module(
    name="PaddedSmoothedCrossEntropyLossNM",
    params={
        "pad_id": tokenizer.pad_id(),
        "label_smoothing": 0.0
    },
    collection="nemo_nlp"
)

# tie weights of softmax layer and embedding layers from encoder and decoder

t_log_softmax.log_softmax.dense.weight = encoder.embedding_layer.token_embedding.weight
decoder.embedding_layer.token_embedding.weight = encoder.embedding_layer.token_embedding.weight
decoder.embedding_layer.position_embedding.pos_enc = encoder.embedding_layer.position_embedding.weight

# t_log_softmax.tie_weights_with(
#     encoder,
#     weight_names=["log_softmax.dense.weight"],
#     name2name_and_transform={
#         "log_softmax.dense.weight": ("embedding_layer.weight", 0)
#     }
# )
#
# decoder.tie_weights_with(
#     encoder,
#     weight_names=["embeddings.word_embedding.weight"],
#     name2name_and_transform={
#         "embeddings.word_embedding.weight": ("embedding_layer.weight", 0)
#     }
# )

# training pipeline
src, src_mask, tgt, tgt_mask, labels, sent_ids = train_data_layer()
input_type_ids = zeros_transform(input_type_ids=src)
src_hiddens = encoder(input_ids=src,
                      input_type_ids=input_type_ids,
                      input_mask=src_mask)
tgt_hiddens = decoder(input_ids_tgt=tgt,
                      hidden_states_src=src_hiddens,
                      input_mask_src=src_mask,
                      input_mask_tgt=tgt_mask)
log_softmax = t_log_softmax(hidden_states=tgt_hiddens)
train_loss = loss(log_probs=log_softmax, target_ids=labels)

# evaluation pipeline with beam search on top of the model output
src_, src_mask_, tgt_, tgt_mask_, labels_, sent_ids_ = eval_data_layer()
input_type_ids_ = zeros_transform(input_type_ids=src_)
src_hiddens_ = encoder(input_ids=src_,
                       input_type_ids=input_type_ids_,
                       input_mask=src_mask_)
tgt_hiddens_ = decoder(input_ids_tgt=tgt_,
                       hidden_states_src=src_hiddens_,
                       input_mask_src=src_mask_,
                       input_mask_tgt=tgt_mask_)
log_softmax_ = t_log_softmax(hidden_states=tgt_hiddens_)
eval_loss = loss_eval(log_probs=log_softmax_, target_ids=labels_)
beam_trans = beam_translator(
  hidden_states_src=src_hiddens_, input_mask_src=src_mask_
)

# callback which prints training loss once in a while
callback = nemo.core.SimpleLossLoggerCallback(
    tensor_list2string=lambda x: str(x[0].item()),
    tensorboard_writer=tb_writer,
    step_frequency=100)
# callback which calculates evaluation loss without label smoothing
# and BLEU scores between outputs of beam search and reference translations
callback_dev = nemo.core.EvaluatorCallback(
    eval_tensors=[tgt_, eval_loss, beam_trans, sent_ids_],
    user_iter_callback=lambda x, y: eval_iter_callback(x, y, tokenizer),
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(
        x, validation_dataset=args.dataset_dir + "test." + args.tgt_lang),
    eval_step=args.eval_step_frequency,
    tensorboard_writer=tb_writer)

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
            "num_epochs": args.max_num_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay
        }
    }
)
optimizer.train(
    tensors_to_optimize=[train_loss],
    local_rank=args.local_rank,
    tensors_to_evaluate=[],
    lr_policy=lr_policy,
    callbacks=[callback, callback_dev],
    batches_per_step=args.batch_accumulation_steps
)