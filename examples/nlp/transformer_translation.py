# Copyright (c) 2019 NVIDIA Corporation
import nemo
import math
import logging
import argparse
from nemo.utils.lr_policies import SquareAnnealing, CosineAnnealing, \
    InverseSquareRootAnnealing
from nemo_nlp.callbacks.translation import eval_iter_callback, \
    eval_epochs_done_callback
from nemo_nlp import YouTokenToMeTokenizer

console = logging.StreamHandler()
console.setLevel(logging.INFO)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

parser = argparse.ArgumentParser(description='Transformer EN-DE translation')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--lr", default=0.0005, type=float)
parser.add_argument("--optimizer", default="adam", type=str)
parser.add_argument("--lr_decay_policy", default="noam", type=str)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--warmup_steps", default=2000, type=int)
parser.add_argument("--batch_size", default=4096, type=int)
parser.add_argument("--eval_batch_size", default=256, type=int)
parser.add_argument("--max_sequence_length", default=256, type=int)
parser.add_argument("--d_model", default=512, type=int)
parser.add_argument("--d_inner", default=2048, type=int)
parser.add_argument("--num_layers", default=6, type=int)
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
parser.add_argument(
  "--decoder_first_sub_layer", default="self_attention", type=str)
parser.add_argument("--eval_step_frequency", default=1000, type=int)
parser.add_argument("--max_num_steps", default=100000, type=int)
parser.add_argument("--dataset_dir", default="/dataset/", type=str)
parser.add_argument("--src_lang", default="en", type=str)
parser.add_argument("--tgt_lang", default="de", type=str)
parser.add_argument("--fp16", default=0, type=int, choices=[0, 1, 2])
parser.add_argument("--beam_size", default=4, type=int)
parser.add_argument("--len_pen", default=0.0, type=float)
parser.add_argument("--batch_accumulation_steps", default=1, type=int)
parser.add_argument("--max_num_epochs", default=1000, type=int)
parser.add_argument("--train_dataset", default="train", type=str)
parser.add_argument("--eval_dataset", default="test", type=str)
parser.add_argument("--tokenizer_model", default="m_common.model", type=str)
args = parser.parse_args()

# create TensorboardX logger to log training statistics
name = "transformer_big-lr{0}-opt{1}-warmup{2}-{3}-bs{4}".format(
    args.lr, args.optimizer, args.warmup_steps,
    args.lr_decay_policy, args.batch_size)
tb_writer = None #SummaryWriter(name)

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

# define tokenizer, in this example we use SentencePiece tokenizer trained
# on joint English & German data for both source and target languages
tokenizer = YouTokenToMeTokenizer(args.dataset_dir + args.tokenizer_model)
#tokenizer = SentencePieceTokenizer(args.dataset_dir + "m_common.model")
vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)

# instantiate necessary modules for the whole translation pipeline, namely
# data layers, encoder, decoder, output log_softmax, beam_search_translator
# and loss function
train_dataset_src = args.dataset_dir + args.train_dataset + "." + args.src_lang
train_dataset_tgt = args.dataset_dir + args.train_dataset + "." + args.tgt_lang
eval_dataset_src = args.dataset_dir + args.eval_dataset + "." + args.src_lang
eval_dataset_tgt = args.dataset_dir + args.eval_dataset + "." + args.tgt_lang

train_data_layer = neural_factory.get_module(
    name="TranslationDataLayer",
    params={
        "tokenizer_src": tokenizer,
        "tokenizer_tgt": tokenizer,
        "dataset_src": train_dataset_src,
        "dataset_tgt": train_dataset_tgt,
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
        "dataset_src": eval_dataset_src,
        "dataset_tgt": eval_dataset_tgt,
        "tokens_in_batch": args.eval_batch_size,
        "clean": False
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
        "fully_connected_dropout": args.fully_connected_dropout,
        "vocab_size": vocab_size,
        "max_seq_length": args.max_sequence_length,
        "embedding_dropout": args.embedding_dropout,
        "first_sub_layer": args.encoder_first_sub_layer,
        "attn_score_dropout": args.attn_score_dropout,
        "attn_layer_dropout": args.attn_layer_dropout,
        "conv_kernel_size": args.conv_kernel_size,
        "conv_weight_dropout": args.conv_weight_dropout,
        "conv_layer_dropout": args.conv_layer_dropout
    },
    collection="nemo_nlp"
)
t_decoder = neural_factory.get_module(
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
        "first_sub_layer": args.decoder_first_sub_layer,
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
beam_translator = neural_factory.get_module(
    name="BeamSearchTranslatorNM",
    params={
        "decoder": t_decoder,
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
t_loss = neural_factory.get_module(
    name="PaddedSmoothedCrossEntropyLossNM",
    params={
        "pad_id": tokenizer.pad_id(),
        "label_smoothing": 0.1
    },
    collection="nemo_nlp"
)
t_loss_eval = neural_factory.get_module(
    name="PaddedSmoothedCrossEntropyLossNM",
    params={
        "pad_id": tokenizer.pad_id(),
        "label_smoothing": 0
    },
    collection="nemo_nlp"
)

t_log_softmax.log_softmax.dense.weight = \
    t_encoder.embedding_layer.token_embedding.weight
t_decoder.embedding_layer.token_embedding.weight = \
    t_encoder.embedding_layer.token_embedding.weight

# training pipeline
src, src_mask, tgt, tgt_mask, labels, sent_ids = train_data_layer()
src_hiddens = t_encoder(input_ids=src, input_mask_src=src_mask)
tgt_hiddens = t_decoder(input_ids_tgt=tgt,
                        hidden_states_src=src_hiddens,
                        input_mask_src=src_mask,
                        input_mask_tgt=tgt_mask)
log_softmax = t_log_softmax(hidden_states=tgt_hiddens)
train_loss = t_loss(log_probs=log_softmax, target_ids=labels)

# evaluation pipeline with beam search on top of the model output
src_, src_mask_, tgt_, tgt_mask_, labels_, sent_ids_ = eval_data_layer()
src_hiddens_ = t_encoder(input_ids=src_, input_mask_src=src_mask_)
tgt_hiddens_ = t_decoder(input_ids_tgt=tgt_,
                         hidden_states_src=src_hiddens_,
                         input_mask_src=src_mask_,
                         input_mask_tgt=tgt_mask_)
log_softmax_ = t_log_softmax(hidden_states=tgt_hiddens_)
eval_loss = t_loss_eval(log_probs=log_softmax_, target_ids=labels_)
beam_trans = beam_translator(
    hidden_states_src=src_hiddens_, input_mask_src=src_mask_)

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
        x, validation_dataset=eval_dataset_tgt),
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
    tensors_to_evaluate=[],
    lr_policy=lr_policy,
    callbacks=[callback, callback_dev],
    batches_per_step=args.batch_accumulation_steps
)
