# Copyright (c) 2019 NVIDIA Corporation
import nemo
import nemo_nlp
import math
import logging
from nemo.utils.lr_policies import CosineAnnealing
from nemo_nlp.callbacks.language_modeling import eval_iter_callback, \
    eval_epochs_done_callback

console = logging.StreamHandler()
console.setLevel(logging.INFO)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

parser = nemo.utils.NemoArgParser(description='Transformer for WT-103 LM')
parser.set_defaults(
    train_dataset="train.txt",
    eval_datasets=["valid.txt"],
    checkpoint_dir="transformer_lm",
    optimizer="novograd",
    num_epochs=1000,
    batch_size=32,
    eval_batch_size=32,
    lr=0.002,
    beta1=0.95,
    beta2=0.25,
    weight_decay=0,
    warmup_steps=1000,
    max_steps=50000,
    iter_per_step=1,
    checkpoint_save_freq=10000,
    eval_freq=1000
)
parser.add_argument("--data_root", default="../../tests/data/wikitext-2/",
                    type=str)
parser.add_argument("--d_model", default=384, type=int)
parser.add_argument("--d_inner", default=1536, type=int)
parser.add_argument("--num_layers", default=12, type=int)
parser.add_argument("--num_attn_heads", default=6, type=int)
parser.add_argument("--embedding_dropout", default=0.2, type=float)
parser.add_argument("--fully_connected_dropout", default=0.2, type=float)
parser.add_argument("--attn_score_dropout", default=0.2, type=float)
parser.add_argument("--attn_layer_dropout", default=0.2, type=float)
parser.add_argument("--max_sequence_length", default=256, type=int)
parser.add_argument("--label_smoothing", default=0.1, type=float)
parser.add_argument("--beam_size", default=4, type=int)
parser.add_argument("--tokenizer_model", default="vocab.txt", type=str)
parser.add_argument("--predict_last_k", default=16, type=int)
parser.add_argument("--interactive", action="store_true")
args = parser.parse_args()

# create TensorboardX logger to log training statistics
name = f"transformer-lm-lr_{args.lr}-optim_{args.optimizer}-" \
    f"warmup_{args.warmup_steps}-bs_{args.batch_size}"
tb_writer = None  # SummaryWriter(name)

# instantiate Neural Factory with supported backend
device = nemo.core.DeviceType.AllGpu if args.local_rank is not None \
    else nemo.core.DeviceType.GPU
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=nemo.core.Optimization.mxprO2,
    placement=device)

# define tokenizer, in this example we use word-level tokenizer
# we also adjust the vocabulary size to make it multiple of 8 to accelerate
# training in fp16 mode with the use of Tensor Cores
tokenizer = nemo_nlp.WordTokenizer(f"{args.data_root}/{args.tokenizer_model}")
vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)

# instantiate necessary modules for the whole translation pipeline, namely
# data layers, encoder, decoder, output log_softmax, beam_search_translator
# and loss function
train_data_layer = nemo_nlp.LanguageModelingDataLayer(
    factory=neural_factory,
    tokenizer=tokenizer,
    dataset=f"{args.data_root}/{args.train_dataset}",
    max_seq_length=args.max_sequence_length,
    batch_size=args.batch_size,
    batch_step=args.max_sequence_length)
eval_data_layer = nemo_nlp.LanguageModelingDataLayer(
    factory=neural_factory,
    tokenizer=tokenizer,
    dataset=f"{args.data_root}/{args.eval_datasets[0]}",
    max_seq_length=args.max_sequence_length,
    batch_size=args.batch_size,
    batch_step=args.predict_last_k)
encoder = nemo_nlp.TransformerEncoderNM(
    factory=neural_factory,
    d_model=args.d_model,
    d_inner=args.d_inner,
    num_layers=args.num_layers,
    num_attn_heads=args.num_attn_heads,
    fully_connected_dropout=args.fully_connected_dropout,
    vocab_size=vocab_size,
    mask_future=True,
    attn_score_dropout=args.attn_score_dropout,
    attn_layer_dropout=args.attn_layer_dropout,
    max_seq_length=args.max_sequence_length)
log_softmax = nemo_nlp.TransformerLogSoftmaxNM(
    factory=neural_factory,
    vocab_size=vocab_size,
    d_model=args.d_model)
loss = nemo_nlp.PaddedSmoothedCrossEntropyLossNM(
    factory=neural_factory,
    pad_id=tokenizer.pad_id(),
    label_smoothing=args.label_smoothing)

# tie weight of embedding and log_softmax layers
log_softmax.log_softmax.dense.weight = \
    encoder.embedding_layer.token_embedding.weight

# training pipeline
src, src_mask, labels = train_data_layer()
src_hiddens = encoder(input_ids=src, input_mask_src=src_mask)
log_probs = log_softmax(hidden_states=src_hiddens)
train_loss = loss(log_probs=log_probs, target_ids=labels)

# evaluation pipeline
src_, src_mask_, labels_ = eval_data_layer()
src_hiddens_ = encoder(input_ids=src_, input_mask_src=src_mask_)
log_probs_ = log_softmax(hidden_states=src_hiddens_)
eval_loss = loss(log_probs=log_probs_, target_ids=labels_)


def print_loss(x):
    loss = str(x[0].item())
    print(f"Training loss: {loss}")


# callback which prints training loss once in a while
callback_train = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    step_freq=100,
    print_func=print_loss,
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=tb_writer)

# callback which calculates evaluation loss
callback_eval = nemo.core.EvaluatorCallback(
    eval_tensors=[eval_loss],
    user_iter_callback=eval_iter_callback,
    user_epochs_done_callback=eval_epochs_done_callback,
    eval_step=args.eval_freq,
    tb_writer=tb_writer)

# callback which saves checkpoints once in a while
callback_ckpt = nemo.core.CheckpointCallback(
    folder=args.checkpoint_dir,
    step_freq=args.checkpoint_save_freq,
    checkpoints_to_keep=-1)

# define learning rate decay policy
lr_policy = CosineAnnealing(args.max_steps, warmup_steps=args.warmup_steps)

# define and launch training algorithm (optimizer)
max_num_epochs = 0 if args.interactive else args.num_epochs

optimizer = neural_factory.get_trainer()

callbacks = [callback_ckpt]
if not args.interactive:
    callbacks.extend([callback_train, callback_eval])
optimizer.train(
    tensors_to_optimize=[train_loss],
    callbacks=callbacks,
    optimizer=args.optimizer,
    lr_policy=lr_policy,
    optimization_params={
        "num_epochs": max_num_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": (args.beta1, args.beta2)
    },
    batches_per_step=args.iter_per_step
)
