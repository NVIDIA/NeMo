# Copyright (c) 2019 NVIDIA Corporation
import nemo
import nemo_nlp
import logging
from nemo.utils.lr_policies import CosineAnnealing
from nemo_nlp.callbacks.translation import eval_iter_callback, \
    eval_epochs_done_callback

console = logging.StreamHandler()
console.setLevel(logging.INFO)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

parser = nemo.utils.NemoArgParser(description='Transformer for WMT16 En-De')
parser.set_defaults(
    train_dataset="train",
    eval_datasets=["valid"],
    checkpoint_dir="transformer_nmt",
    optimizer="novograd",
    num_epochs=1000,
    batch_size=4096,
    eval_batch_size=4096,
    lr=0.005,
    beta1=0.95,
    beta2=0.25,
    weight_decay=0,
    max_steps=50000,
    iter_per_step=1,
    checkpoint_save_freq=10000,
    eval_freq=1000
)
parser.add_argument("--data_root", default="../../tests/data/en_de/", type=str)
parser.add_argument("--src_lang", default="en", type=str)
parser.add_argument("--tgt_lang", default="de", type=str)
parser.add_argument("--d_model", default=512, type=int)
parser.add_argument("--d_inner", default=2048, type=int)
parser.add_argument("--num_layers", default=4, type=int)
parser.add_argument("--num_attn_heads", default=8, type=int)
parser.add_argument("--embedding_dropout", default=0.2, type=float)
parser.add_argument("--ffn_dropout", default=0.2, type=float)
parser.add_argument("--attn_score_dropout", default=0.2, type=float)
parser.add_argument("--attn_layer_dropout", default=0.2, type=float)
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--max_sequence_length", default=256, type=int)
parser.add_argument("--label_smoothing", default=0.1, type=float)
parser.add_argument("--beam_size", default=4, type=int)
parser.add_argument("--tokenizer_model", default="bpe8k_yttm.model", type=str)
parser.add_argument("--interactive", action="store_true")
args = parser.parse_args()

# create TensorboardX logger to log training statistics
name = f"transformer-nmt_{args.src_lang}_{args.tgt_lang}-lr_{args.lr}-" \
    f"optim_{args.optimizer}-warmup_{args.warmup_steps}-bs_{args.batch_size}"
tb_writer = None  # SummaryWriter(name)

# instantiate Neural Factory with supported backend
device = nemo.core.DeviceType.AllGpu if args.local_rank is not None \
    else nemo.core.DeviceType.GPU
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=nemo.core.Optimization.mxprO2,
    placement=device)

# define tokenizer, in this example we use YouTokenToMe tokenizer trained
# on joint English & German data for both source and target languages
tokenizer = nemo_nlp.YouTokenToMeTokenizer(
    model_path=f"{args.data_root}/{args.tokenizer_model}")
vocab_size = tokenizer.vocab_size

# instantiate necessary modules for the whole translation pipeline, namely
# data layers, encoder, decoder, output log_softmax, beam_search_translator
# and loss function
train_data_layer = nemo_nlp.TranslationDataLayer(
    factory=neural_factory,
    tokenizer_src=tokenizer,
    tokenizer_tgt=tokenizer,
    dataset_src=f"{args.data_root}/{args.train_dataset}.{args.src_lang}",
    dataset_tgt=f"{args.data_root}/{args.train_dataset}.{args.tgt_lang}",
    tokens_in_batch=args.batch_size,
    clean=True)
eval_data_layer = nemo_nlp.TranslationDataLayer(
    factory=neural_factory,
    tokenizer_src=tokenizer,
    tokenizer_tgt=tokenizer,
    dataset_src=f"{args.data_root}/{args.eval_datasets[0]}.{args.src_lang}",
    dataset_tgt=f"{args.data_root}/{args.eval_datasets[0]}.{args.tgt_lang}",
    tokens_in_batch=args.eval_batch_size)
encoder = nemo_nlp.TransformerEncoderNM(
    factory=neural_factory,
    d_model=args.d_model,
    d_inner=args.d_inner,
    num_layers=args.num_layers,
    embedding_dropout=args.embedding_dropout,
    num_attn_heads=args.num_attn_heads,
    ffn_dropout=args.ffn_dropout,
    vocab_size=vocab_size,
    attn_score_dropout=args.attn_score_dropout,
    attn_layer_dropout=args.attn_layer_dropout,
    max_seq_length=args.max_sequence_length)
decoder = nemo_nlp.TransformerDecoderNM(
    factory=neural_factory,
    d_model=args.d_model,
    d_inner=args.d_inner,
    num_layers=args.num_layers,
    embedding_dropout=args.embedding_dropout,
    num_attn_heads=args.num_attn_heads,
    ffn_dropout=args.ffn_dropout,
    vocab_size=vocab_size,
    attn_score_dropout=args.attn_score_dropout,
    attn_layer_dropout=args.attn_layer_dropout,
    max_seq_length=args.max_sequence_length)
log_softmax = nemo_nlp.TransformerLogSoftmaxNM(
    factory=neural_factory,
    vocab_size=vocab_size,
    d_model=args.d_model)
beam_search = nemo_nlp.BeamSearchTranslatorNM(
    factory=neural_factory,
    decoder=decoder,
    log_softmax=log_softmax,
    max_seq_length=args.max_sequence_length,
    beam_size=args.beam_size,
    bos_token=tokenizer.bos_id(),
    pad_token=tokenizer.pad_id(),
    eos_token=tokenizer.eos_id())
loss = nemo_nlp.PaddedSmoothedCrossEntropyLossNM(
    factory=neural_factory,
    pad_id=tokenizer.pad_id(),
    label_smoothing=args.label_smoothing)

# tie weight of embedding and log_softmax layers
log_softmax.log_softmax.dense.weight = \
    encoder.embedding_layer.token_embedding.weight
decoder.embedding_layer.token_embedding.weight = \
    encoder.embedding_layer.token_embedding.weight

# training pipeline
src, src_mask, tgt, tgt_mask, labels, sent_ids = train_data_layer()
src_hiddens = encoder(input_ids=src, input_mask_src=src_mask)
tgt_hiddens = decoder(input_ids_tgt=tgt,
                      hidden_states_src=src_hiddens,
                      input_mask_src=src_mask,
                      input_mask_tgt=tgt_mask)
log_probs = log_softmax(hidden_states=tgt_hiddens)
train_loss = loss(log_probs=log_probs, target_ids=labels)

# evaluation pipeline
src_, src_mask_, tgt_, tgt_mask_, labels_, sent_ids_ = eval_data_layer()
src_hiddens_ = encoder(input_ids=src_, input_mask_src=src_mask_)
tgt_hiddens_ = decoder(input_ids_tgt=tgt_,
                       hidden_states_src=src_hiddens_,
                       input_mask_src=src_mask_,
                       input_mask_tgt=tgt_mask_)
log_probs_ = log_softmax(hidden_states=tgt_hiddens_)
eval_loss = loss(log_probs=log_probs_, target_ids=labels_)
beam_trans = beam_search(hidden_states_src=src_hiddens_,
                         input_mask_src=src_mask_)


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

# callback which calculates evaluation loss and both common BLEU and SacreBLEU
# scores between outputs of beam search and reference translations
valid_dataset = f"{args.data_root}/{args.eval_datasets[0]}.{args.tgt_lang}"
callback_eval = nemo.core.EvaluatorCallback(
    eval_tensors=[tgt_, eval_loss, beam_trans, sent_ids_],
    user_iter_callback=lambda x, y: eval_iter_callback(x, y, tokenizer),
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(
        x, validation_dataset=valid_dataset),
    eval_step=args.eval_freq,
    tb_writer=tb_writer)

# callback which saves checkpoints once in a while
callback_ckpt = nemo.core.CheckpointCallback(
    folder=args.checkpoint_dir,
    step_freq=args.checkpoint_save_freq,
    checkpoints_to_keep=1)

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

# code below was not covered in the tutorial, it is used to generate
# translations on the fly with pre-trained model
if args.interactive:

    import torch

    # set all modules into evaluation mode (turn off dropout)
    encoder.eval()
    decoder.eval()
    log_softmax.eval()

    # helper function which takes text, tokenizes it, and translates
    def translate_sentence(text):
        ids = tokenizer.text_to_ids(text)
        ids = [tokenizer.bos_id()] + ids + [tokenizer.eos_id()]
        ids_tensor = torch.Tensor(ids).long().to(encoder._device).unsqueeze(0)
        ids_mask = torch.ones_like(ids_tensor).half()
        encoder_states = encoder.forward(ids_tensor, ids_mask).half()
        translation_ids = beam_search.forward(encoder_states, ids_mask)
        ids_list = list(translation_ids.detach().cpu().numpy()[0])
        translation_text = tokenizer.ids_to_text(ids_list)
        return translation_text

    print()
    print("========== Interactive translation mode ==========")
    while True:
        print("Type text to translate, type STOP to exit.", "\n")
        input_text = input()
        if input_text == "STOP":
            print("============ Exiting translation mode ============")
            break
        print(translate_sentence(input_text), "\n")
