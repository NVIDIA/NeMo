# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
See the tutorial and download the data here:
https://nvidia.github.io/NeMo/nlp/
neural-machine-translation.html#translation-with-pretrained-model
"""
import torch

import nemo
import nemo.collections.nlp as nemo_nlp
from nemo import logging
from nemo.collections.nlp.callbacks.machine_translation_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.core import WeightShareTransform
from nemo.utils.lr_policies import get_lr_policy

parser = nemo.utils.NemoArgParser(description='Transformer for Neural Machine Translation')
parser.set_defaults(
    train_dataset="train",
    eval_datasets=["valid"],
    work_dir="outputs/transformer_nmt",
    optimizer="novograd",
    batch_size=4096,
    eval_batch_size=4096,
    lr_policy='CosineAnnealing',
    lr=0.005,
    weight_decay=0,
    max_steps=500,
    iter_per_step=1,
    eval_freq=1000,
)
parser.add_argument("--data_dir", default="../../../tests/data/en_de", type=str)
parser.add_argument("--dataset_name", default="wmt16", type=str)
parser.add_argument("--src_lang", default="en", type=str)
parser.add_argument("--tgt_lang", default="de", choices=['de', 'zh'], type=str)
parser.add_argument("--d_model", default=512, type=int)
parser.add_argument("--d_inner", default=2048, type=int)
parser.add_argument("--num_layers", default=4, type=int)
parser.add_argument("--num_attn_heads", default=8, type=int)
parser.add_argument("--embedding_dropout", default=0.2, type=float)
parser.add_argument("--ffn_dropout", default=0.2, type=float)
parser.add_argument("--attn_score_dropout", default=0.2, type=float)
parser.add_argument("--attn_layer_dropout", default=0.2, type=float)
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--label_smoothing", default=0.1, type=float)
parser.add_argument("--beam_size", default=4, type=int)
# pass a YouTokenToMe model to YouTokenToMeTokenizer for en
parser.add_argument("--src_tokenizer_model", default="bpe8k_yttm.model", type=str)
# pass a YouTokenToMe model to YouTokenToMeTokenizer for de
# if the target is zh, we should pass a vocabulary file, e.g. zh_vocab.txt
parser.add_argument("--tgt_tokenizer_model", default="bpe8k_yttm.model", type=str)
parser.add_argument("--interactive", action="store_true")
parser.add_argument("--save_epoch_freq", default=5, type=int)
parser.add_argument("--save_step_freq", default=-1, type=int)
parser.add_argument("--restore_checkpoint_from", default=None, type=str)

args = parser.parse_args()

work_dir = f'{args.work_dir}/{args.dataset_name.upper()}'
nf = nemo.core.NeuralModuleFactory(
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=args.work_dir,
    create_tb_writer=True,
    files_to_copy=[__file__],
)

# tie weight of embedding and log_softmax layers if use the same tokenizer
# for the source and the target
tie_weight = False

"""Define tokenizer
When the src and tgt languages are very different, it's better to use separate
tokenizers.
"""
if args.src_lang == 'en' and args.tgt_lang == 'de':
    """
    We use YouTokenToMe tokenizer trained on joint
    English & German data for both source and target languages.
    """
    src_tokenizer = nemo_nlp.data.YouTokenToMeTokenizer(model_path=f"{args.data_dir}/{args.src_tokenizer_model}")
    src_vocab_size = src_tokenizer.vocab_size
    if args.src_tokenizer_model == args.tgt_tokenizer_model:
        tgt_tokenizer = src_tokenizer
        # source and target use the same tokenizer, set tie_weight to True
        tie_weight = True
    else:
        tgt_tokenizer = nemo_nlp.data.YouTokenToMeTokenizer(model_path=f"{args.data_dir}/{args.tgt_tokenizer_model}")
        # source and target use different tokenizers, set tie_weight to False
        tie_weight = False
    tgt_vocab_size = tgt_tokenizer.vocab_size
elif args.src_lang == 'en' and args.tgt_lang == 'zh':
    """
    We use YouTokenToMeTokenizer for src since the src contains English words
    and CharTokenizer for tgt since the tgt contains Chinese characters.
    """
    src_tokenizer = nemo_nlp.data.YouTokenToMeTokenizer(model_path=f"{args.data_dir}/{args.src_tokenizer_model}")
    src_vocab_size = src_tokenizer.vocab_size
    tgt_tokenizer = nemo_nlp.data.CharTokenizer(vocab_path=f"{args.data_dir}/{args.tgt_tokenizer_model}")
    tgt_vocab_size = tgt_tokenizer.vocab_size
    # source and target use different tokenizers, set tie_weight to False
    tie_weight = False
else:
    raise ValueError(f"Unsupported language pair:{args.src_lang}-{args.tgt_lang}.")

# instantiate necessary modules for the whole translation pipeline, namely
# data layers, encoder, decoder, output log_softmax, beam_search_translator
# and loss function
encoder = nemo_nlp.nm.trainables.TransformerEncoderNM(
    d_model=args.d_model,
    d_inner=args.d_inner,
    num_layers=args.num_layers,
    embedding_dropout=args.embedding_dropout,
    num_attn_heads=args.num_attn_heads,
    ffn_dropout=args.ffn_dropout,
    vocab_size=src_vocab_size,
    attn_score_dropout=args.attn_score_dropout,
    attn_layer_dropout=args.attn_layer_dropout,
    max_seq_length=args.max_seq_length,
)

decoder = nemo_nlp.nm.trainables.TransformerDecoderNM(
    d_model=args.d_model,
    d_inner=args.d_inner,
    num_layers=args.num_layers,
    embedding_dropout=args.embedding_dropout,
    num_attn_heads=args.num_attn_heads,
    ffn_dropout=args.ffn_dropout,
    vocab_size=tgt_vocab_size,
    attn_score_dropout=args.attn_score_dropout,
    attn_layer_dropout=args.attn_layer_dropout,
    max_seq_length=args.max_seq_length,
)

log_softmax = nemo_nlp.nm.trainables.TokenClassifier(
    args.d_model, num_classes=tgt_tokenizer.vocab_size, num_layers=1, log_softmax=True
)

beam_search = nemo_nlp.nm.trainables.BeamSearchTranslatorNM(
    decoder=decoder,
    log_softmax=log_softmax,
    max_seq_length=args.max_seq_length,
    beam_size=args.beam_size,
    bos_token=tgt_tokenizer.bos_id,
    pad_token=tgt_tokenizer.pad_id,
    eos_token=tgt_tokenizer.eos_id,
)

loss_fn = nemo_nlp.nm.losses.SmoothedCrossEntropyLoss(
    pad_id=tgt_tokenizer.pad_id, label_smoothing=args.label_smoothing
)

if tie_weight:
    # log_softmax.mlp.last_linear_layer.weight = encoder.embedding_layer.token_embedding.weight
    log_softmax.tie_weights_with(
        encoder,
        weight_names=["mlp.last_linear_layer.weight"],
        name2name_and_transform={
            "mlp.last_linear_layer.weight": ("embedding_layer.token_embedding.weight", WeightShareTransform.SAME)
        },
    )
    # decoder.embedding_layer.token_embedding.weight = encoder.embedding_layer.token_embedding.weight
    decoder.tie_weights_with(
        encoder,
        weight_names=["embedding_layer.token_embedding.weight"],
        name2name_and_transform={
            "embedding_layer.token_embedding.weight": (
                "embedding_layer.token_embedding.weight",
                WeightShareTransform.SAME,
            )
        },
    )


def create_pipeline(dataset_src, dataset_tgt, tokens_in_batch, clean=False, training=True):
    data_layer = nemo_nlp.nm.data_layers.machine_translation_datalayer.TranslationDataLayer(
        tokenizer_src=src_tokenizer,
        tokenizer_tgt=tgt_tokenizer,
        dataset_src=dataset_src,
        dataset_tgt=dataset_tgt,
        tokens_in_batch=tokens_in_batch,
        clean=clean,
    )
    src, src_mask, tgt, tgt_mask, labels, sent_ids = data_layer()
    src_hiddens = encoder(input_ids=src, input_mask_src=src_mask)
    tgt_hiddens = decoder(
        input_ids_tgt=tgt, hidden_states_src=src_hiddens, input_mask_src=src_mask, input_mask_tgt=tgt_mask
    )
    logits = log_softmax(hidden_states=tgt_hiddens)
    loss = loss_fn(logits=logits, labels=labels)
    beam_results = None
    if not training:
        beam_results = beam_search(hidden_states_src=src_hiddens, input_mask_src=src_mask)
    return loss, [tgt, loss, beam_results, sent_ids]


train_dataset_src = f"{args.data_dir}/{args.train_dataset}.{args.src_lang}"
train_dataset_tgt = f"{args.data_dir}/{args.train_dataset}.{args.tgt_lang}"

train_loss, _ = create_pipeline(train_dataset_src, train_dataset_tgt, args.batch_size, clean=True)

eval_dataset_src = f"{args.data_dir}/{args.eval_datasets[0]}.{args.src_lang}"
eval_dataset_tgt = f"{args.data_dir}/{args.eval_datasets[0]}.{args.tgt_lang}"

eval_loss, eval_tensors = create_pipeline(eval_dataset_src, eval_dataset_tgt, args.eval_batch_size, training=False)

# callback which prints training loss once in a while
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    step_freq=100,
    print_func=lambda x: logging.info(str(x[0].item())),
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=nf.tb_writer,
)

# callback which calculates evaluation loss and both common BLEU and SacreBLEU
# scores between outputs of beam search and reference translations
eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=eval_tensors,
    user_iter_callback=lambda x, y: eval_iter_callback(x, y, tgt_tokenizer),
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(x, validation_dataset=eval_dataset_tgt),
    eval_step=args.eval_freq,
    tb_writer=nf.tb_writer,
)

# callback which saves checkpoints once in a while
ckpt_dir = nf.checkpoint_dir if not args.interactive else args.restore_checkpoint_from
ckpt_callback = nemo.core.CheckpointCallback(
    folder=ckpt_dir, epoch_freq=args.save_epoch_freq, step_freq=args.save_step_freq, checkpoints_to_keep=1
)

# define learning rate decay policy
lr_policy_fn = get_lr_policy(args.lr_policy, total_steps=args.max_steps, warmup_steps=args.warmup_steps)

if args.max_steps is not None and args.num_epochs is not None:
    raise ValueError("Please specify either max_steps or num_epochs.")

if not args.interactive:

    if args.max_steps is not None:
        stop_training_condition = {"max_steps": args.max_steps}
    else:
        stop_training_condition = {"num_epochs": args.num_epochs}

    nf.train(
        tensors_to_optimize=[train_loss],
        callbacks=[train_callback, eval_callback, ckpt_callback],
        optimizer=args.optimizer,
        lr_policy=lr_policy_fn,
        optimization_params={**stop_training_condition, "lr": args.lr, "weight_decay": args.weight_decay},
        batches_per_step=args.iter_per_step,
    )
else:
    nf.train(
        tensors_to_optimize=[train_loss],
        callbacks=[ckpt_callback],
        optimizer=args.optimizer,
        optimization_params={"num_epochs": 0, "lr": args.lr},
    )


def translate_sentence(text):
    """ helper function that takes source text as input, tokenizes it, and
    output its translation
    """
    ids = src_tokenizer.text_to_ids(text)
    ids = [src_tokenizer.bos_id] + ids + [src_tokenizer.eos_id]
    ids_tensor = torch.Tensor(ids).long().to(encoder._device).unsqueeze(0)
    ids_mask = torch.ones_like(ids_tensor)
    encoder_states = encoder.forward(ids_tensor, ids_mask)
    if args.amp_opt_level in ["O1", "O2", "O3"]:
        encoder_states = encoder_states.half()
    translation_ids = beam_search.forward(encoder_states, ids_mask)
    ids_list = list(translation_ids.detach().cpu().numpy()[0])
    translation_text = tgt_tokenizer.ids_to_text(ids_list)
    return translation_text


# code below was not covered in the tutorial, it is used to generate
# translations on the fly with pre-trained model
if args.interactive:
    # set all modules into evaluation mode (turn off dropout)
    encoder.eval()
    decoder.eval()
    log_softmax.eval()
    print("========== Interactive translation mode ==========")
    input_text = 'anything'
    while input_text.strip():
        input_text = input(f'Text in {args.src_lang} to be translated: ')
        print('Translated:', translate_sentence(input_text.strip()))
