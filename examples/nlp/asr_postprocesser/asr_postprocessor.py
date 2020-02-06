# =============================================================================
# Copyright 2019 AI Applications Design Team at NVIDIA. All Rights Reserved.
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

import math
import os

import torch

import nemo.collections.nlp as nemo_nlp
import nemo.collections.nlp.nm.data_layers.machine_translation_datalayer
from nemo import logging
from nemo.collections.nlp.callbacks.machine_translation_callback import (
    eval_epochs_done_callback_wer,
    eval_iter_callback,
)
from nemo.collections.nlp.data.tokenizers.bert_tokenizer import NemoBertTokenizer
from nemo.core.callbacks import CheckpointCallback
from nemo.utils.lr_policies import SquareAnnealing

parser = nemo.utils.NemoArgParser(description='ASR postprocessor')
parser.set_defaults(
    train_dataset="train",
    eval_datasets=["valid"],
    optimizer="novograd",
    amp_opt_level="O1",
    num_epochs=1000,
    batch_size=4096,
    eval_batch_size=1024,
    lr=0.001,
    weight_decay=0,
    max_steps=2000,
    iter_per_step=1,
    checkpoint_save_freq=10000,
    work_dir='outputs/asr_postprocessor',
    eval_freq=200,
)


parser.add_argument("--pretrained_model", default="bert-base-uncased", type=str)
parser.add_argument("--warmup_steps", default=2000, type=int)
parser.add_argument("--d_model", default=768, type=int)
parser.add_argument("--d_inner", default=3072, type=int)
parser.add_argument("--num_layers", default=12, type=int)
parser.add_argument("--num_heads", default=12, type=int)
parser.add_argument("--embedding_dropout", default=0.25, type=float)
parser.add_argument("--max_seq_length", default=512, type=int)
parser.add_argument("--ffn_dropout", default=0.25, type=float)
parser.add_argument("--attn_score_dropout", default=0.25, type=float)
parser.add_argument("--attn_layer_dropout", default=0.25, type=float)
parser.add_argument("--eval_step_frequency", default=2000, type=int)
parser.add_argument("--data_dir", default="/dataset/", type=str)
parser.add_argument("--src_lang", default="pred", type=str)
parser.add_argument("--tgt_lang", default="real", type=str)
parser.add_argument("--beam_size", default=4, type=int)
parser.add_argument("--len_pen", default=0.0, type=float)
parser.add_argument(
    "--restore_from", dest="restore_from", type=str, default="../../scripts/bert-base-uncased_decoder.pt"
)
args = parser.parse_args()

nf = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=args.work_dir,
    create_tb_writer=False,
    files_to_copy=[__file__],
    add_time_to_log_dir=False,
)

tokenizer = NemoBertTokenizer(pretrained_model=args.pretrained_model)
vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)
tokens_to_add = vocab_size - tokenizer.vocab_size

zeros_transform = nemo.backends.pytorch.common.ZerosLikeNM()
encoder = nemo_nlp.nm.trainables.huggingface.BERT(pretrained_model_name=args.pretrained_model)
device = encoder.bert.embeddings.word_embeddings.weight.get_device()
zeros = torch.zeros((tokens_to_add, args.d_model)).to(device=device)
encoder.bert.embeddings.word_embeddings.weight.data = torch.cat(
    (encoder.bert.embeddings.word_embeddings.weight.data, zeros)
)

decoder = nemo_nlp.nm.trainables.TransformerDecoderNM(
    d_model=args.d_model,
    d_inner=args.d_inner,
    num_layers=args.num_layers,
    num_attn_heads=args.num_heads,
    ffn_dropout=args.ffn_dropout,
    vocab_size=vocab_size,
    attn_score_dropout=args.attn_score_dropout,
    attn_layer_dropout=args.attn_layer_dropout,
    max_seq_length=args.max_seq_length,
    embedding_dropout=args.embedding_dropout,
    learn_positional_encodings=True,
    hidden_act="gelu",
)

decoder.restore_from(args.restore_from, local_rank=args.local_rank)

t_log_softmax = nemo_nlp.nm.trainables.TokenClassifier(
    args.d_model, num_classes=vocab_size, num_layers=1, log_softmax=True
)

loss_fn = nemo_nlp.nm.losses.PaddedSmoothedCrossEntropyLossNM(pad_id=tokenizer.pad_id(), label_smoothing=0.1)

beam_search = nemo_nlp.nm.trainables.BeamSearchTranslatorNM(
    decoder=decoder,
    log_softmax=t_log_softmax,
    max_seq_length=args.max_seq_length,
    beam_size=args.beam_size,
    length_penalty=args.len_pen,
    bos_token=tokenizer.bos_id(),
    pad_token=tokenizer.pad_id(),
    eos_token=tokenizer.eos_id(),
)

# tie all embeddings weights
t_log_softmax.mlp.layer0.weight = encoder.bert.embeddings.word_embeddings.weight
decoder.embedding_layer.token_embedding.weight = encoder.bert.embeddings.word_embeddings.weight
decoder.embedding_layer.position_embedding.weight = encoder.bert.embeddings.position_embeddings.weight


def create_pipeline(dataset, tokens_in_batch, clean=False, training=True):
    dataset_src = os.path.join(args.data_dir, dataset + "." + args.src_lang)
    dataset_tgt = os.path.join(args.data_dir, dataset + "." + args.tgt_lang)
    data_layer = nemo_nlp.nm.data_layers.machine_translation_datalayer.TranslationDataLayer(
        tokenizer_src=tokenizer,
        tokenizer_tgt=tokenizer,
        dataset_src=dataset_src,
        dataset_tgt=dataset_tgt,
        tokens_in_batch=tokens_in_batch,
        clean=clean,
    )
    src, src_mask, tgt, tgt_mask, labels, sent_ids = data_layer()
    input_type_ids = zeros_transform(input_type_ids=src)
    src_hiddens = encoder(input_ids=src, token_type_ids=input_type_ids, attention_mask=src_mask)
    tgt_hiddens = decoder(
        input_ids_tgt=tgt, hidden_states_src=src_hiddens, input_mask_src=src_mask, input_mask_tgt=tgt_mask
    )
    log_softmax = t_log_softmax(hidden_states=tgt_hiddens)
    loss = loss_fn(logits=log_softmax, target_ids=labels)
    beam_results = None
    if not training:
        beam_results = beam_search(hidden_states_src=src_hiddens, input_mask_src=src_mask)
    return loss, [tgt, loss, beam_results, sent_ids]


# training pipeline
train_loss, _ = create_pipeline(args.train_dataset, args.batch_size, clean=False)

# evaluation pipelines
all_eval_losses = {}
all_eval_tensors = {}
for eval_dataset in args.eval_datasets:
    eval_loss, eval_tensors = create_pipeline(eval_dataset, args.eval_batch_size, clean=False, training=False)
    all_eval_losses[eval_dataset] = eval_loss
    all_eval_tensors[eval_dataset] = eval_tensors


def print_loss(x):
    loss = x[0].item()
    logging.info("Training loss: {:.4f}".format(loss))


# callbacks
callback_train = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    step_freq=100,
    print_func=print_loss,
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=nf.tb_writer,
)

callbacks = [callback_train]

for eval_dataset in args.eval_datasets:
    callback = nemo.core.EvaluatorCallback(
        eval_tensors=all_eval_tensors[eval_dataset],
        user_iter_callback=lambda x, y: eval_iter_callback(x, y, tokenizer),
        user_epochs_done_callback=eval_epochs_done_callback_wer,
        eval_step=args.eval_freq,
        tb_writer=nf.tb_writer,
    )
    callbacks.append(callback)

checkpointer_callback = CheckpointCallback(folder=args.work_dir, step_freq=args.checkpoint_save_freq)
callbacks.append(checkpointer_callback)

# define learning rate decay policy
lr_policy = SquareAnnealing(total_steps=args.max_steps, min_lr=1e-5, warmup_steps=args.warmup_steps)

# Create trainer and execute training action
nf.train(
    tensors_to_optimize=[train_loss],
    callbacks=callbacks,
    optimizer=args.optimizer,
    lr_policy=lr_policy,
    optimization_params={"num_epochs": 300, "max_steps": args.max_steps, "lr": args.lr, "weight_decay": args.weight_decay},
    batches_per_step=args.iter_per_step,
)
