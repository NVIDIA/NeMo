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
import math

import nemo
import nemo.collections.nlp as nemo_nlp
from nemo import logging
from nemo.collections.nlp.callbacks.lm_transformer_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.collections.nlp.data.datasets.lm_transformer_dataset import LanguageModelDataDesc
from nemo.collections.nlp.nm.data_layers import LanguageModelingDataLayer
from nemo.collections.nlp.nm.losses import SmoothedCrossEntropyLoss
from nemo.collections.nlp.nm.trainables.common import TokenClassifier
from nemo.core import WeightShareTransform
from nemo.utils.lr_policies import CosineAnnealing

parser = nemo.utils.NemoArgParser(description='LM Transformer')
parser.set_defaults(
    train_dataset="train.txt",
    eval_dataset="valid.txt",
    work_dir="outputs/transformer_lm",
    optimizer_kind="novograd",
    amp_opt_level='O1',
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
    eval_freq=1000,
)
parser.add_argument("--data_dir", default="data/lm/wikitext-2", type=str)
parser.add_argument("--dataset_name", default="wikitext-2", type=str)
parser.add_argument("--d_model", default=384, type=int)
parser.add_argument("--d_inner", default=1536, type=int)
parser.add_argument("--num_layers", default=12, type=int)
parser.add_argument("--num_attn_heads", default=6, type=int)
parser.add_argument("--embedding_dropout", default=0.2, type=float)
parser.add_argument("--ffn_dropout", default=0.2, type=float)
parser.add_argument("--attn_score_dropout", default=0.2, type=float)
parser.add_argument("--attn_layer_dropout", default=0.2, type=float)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--do_lower_case", action='store_true')
parser.add_argument("--label_smoothing", default=0.1, type=float)
parser.add_argument("--beam_size", default=4, type=int)
parser.add_argument("--tokenizer_model", default="vocab.txt", type=str)
parser.add_argument("--predict_last_k", default=16, type=int)
parser.add_argument("--save_epoch_freq", default=1, type=int)
parser.add_argument("--save_step_freq", default=-1, type=int)
parser.add_argument("--interactive", action="store_true")
args = parser.parse_args()

"""
To get the data, go to tests/data and run get_wt2.sh
Then run create_vocab.py
"""

work_dir = f'{args.work_dir}/{args.dataset_name.upper()}'
nf = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=args.work_dir,
    create_tb_writer=True,
    files_to_copy=[__file__],
)

data_desc = LanguageModelDataDesc(args.dataset_name, args.data_dir, args.do_lower_case)

# define tokenizer, in this example we use word-level tokenizer
# we also adjust the vocabulary size to make it multiple of 8 to accelerate
# training in fp16 mode with the use of Tensor Cores
tokenizer = nemo_nlp.data.WordTokenizer(f"{args.data_dir}/{args.tokenizer_model}")
vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)

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
    vocab_size=vocab_size,
    mask_future=True,
    attn_score_dropout=args.attn_score_dropout,
    attn_layer_dropout=args.attn_layer_dropout,
    max_seq_length=args.max_seq_length,
)

log_softmax = TokenClassifier(args.d_model, num_classes=vocab_size, num_layers=1, log_softmax=True)

loss = SmoothedCrossEntropyLoss(pad_id=tokenizer.pad_id, label_smoothing=args.label_smoothing)

# tie weight of embedding and log_softmax layers
# log_softmax.mlp.last_linear_layer.weight = encoder.embedding_layer.token_embedding.weight
log_softmax.tie_weights_with(
    encoder,
    weight_names=["mlp.layer0.weight"],
    name2name_and_transform={
        "mlp.layer0.weight": ("embedding_layer.token_embedding.weight", WeightShareTransform.SAME)
    },
)


def create_pipeline(
    dataset, max_seq_length=args.max_seq_length, batch_step=args.max_seq_length, batch_size=args.batch_size
):
    data_layer = LanguageModelingDataLayer(dataset, tokenizer, max_seq_length, batch_size, batch_step)
    input_data = data_layer()
    src_hiddens = encoder(input_ids=input_data.input_ids, input_mask_src=input_data.input_mask)
    logits = log_softmax(hidden_states=src_hiddens)
    return loss(logits=logits, labels=input_data.labels)


train_loss = create_pipeline(
    f"{args.data_dir}/{args.train_dataset}",
    args.max_seq_length,
    batch_step=args.max_seq_length,
    batch_size=args.batch_size,
)
eval_loss = create_pipeline(
    f"{args.data_dir}/{args.eval_dataset}",
    args.max_seq_length,
    batch_step=args.predict_last_k,
    batch_size=args.eval_batch_size,
)

# callback which prints training loss once in a while
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    step_freq=100,
    print_func=lambda x: logging.info(str(x[0].item())),
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=nf.tb_writer,
)

# callback which calculates evaluation loss
eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=[eval_loss],
    user_iter_callback=eval_iter_callback,
    user_epochs_done_callback=eval_epochs_done_callback,
    eval_step=args.eval_freq,
    tb_writer=nf.tb_writer,
)

# callback which saves checkpoints once in a while
callback_ckpt = nemo.core.CheckpointCallback(
    folder=nf.checkpoint_dir, epoch_freq=args.save_epoch_freq, step_freq=args.save_step_freq, checkpoints_to_keep=-1
)

# define learning rate decay policy
lr_policy_fn = CosineAnnealing(args.max_steps, warmup_steps=args.warmup_steps)

# define and launch training algorithm (optimizer)
max_num_epochs = 0 if args.interactive else args.num_epochs

callbacks = [callback_ckpt]

if not args.interactive:
    callbacks.extend([train_callback, eval_callback])

nf.train(
    tensors_to_optimize=[train_loss],
    callbacks=callbacks,
    lr_policy=lr_policy_fn,
    batches_per_step=args.iter_per_step,
    optimizer=args.optimizer_kind,
    optimization_params={
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": (args.beta1, args.beta2),
    },
)
