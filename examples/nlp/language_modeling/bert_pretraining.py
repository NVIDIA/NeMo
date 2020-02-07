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

"""

To pretrain BERT on raw text dataset run
python bert_pretraining.py \
--amp_opt_level "O0" \
--data_dir data/lm/wikitext-2 \
--dataset_name wikitext-2 \
--work_dir outputs/bert_lm \
--batch_size 64 \
--lr 0.01 \
--lr_policy CosineAnnealing \
--lr_warmup_proportion 0.05 \
--optimizer novograd \
--beta1 0.95 \
--beta2 0.25 \
--tokenizer sentence-piece \
--vocab_size 3200 \
--hidden_size 768 \
--intermediate_size 3072 \
--num_hidden_layers 12 \
--num_attention_heads 12 \
--hidden_act "gelu" \
--save_step_freq 200 \
--num_epochs 10 \
--sample_size 10000000 \
--mask_probability 0.15 \
--short_seq_prob 0.1

To pretrain BERT large on preprocessed dataset,
download and preprocess dataset from here:
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/
Run the script:
./data/create_datasets_from_start.sh
and extract data into data_dir

Then run BERT large on the 512 sequence length dataset
python -m torch.distributed.launch --nproc_per_node=8 bert_pretraining.py \
--batch_size 8 \
--config_file bert_config.json
--data_dir data_dir \
--save_step_freq 200 \
--max_steps 1142857 \
--num_gpus 8 \
--batches_per_step 2 \
--amp_opt_level "O1" \
--lr_policy SquareRootAnnealing \
--beta1 0.9 \
--beta2 0.999 \
--lr_warmup_proportion 0.01 \
--optimizer adam_w \
--weight_decay 0.01 \
--lr 0.875e-4 \
--preprocessed_data

350000 iterations on a DGX1 with 8 V100 32GB GPUs with AMP O1 optimization
should finish under 5 days and yield an MRPC score of ACC/F1 85.05/89.35.
"""
import argparse
import math
import os

from transformers import BertConfig

import nemo.backends.pytorch.common as nemo_common
import nemo.collections.nlp as nemo_nlp
import nemo.core as nemo_core
from nemo import logging
from nemo.collections.nlp.data.datasets.lm_bert_dataset import BERTPretrainingDataDesc
from nemo.utils.lr_policies import get_lr_policy

parser = argparse.ArgumentParser(description='BERT pretraining')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--batches_per_step", default=1, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--lr_policy", default=None, type=str)
parser.add_argument("--lr_warmup_proportion", default=0.05, type=float)
parser.add_argument("--optimizer", default="novograd", type=str)
parser.add_argument("--beta1", default=0.95, type=float)
parser.add_argument("--beta2", default=0.25, type=float)
parser.add_argument("--amp_opt_level", default="O0", type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--tokenizer", default="sentence-piece", type=str, choices=["sentence-piece", "nemo-bert"])
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--sample_size", default=1e7, type=int)
parser.add_argument("--mask_probability", default=0.15, type=float)
parser.add_argument("--short_seq_prob", default=0.1, type=float)
parser.add_argument("--vocab_size", default=3200, type=int)
parser.add_argument("--hidden_size", default=768, type=int)
parser.add_argument("--intermediate_size", default=3072, type=int)
parser.add_argument("--num_hidden_layers", default=12, type=int)
parser.add_argument("--num_attention_heads", default=12, type=int)
parser.add_argument("--hidden_act", default="gelu", type=str)
parser.add_argument(
    "--max_predictions_per_seq",
    default=20,
    type=int,
    help="maximum number of masked tokens to predict,\
                    needed when --preprocessed_data is specified",
)
parser.add_argument("--data_dir", default="data/lm/wikitext-2", type=str)
parser.add_argument(
    "--preprocessed_data", action="store_true", default=False, help="specify if using preprocessed data"
)
parser.add_argument("--gradient_predivide", action="store_true", default=False, help="use gradient predivide")
parser.add_argument("--only_mlm_loss", action="store_true", default=False, help="use only masked language model loss")
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="if specified overrides --num_epochs.\
                        Used for preprocessed data",
)
parser.add_argument("--dataset_name", default="wikitext-2", type=str)
parser.add_argument("--load_dir", default=None, type=str)
parser.add_argument("--bert_checkpoint", default=None, type=str, help="specify path to pretrained BERT weights")
parser.add_argument("--work_dir", default="outputs/bert_lm", type=str)
parser.add_argument("--save_epoch_freq", default=1, type=int)
parser.add_argument("--save_step_freq", default=100, type=int)
parser.add_argument("--print_step_freq", default=25, type=int)
parser.add_argument("--config_file", default=None, type=str, help="The BERT model config")
args = parser.parse_args()

nf = nemo_core.NeuralModuleFactory(
    backend=nemo_core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=args.work_dir,
    create_tb_writer=True,
    files_to_copy=[__file__],
    add_time_to_log_dir=False,
)

if args.config_file is not None:
    config = BertConfig.from_json_file(args.config_file).to_dict()
    args.vocab_size = config['vocab_size']
    args.hidden_size = config['hidden_size']
    args.num_hidden_layers = config['num_hidden_layers']
    args.num_attention_heads = config['num_attention_heads']
    args.intermediate_size = config['intermediate_size']
    args.hidden_act = config['hidden_act']
    args.max_seq_length = config['max_position_embeddings']

if not args.preprocessed_data:
    special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    data_desc = BERTPretrainingDataDesc(
        args.dataset_name, args.data_dir, args.vocab_size, args.sample_size, special_tokens, 'train.txt'
    )
    if args.tokenizer == "sentence-piece":
        logging.info("To use SentencePieceTokenizer.")
        tokenizer = nemo_nlp.data.SentencePieceTokenizer(model_path=data_desc.tokenizer_model)
        tokenizer.add_special_tokens(special_tokens)
    elif args.tokenizer == "nemo-bert":
        logging.info("To use NemoBertTokenizer.")
        vocab_file = os.path.join(args.data_dir, 'vocab.txt')
        # To train on a Chinese dataset, use NemoBertTokenizer
        tokenizer = nemo_nlp.data.NemoBertTokenizer(vocab_file=vocab_file)
    else:
        raise ValueError("Please add your tokenizer " "or use sentence-piece or nemo-bert.")
    args.vocab_size = tokenizer.vocab_size

print(vars(args))
bert_model = nemo_nlp.nm.trainables.huggingface.BERT(
    vocab_size=args.vocab_size,
    num_hidden_layers=args.num_hidden_layers,
    hidden_size=args.hidden_size,
    num_attention_heads=args.num_attention_heads,
    intermediate_size=args.intermediate_size,
    max_position_embeddings=args.max_seq_length,
    hidden_act=args.hidden_act,
)

if args.bert_checkpoint is not None:
    bert_model.restore_from(args.bert_checkpoint)

""" create necessary modules for the whole translation pipeline, namely
data layers, BERT encoder, and MLM and NSP loss functions
"""

mlm_classifier = nemo_nlp.nm.trainables.token_classification_nm.BertTokenClassifier(
    args.hidden_size, num_classes=args.vocab_size, activation=args.hidden_act, log_softmax=True
)
mlm_loss_fn = nemo_nlp.nm.losses.MaskedLanguageModelingLossNM()
if not args.only_mlm_loss:
    nsp_classifier = nemo_nlp.nm.trainables.sequence_classification_nm.SequenceClassifier(
        args.hidden_size, num_classes=2, num_layers=2, activation='tanh', log_softmax=False
    )
    nsp_loss_fn = nemo_common.CrossEntropyLoss()

    bert_loss = nemo_nlp.nm.losses.LossAggregatorNM(num_inputs=2)

# tie weights of MLM softmax layer and embedding layer of the encoder
if mlm_classifier.mlp.last_linear_layer.weight.shape != bert_model.bert.embeddings.word_embeddings.weight.shape:
    raise ValueError("Final classification layer does not match embedding " "layer.")
mlm_classifier.mlp.last_linear_layer.weight = bert_model.bert.embeddings.word_embeddings.weight


def create_pipeline(data_file, batch_size, preprocessed_data=False, batches_per_step=1, **kwargs):
    if not preprocessed_data:
        max_seq_length, mask_probability, short_seq_prob = (
            kwargs['max_seq_length'],
            kwargs['mask_probability'],
            kwargs['short_seq_prob'],
        )
        data_layer = nemo_nlp.nm.data_layers.lm_bert_datalayer.BertPretrainingDataLayer(
            tokenizer, data_file, max_seq_length, mask_probability, short_seq_prob, batch_size=batch_size
        )
    else:
        training, max_predictions_per_seq = (kwargs['training'], kwargs['max_predictions_per_seq'])
        data_layer = nemo_nlp.nm.data_layers.lm_bert_datalayer.BertPretrainingPreprocessedDataLayer(
            data_file, max_predictions_per_seq, batch_size=batch_size, training=training
        )

    steps_per_epoch = math.ceil(len(data_layer) / (batch_size * args.num_gpus * batches_per_step))

    (input_ids, input_type_ids, input_mask, output_ids, output_mask, nsp_labels) = data_layer()
    hidden_states = bert_model(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
    mlm_logits = mlm_classifier(hidden_states=hidden_states)
    mlm_loss = mlm_loss_fn(logits=mlm_logits, output_ids=output_ids, output_mask=output_mask)
    if not args.only_mlm_loss:
        nsp_logits = nsp_classifier(hidden_states=hidden_states)
        nsp_loss = nsp_loss_fn(logits=nsp_logits, labels=nsp_labels)
        loss = bert_loss(loss_1=mlm_loss, loss_2=nsp_loss)
    else:
        loss = mlm_loss
        nsp_loss = None
    return loss, mlm_loss, nsp_loss, steps_per_epoch


if not args.preprocessed_data:
    train_loss, mlm_loss, nsp_loss, steps_per_epoch = create_pipeline(
        data_file=data_desc.train_file,
        preprocessed_data=False,
        max_seq_length=args.max_seq_length,
        mask_probability=args.mask_probability,
        short_seq_prob=args.short_seq_prob,
        batch_size=args.batch_size,
        batches_per_step=args.batches_per_step,
    )
else:
    max_pred_len = args.max_predictions_per_seq
    train_loss, mlm_loss, nsp_loss, steps_per_epoch = create_pipeline(
        data_file=args.data_dir,
        preprocessed_data=True,
        max_predictions_per_seq=max_pred_len,
        training=True,
        batch_size=args.batch_size,
        batches_per_step=args.batches_per_step,
    )

print("steps per epoch", steps_per_epoch)
# callback which prints training loss and perplexity once in a while
if not args.only_mlm_loss:
    log_tensors = [train_loss, mlm_loss, nsp_loss]
    print_msg = "Loss: {:.3f} MLM Loss: {:.3f} NSP Loss: {:.3f}"
else:
    log_tensors = [train_loss]
    print_msg = "Loss: {:.3f}"
train_callback = nemo_core.SimpleLossLoggerCallback(
    tensors=log_tensors,
    step_freq=args.print_step_freq,
    print_func=lambda x: logging.info(print_msg.format(*[y.item() for y in x])),
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=nf.tb_writer,
)

ckpt_callback = nemo_core.CheckpointCallback(
    folder=nf.checkpoint_dir,
    epoch_freq=args.save_epoch_freq,
    load_from_folder=args.load_dir,
    step_freq=args.save_step_freq,
)

# define learning rate decay policy
if args.lr_policy is not None:
    if args.max_steps < 0:
        lr_policy_fn = get_lr_policy(
            args.lr_policy, total_steps=args.num_epochs * steps_per_epoch, warmup_ratio=args.lr_warmup_proportion
        )
    else:
        lr_policy_fn = get_lr_policy(
            args.lr_policy, total_steps=args.max_steps, warmup_ratio=args.lr_warmup_proportion
        )
else:
    lr_policy_fn = None

config_path = f'{nf.checkpoint_dir}/bert-config.json'
if not os.path.exists(config_path):
    bert_model.config.to_json_file(config_path)

# define and launch training algorithm (optimizer)
optimization_params = {
    "batch_size": args.batch_size,
    "lr": args.lr,
    "betas": (args.beta1, args.beta2),
    "weight_decay": args.weight_decay,
}

if args.max_steps < 0:
    optimization_params['num_epochs'] = args.num_epochs
else:
    optimization_params['max_steps'] = args.max_steps
nf.train(
    tensors_to_optimize=[train_loss],
    lr_policy=lr_policy_fn,
    callbacks=[train_callback, ckpt_callback],
    optimizer=args.optimizer,
    batches_per_step=args.batches_per_step,
    gradient_predivide=args.gradient_predivide,
    optimization_params=optimization_params,
)
