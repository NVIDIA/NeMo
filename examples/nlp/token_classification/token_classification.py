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
Tutorial on how to use this script to solve NER task could be found here:
https://nvidia.github.io/NeMo/nlp/intro.html#named-entity-recognition
"""

import argparse
import os

import nemo.collections.nlp as nemo_nlp
import nemo.collections.nlp.utils.data_utils
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.collections.nlp.callbacks.token_classification_callback import eval_epochs_done_callback, eval_iter_callback
from nemo.collections.nlp.data.datasets.datasets_utils.data_preprocessing import calc_class_weights
from nemo.collections.nlp.nm.data_layers import BertTokenClassificationDataLayer
from nemo.collections.nlp.nm.trainables import TokenClassifier
from nemo.utils.lr_policies import get_lr_policy

# Parsing arguments
"""Provide extra arguments required for tasks."""
parser = argparse.ArgumentParser(description="Token classification with pretrained BERT")
parser.add_argument("--local_rank", default=None, type=int)

# training arguments
parser.add_argument(
    "--work_dir",
    default='output',
    type=str,
    help="The output directory where the model prediction and checkpoints will be written.",
)
parser.add_argument("--no_time_to_log_dir", action="store_true", help="whether to add time to work_dir or not")
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=5, type=int)
parser.add_argument("--amp_opt_level", default="O0", type=str, choices=["O0", "O1", "O2"])
parser.add_argument(
    "--save_epoch_freq",
    default=1,
    type=int,
    help="Frequency of saving checkpoint '-1' - step checkpoint won't be saved",
)
parser.add_argument(
    "--save_step_freq",
    default=-1,
    type=int,
    help="Frequency of saving checkpoint '-1' - step checkpoint won't be saved",
)
parser.add_argument(
    "--eval_step_freq", default=-1, type=int, help="Frequency of evaluation, -1 to evaluate every epoch"
)
parser.add_argument("--loss_step_freq", default=250, type=int, help="Frequency of printing loss")
parser.add_argument("--use_weighted_loss", action='store_true', help="Flag to indicate whether to use weighted loss")

# learning rate arguments
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--lr", default=5e-5, type=float)
parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--optimizer_kind", default="adam", type=str)

# task specific arguments
parser.add_argument("--fc_dropout", default=0.5, type=float)
parser.add_argument("--num_fc_layers", default=2, type=int)

# data arguments
parser.add_argument("--data_dir", default="/data", type=str)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--ignore_start_end", action='store_false')
parser.add_argument("--ignore_extra_tokens", action='store_false')
parser.add_argument("--none_label", default='O', type=str)
parser.add_argument("--mode", default='train_eval', choices=["train_eval", "train"], type=str)
parser.add_argument("--no_shuffle_data", action='store_false', dest="shuffle_data")
parser.add_argument("--use_cache", action='store_true', help="Whether to cache preprocessed data")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
parser.add_argument("--batches_per_step", default=1, type=int, help="Number of iterations per step.")
parser.add_argument(
    "--tokenizer",
    default="nemobert",
    type=str,
    choices=["nemobert", "sentencepiece"],
    help="tokenizer to use, only relevant when using custom pretrained checkpoint.",
)
parser.add_argument(
    "--vocab_file", default=None, help="Path to the vocab file. Required for pretrained Megatron models"
)
parser.add_argument(
    "--tokenizer_model",
    default=None,
    type=str,
    help="Path to pretrained tokenizer model, only used if --tokenizer is sentencepiece",
)
parser.add_argument(
    "--do_lower_case",
    action='store_true',
    help="Whether to lower case the input text. True for uncased models, False for cased models. "
    + "Only applicable when tokenizer is build with vocab file",
)

# model arguments
parser.add_argument(
    "--pretrained_model_name",
    default="bert-base-uncased",
    type=str,
    help="Name of the pre-trained model",
    choices=nemo_nlp.nm.trainables.get_pretrained_lm_models_list(),
)
parser.add_argument("--bert_checkpoint", default=None, type=str, help="Path to bert pretrained  checkpoint")
parser.add_argument("--bert_config", default=None, type=str, help="Path to bert config file in json format")


args = parser.parse_args()
logging.info(args)

if not os.path.exists(args.data_dir):
    raise FileNotFoundError(
        "Dataset not found. For NER, CoNLL-2003 dataset"
        "can be obtained at"
        "https://github.com/kyzhouhzau/BERT"
        "-NER/tree/master/data."
    )

nf = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=args.work_dir,
    create_tb_writer=True,
    files_to_copy=[__file__],
    add_time_to_log_dir=not args.no_time_to_log_dir,
)

output_file = f'{nf.work_dir}/output.txt'


model = nemo_nlp.nm.trainables.get_pretrained_lm_model(
    pretrained_model_name=args.pretrained_model_name,
    config=args.bert_config,
    vocab=args.vocab_file,
    checkpoint=args.bert_checkpoint,
)

hidden_size = model.hidden_size

tokenizer = nemo.collections.nlp.data.tokenizers.get_tokenizer(
    tokenizer_name=args.tokenizer,
    pretrained_model_name=args.pretrained_model_name,
    tokenizer_model=args.tokenizer_model,
    vocab_file=args.vocab_file,
    do_lower_case=args.do_lower_case,
)


def create_pipeline(
    pad_label=args.none_label,
    max_seq_length=args.max_seq_length,
    batch_size=args.batch_size,
    num_gpus=args.num_gpus,
    mode='train',
    batches_per_step=args.batches_per_step,
    label_ids=None,
    ignore_extra_tokens=args.ignore_extra_tokens,
    ignore_start_end=args.ignore_start_end,
    use_cache=args.use_cache,
    dropout=args.fc_dropout,
    num_layers=args.num_fc_layers,
    classifier=TokenClassifier,
):

    logging.info(f"Loading {mode} data...")
    shuffle = args.shuffle_data if mode == 'train' else False

    text_file = f'{args.data_dir}/text_{mode}.txt'
    label_file = f'{args.data_dir}/labels_{mode}.txt'

    if not (os.path.exists(text_file) or (os.path.exists(label_file))):
        raise FileNotFoundError(
            f'{text_file} or {label_file} not found. \
           The data should be splitted into 2 files: text.txt and labels.txt. \
           Each line of the text.txt file contains text sequences, where words\
           are separated with spaces. The labels.txt file contains \
           corresponding labels for each word in text.txt, the labels are \
           separated with spaces. Each line of the files should follow the \
           format:  \
           [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and \
           [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).'
        )

    data_layer = BertTokenClassificationDataLayer(
        tokenizer=tokenizer,
        text_file=text_file,
        label_file=label_file,
        pad_label=pad_label,
        label_ids=label_ids,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        shuffle=shuffle,
        ignore_extra_tokens=ignore_extra_tokens,
        ignore_start_end=ignore_start_end,
        use_cache=use_cache,
    )

    (input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, labels) = data_layer()

    if mode == 'train':
        label_ids = data_layer.dataset.label_ids
        class_weights = None

        if args.use_weighted_loss:
            logging.info(f"Using weighted loss")
            label_freqs = data_layer.dataset.label_frequencies
            class_weights = calc_class_weights(label_freqs)

        classifier = classifier(
            hidden_size=hidden_size, num_classes=len(label_ids), dropout=dropout, num_layers=num_layers
        )

        task_loss = CrossEntropyLossNM(logits_ndim=3, weight=class_weights)

    hidden_states = model(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
    logits = classifier(hidden_states=hidden_states)

    if mode == 'train':
        loss = task_loss(logits=logits, labels=labels, loss_mask=loss_mask)
        steps_per_epoch = len(data_layer) // (batch_size * num_gpus * batches_per_step)
        tensors_to_evaluate = [loss, logits]
        return tensors_to_evaluate, loss, steps_per_epoch, label_ids, classifier
    else:
        tensors_to_evaluate = [logits, labels, subtokens_mask]
        return tensors_to_evaluate, data_layer


callbacks = []
train_tensors, train_loss, steps_per_epoch, label_ids, classifier = create_pipeline()
logging.info(f"steps_per_epoch = {steps_per_epoch}")
# Create trainer and execute training action
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=train_tensors,
    print_func=lambda x: logging.info("Loss: {:.3f}".format(x[0].item())),
    get_tb_values=lambda x: [["loss", x[0]]],
    step_freq=args.loss_step_freq,
    tb_writer=nf.tb_writer,
)
callbacks.append(train_callback)


if "eval" in args.mode:
    eval_tensors, data_layer = create_pipeline(mode='dev', label_ids=label_ids, classifier=classifier)
    eval_callback = nemo.core.EvaluatorCallback(
        eval_tensors=eval_tensors,
        user_iter_callback=lambda x, y: eval_iter_callback(x, y),
        user_epochs_done_callback=lambda x: eval_epochs_done_callback(x, label_ids, f'{nf.work_dir}/graphs'),
        tb_writer=nf.tb_writer,
        eval_step=args.eval_step_freq if args.eval_step_freq > 0 else steps_per_epoch,
    )
    callbacks.append(eval_callback)

ckpt_callback = nemo.core.CheckpointCallback(
    folder=nf.checkpoint_dir, epoch_freq=args.save_epoch_freq, step_freq=args.save_step_freq
)
callbacks.append(ckpt_callback)

lr_policy_fn = get_lr_policy(
    args.lr_policy, total_steps=args.num_epochs * steps_per_epoch, warmup_ratio=args.lr_warmup_proportion
)

nf.train(
    tensors_to_optimize=[train_loss],
    callbacks=callbacks,
    lr_policy=lr_policy_fn,
    batches_per_step=args.batches_per_step,
    optimizer=args.optimizer_kind,
    optimization_params={"num_epochs": args.num_epochs, "lr": args.lr, "weight_decay": args.weight_decay},
)
