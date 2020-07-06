# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import random
from argparse import ArgumentParser

import pytorch_lightning as pl
from transformers import BertConfig

from nemo.collections.nlp.models.lm_model import BERTLMModel


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data folder")
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--sample_size", default=1e7, type=int, help="Data sample size.")
    parser.add_argument("--num_epochs", default=10, type=int, help="Number of training epochs.")
    parser.add_argument(
        "--mask_probability",
        default=0.15,
        type=float,
        help="Probability of masking a token in the input text during data processing.",
    )
    parser.add_argument(
        "--short_seq_prob",
        default=0.1,
        type=float,
        help="Probability of having a sequence shorter than the maximum sequence length `max_seq_length` in data processing.",
    )
    parser.add_argument("--config_file", default=None, type=str, help="The BERT model config")
    parser.add_argument("--pretrained_model_name", default='bert-base-cased', type=str, help="pretrained model name")
    parser.add_argument("--do_lower_case", action='store_true', help="lower case data")
    parser.add_argument(
        "--tokenizer_name", default='nemobert', type=str, choices=['sentencepiece', 'nemobert'], help="Tokenizer type"
    )
    parser.add_argument("--tokenizer_model", default=None, type=str, help="Tokenizer file for sentence piece")
    parser.add_argument("--num_gpus", default=1, type=int, help="Number Gpus")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size per worker for each model pass.")
    parser.add_argument(
        "--accumulate_grad_batches", default=1, type=int, help="Accumulates grads every k batches.",
    )
    parser.add_argument("--lr", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument(
        "--lr_policy", default=None, type=str, help="Learning rate policy.",
    )
    parser.add_argument(
        "--lr_warmup_proportion", default=0.05, type=float, help="Warm up proportion of total training iterations."
    )
    parser.add_argument(
        "--optimizer",
        default="adam_w",
        type=str,
        choices=["adam", "adam_w"],
        help="Optimizer algorithm for training.",
    )
    parser.add_argument(
        "--betas",
        default=(0.9, 0.999),
        type=tuple,
        help="coefficients used for computing running averages of gradient and its square",
    )
    parser.add_argument("--max_pred_length", default=128, type=int, help="Number Gpus")
    parser.add_argument(
        "--amp_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Automatic Mixed Precision optimization level.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay parameter of the optimizer.")
    parser.add_argument(
        "--only_mlm_loss", action="store_true", default=False, help="use only masked language model loss"
    )
    parser.add_argument(
        "--load_dir",
        default=None,
        type=str,
        help="Directory with weights and optimizer checkpoints. Used for resuming training.",
    )
    parser.add_argument(
        "--bert_checkpoint",
        default=None,
        type=str,
        help="Path to BERT encoder weights file. Used for encoder initialization for finetuning.",
    )
    parser.add_argument("--grad_norm_clip", type=float, default=-1, help="gradient clipping")
    parser.add_argument("--save_epoch_freq", default=1, type=int, help="Save checkpoints every given epoch.")
    parser.add_argument("--save_step_freq", default=100, type=int, help="Save checkpoints every given iteration.")
    parser.add_argument(
        "--train_step_freq", default=25, type=int, help="Print training metrics every given iteration."
    )
    parser.add_argument(
        "--eval_step_freq", default=25, type=int, help="Print evaluation metrics every given iteration."
    )
    parser.add_argument(
        "--max_predictions_per_seq",
        default=20,
        type=int,
        help="Maximum number of masked tokens to predict. Need to match the number of masked tokens in the input data sets.",
    )
    parser.add_argument(
        "--max_steps", default=100, type=int, help="Number of training steps.",
    )

    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    train_data_file = os.path.join(args.data_dir, 'train.txt')
    valid_data_file = os.path.join(args.data_dir, 'valid.txt')
    vocab_size = None
    if args.config_file:
        vocab_size = json.load(open(args.config_file))['vocab_size']
    preprocessing_args = {
        'data_file': train_data_file,
        'tokenizer_name': args.tokenizer_name,
        'vocab_size': vocab_size,
        'sample_size': args.sample_size,
        'pretrained_model_name': args.pretrained_model_name,
        'tokenizer_model': args.tokenizer_model,
        'do_lower_case': args.do_lower_case,
    }
    bert_model = BERTLMModel(
        pretrained_model_name=args.pretrained_model_name,
        config_file=args.config_file,
        preprocessing_args=preprocessing_args,
    )
    bert_model.setup_training_data(
        train_data_layer_params={
            'dataset': train_data_file,
            'max_predictions_per_seq': args.max_predictions_per_seq,
            'batch_size': args.batch_size,
            'max_seq_length': args.max_seq_length,
            'mask_probability': args.short_seq_prob,
            'short_seq_prob': args.short_seq_prob,
        },
    )
    bert_model.setup_validation_data(
        val_data_layer_params={
            'dataset': valid_data_file,
            'max_predictions_per_seq': args.max_predictions_per_seq,
            'batch_size': args.batch_size,
            'max_seq_length': args.max_seq_length,
            'mask_probability': args.short_seq_prob,
            'short_seq_prob': args.short_seq_prob,
        },
    )

    optim_params = {'lr': args.lr, 'weight_decay': args.weight_decay, 'betas': args.betas}
    bert_model.setup_optimization(optim_params, optimizer=args.optimizer)

    trainer = pl.Trainer(
        num_sanity_val_steps=10,
        amp_level=args.amp_level,
        precision=16,
        gpus=args.num_gpus,
        max_epochs=args.num_epochs,
        # max_steps=args.max_steps,
        distributed_backend='ddp',
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    trainer.fit(bert_model)


if __name__ == '__main__':
    main()
