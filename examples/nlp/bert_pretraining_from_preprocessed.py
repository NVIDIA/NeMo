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

import random
from argparse import ArgumentParser

import pytorch_lightning as pl
from transformers import BertConfig

from nemo.collections.nlp.models.lm_model import BERTLMModel


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data folder")
    parser.add_argument("--num_classes", type=int, default=30522, help="Number of classes")
    parser.add_argument("--config_file", default=None, type=str, help="The BERT model config")
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
        default="novograd",
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
    parser.add_argument("--max_seq_length", default=128, type=int)
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
    parser.add_argument(
        "--work_dir", default="outputs/bert_lm", type=str, help="Output directory for checkpoints, logs etc."
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

    bert_model = BERTLMModel(pretrained_model_name='bert-base-uncased', config_file=args.config_file)
    bert_model.setup_training_data(
        train_data_layer_params={
            'train_data': args.data_dir,
            'max_pred_length': args.max_pred_length,
            'batch_size': args.batch_size,
        },
    )

    optim_params = {'lr': args.lr, 'weight_decay': args.weight_decay, 'betas': args.betas}
    bert_model.setup_optimization(optim_params)

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        amp_level=args.amp_level,
        precision=16,
        gpus=args.num_gpus,
        max_steps=args.max_steps,
        distributed_backend='ddp',
        replace_sampler_ddp=False,
        accumulate_grad_batches=args.accumulate_grad_batches,
        # progress_bar_refresh_rate=1,
    )
    trainer.fit(bert_model)


if __name__ == '__main__':
    main()
