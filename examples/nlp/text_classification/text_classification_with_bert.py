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

# TODO: WIP

import os
from argparse import ArgumentParser

import pytorch_lightning as pl

from nemo.collections.nlp.models.text_classification_model import TextClassificationModel
from nemo.core.optim.lr_scheduler import WarmupAnnealing
from nemo.utils.arguments import add_optimizer_args, add_scheduler_args


def main():
    parser = ArgumentParser(description='Sentence classification with pretrained BERT models')

    parser = add_optimizer_args(parser, optimizer="adam", default_lr="2e-5", default_opt_args={"weight_decay": 0.01})
    parser = add_scheduler_args(parser)

    # Data Arguments
    parser.add_argument("--work_dir", default='outputs', type=str)
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        type=str,
        help="The folder containing the checkpoints for the model to continue training",
    )
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--train_file_prefix", default='train', type=str, help="train file prefix")
    parser.add_argument("--eval_file_prefix", default='dev', type=str, help="eval file prefix")
    parser.add_argument("--no_shuffle", action='store_false', dest="shuffle", help="Shuffle is enabled by default.")
    parser.add_argument("--num_train_samples", default=-1, type=int, help="Number of samples to use for training")
    parser.add_argument("--num_val_samples", default=-1, type=int, help="Number of samples to use for evaluation")
    parser.add_argument("--num_workers", default=2, type=int, help="The number of workers for the data loaders.")
    parser.add_argument(
        "--pin_memory", action='store_true', help="Whether to enable the pin_memory feature of the data loaders."
    )
    parser.add_argument(
        "--use_cache", action='store_true', help="When specified loads and stores cache preprocessed data."
    )

    # BERT Arguments
    parser.add_argument(
        '--pretrained_model_name',
        default='roberta-base',
        type=str,
        help='Name of the pre-trained model',
        # choices=nemo_nlp.modules.common.get_pretrained_lm_models_list(),
    )
    parser.add_argument("--bert_checkpoint", default=None, type=str, help="Path to pre-trained BERT checkpoint")
    parser.add_argument("--bert_config", default=None, type=str, help="Path to bert config file in json format")
    parser.add_argument(
        "--tokenizer",
        default="nemobert",
        type=str,
        choices=["nemobert", "sentencepiece"],
        help="tokenizer to use, only relevant when using custom pretrained checkpoint.",
    )
    parser.add_argument("--vocab_file", default=None, type=str, help="Path to the vocab file.")
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
        + "For tokenizer only applicable when tokenizer is build with vocab file.",
    )
    parser.add_argument("--batch_size", default=32, type=int, help="Training and evaluation batch size")
    parser.add_argument(
        "--max_seq_length",
        default=36,
        type=int,
        help="The maximum total input sequence length after tokenization.Sequences longer than this will be \
                        truncated, sequences shorter will be padded.",
    )

    # Model Arguments
    parser.add_argument("--num_output_layers", default=2, type=int, help="Number of layers in the Classifier")
    parser.add_argument("--fc_dropout", default=0.1, type=float, help="Dropout rate")
    parser.add_argument("--class_balancing", default="None", type=str, choices=["None", "weighted_loss"])

    # Training Arguments
    parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs")
    parser.add_argument("--num_nodes", default=1, type=int, help="Number of nodes")
    parser.add_argument("--max_epochs", default=100, type=int, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help="Maximum number of steps to train. If it is set, num_epochs would get ignored.",
    )
    parser.add_argument(
        "--accumulate_grad_batches", default=1, type=int, help="Accumulates grads every k batches.",
    )
    parser.add_argument(
        "--amp_level", default="O0", type=str, choices=["O0", "O1", "O2"], help="01/02 to enable mixed precision"
    )
    parser.add_argument("--local_rank", default=None, type=int, help="For distributed training: local_rank")

    # Validation Arguments
    parser.add_argument("--save_epoch_freq", default=1, type=int, help="Epoch frequency of saving checkpoints")
    parser.add_argument("--save_step_freq", default=-1, type=int, help="Step frequency of saving checkpoints")
    parser.add_argument('--loss_step_freq', default=25, type=int, help='Frequency of printing loss')
    parser.add_argument('--eval_epoch_freq', default=1, type=int, help='Frequency of evaluation')

    args = parser.parse_args()

    text_classification_model = TextClassificationModel(
        data_dir=args.data_dir,
        pretrained_model_name=args.pretrained_model_name,
        bert_config=args.bert_config,
        num_output_layers=args.num_output_layers,
        fc_dropout=args.fc_dropout,
        class_balancing=args.class_balancing,
    )

    dataloader_params_train = {
        "max_seq_length": args.max_seq_length,
        "num_samples": args.num_train_samples,
        "shuffle": args.shuffle,
        "use_cache": args.use_cache,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }

    text_classification_model.setup_training_data(
        file_path=os.path.join(args.data_dir, f'{args.train_file_prefix}.tsv'),
        dataloader_params=dataloader_params_train,
    )

    dataloader_params_val = {
        "max_seq_length": args.max_seq_length,
        "num_samples": args.num_val_samples,
        "shuffle": False,
        "use_cache": args.use_cache,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }

    text_classification_model.setup_validation_data(
        file_path=os.path.join(args.data_dir, f'{args.eval_file_prefix}.tsv'), dataloader_params=dataloader_params_val,
    )

    # Setup optimizer and scheduler
    scheduler_args = {
        'monitor': 'val_loss',  # pytorch lightning requires this value
        'max_steps': args.max_steps,
    }

    scheduler_args["name"] = 'WarmupAnnealing'  # name of the scheduler
    scheduler_args["args"] = {
        "name": "auto",  # name of the scheduler config
        "params": {
            'warmup_ratio': args.warmup_ratio,
            'warmup_steps': args.warmup_steps,
            'last_epoch': args.last_epoch,
        },
    }

    if args.max_steps is None:
        if args.gpus == 0:
            # training on CPU
            iters_per_batch = args.max_epochs / float(args.num_nodes * args.accumulate_grad_batches)
        else:
            iters_per_batch = args.max_epochs / float(args.gpus * args.num_nodes * args.accumulate_grad_batches)
        scheduler_args['iters_per_batch'] = iters_per_batch
    else:
        scheduler_args['iters_per_batch'] = None

    text_classification_model.setup_optimization(
        optim_params={
            'name': args.optimizer,  # name of the optimizer
            'lr': args.lr,
            'args': {
                "name": "auto",  # name of the optimizer config
                "params": {},  # Put args.opt_args here explicitly
            },
            'sched': scheduler_args,
        }
    )

    trainer = pl.Trainer(
        check_val_every_n_epoch=args.eval_epoch_freq,
        amp_level=args.amp_level,
        precision=32 if args.amp_level == "O0" else 16,  # TODO: How to set precision?
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        distributed_backend=None
        if args.gpus == 1
        else "ddp",  # TODO: How to switch between multi-gpu, single-gpu, multi-node here?
    )
    trainer.fit(text_classification_model)


if __name__ == '__main__':
    main()
