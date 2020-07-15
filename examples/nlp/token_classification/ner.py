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

import nemo.collections.nlp as nemo_nlp
from nemo.utils.arguments import add_optimizer_args, add_scheduler_args


def main():
    parser = ArgumentParser()
    parser = add_optimizer_args(parser, optimizer="adam", default_lr="2e-5", default_opt_args={"weight_decay": 0.01})
    parser = add_scheduler_args(parser)

    parser.add_argument("--data_dir", type=str, required=True, default='', help="Path to data folder")
    parser.add_argument("--num_epochs", default=3, type=int, help="Number of epochs to train")
    parser.add_argument(
        "--pretrained_model_name",
        default="bert-base-uncased",
        type=str,
        help="Pretrained language model name",
        choices=nemo_nlp.modules.get_pretrained_lm_models_list(),
    )
    parser.add_argument("--work_dir", default='output', type=str, help="Number of epochs to train")

    # Training Arguments
    parser.add_argument(
        "--gpus", default=2, type=int, help="Number of GPUs",
    )
    parser.add_argument("--num_nodes", default=1, type=int, help="Number of nodes")
    parser.add_argument("--max_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help="Maximum number of steps to train. If it is set, num_epochs would get ignored.",
    )
    parser.add_argument(
        "--scheduler",
        default='WarmupAnnealing',
        type=str,
        choices=["SquareRootAnnealing", "CosineAnnealing", "WarmupAnnealing"],
        help="Scheduler.",
    )
    parser.add_argument(
        "--accumulate_grad_batches", default=1, type=int, help="Accumulates grads every k batches.",
    )
    parser.add_argument(
        "--amp_level", default="O0", type=str, choices=["O0", "O1", "O2"], help="01/02 to enable mixed precision"
    )
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(
            "Dataset not found. For NER, CoNLL-2003 dataset can be obtained at"
            "https://github.com/kyzhouhzau/BERT-NER/tree/master/data."
        )

    model = nemo_nlp.models.NERModel(data_dir=args.data_dir)
    model.setup_training_data(args.data_dir, train_data_layer_config={'shuffle': True})
    model.setup_validation_data(data_dir=args.data_dir, val_data_layer_config={'shuffle': False})
    # Setup optimizer and scheduler
    scheduler_args = {
        'monitor': 'val_loss',  # pytorch lightning requires this value
        'max_steps': args.max_steps,
    }

    if args.max_epochs:
        iters_per_batch = args.max_epochs / float(args.gpus * args.num_nodes * args.accumulate_grad_batches)
        scheduler_args['iters_per_batch'] = iters_per_batch
    else:
        scheduler_args['iters_per_batch'] = None

    scheduler_args["name"] = args.scheduler  # name of the scheduler
    scheduler_args["args"] = {
        "name": "auto",  # name of the scheduler config
        "params": {
            'warmup_ratio': args.warmup_ratio,
            'warmup_steps': args.warmup_steps,
            'last_epoch': args.last_epoch,
        },
    }

    model.setup_optimization(
        optim_config={
            'name': args.optimizer,  # name of the optimizer
            'lr': args.lr,
            'args': {
                "name": "auto",  # name of the optimizer config
                "params": {},  # Put args.opt_args here explicitly
            },
            'sched': scheduler_args,
        }
    )

    # multi GPU
    trainer = pl.Trainer(
        val_check_interval=1.0,
        amp_level=args.amp_level,
        precision=32 if args.amp_level == "O0" else 16,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        distributed_backend=None if args.gpus == 1 else 'ddp',
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
