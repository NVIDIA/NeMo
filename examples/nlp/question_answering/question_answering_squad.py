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
import sys

import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl

from nemo.collections.nlp.models.qa_model import QAModel
from nemo.core.optim.lr_scheduler import CosineAnnealing, SquareRootAnnealing, WarmupAnnealing
from nemo.utils import logging


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config: {cfg.pretty()}')
    trainer = pl.Trainer(**cfg.pl.trainer)
    question_answering_model = QAModel(cfg.model, trainer=trainer)
    trainer.fit(question_answering_model)

if __name__ == '__main__':
    main()

@hydra.main(config_path="conf", config_name="config")
def main():

    model = QAModel(
        pretrained_model_name=args.pretrained_model_name, config_file=args.config_file, num_classes=2, num_layers=1,
    )
    model.setup_training_data(
        train_data_layer_config={
            'data_file': args.train_file,
            'doc_stride': args.doc_stride,
            'max_query_length': args.max_query_length,
            'max_seq_length': args.max_seq_length,
            'version_2_with_negative': args.version_2_with_negative,
            'use_cache': args.use_cache,
            'batch_size': args.batch_size,
        },
    )
    model.setup_validation_data(
        val_data_layer_config={
            'data_file': args.eval_file,
            'doc_stride': args.doc_stride,
            'max_query_length': args.max_query_length,
            'max_seq_length': args.max_seq_length,
            'version_2_with_negative': args.version_2_with_negative,
            'use_cache': args.use_cache,
            'batch_size': args.batch_size,
        },
    )
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

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        val_check_interval=args.val_check_interval,
        amp_level=args.amp_level,
        precision=args.precision,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        distributed_backend='ddp',
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
