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


import os

import hydra

import pytorch_lightning as pl

from nemo.collections.nlp.models.text_classification_model import TextClassificationModel
from nemo.core.optim.lr_scheduler import WarmupAnnealing
from nemo.utils import logging
from nemo.utils.arguments import add_optimizer_args, add_scheduler_args

@hydra.main(config_name="config")
def main(cfg):
    logging.info(f'Config: {cfg.pretty}')

    text_classification_model = TextClassificationModel(
        data_dir=cfg.data_dir,
        pretrained_model_name=cfg.pretrained_model_name,
        bert_config=cfg.bert_config,
        num_output_layers=cfg.num_output_layers,
        fc_dropout=cfg.fc_dropout,
        class_balancing=cfg.class_balancing,
    )

    dataloader_params_train = {
        "max_seq_length": cfg.max_seq_length,
        "num_samples": cfg.num_train_samples,
        "shuffle": cfg.shuffle,
        "use_cache": cfg.use_cache,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
    }

    text_classification_model.setup_training_data(
        file_path=os.path.join(cfg.data_dir, f'{cfg.train_file_prefix}.tsv'),
        dataloader_params=dataloader_params_train,
    )

    dataloader_params_val = {
        "max_seq_length": cfg.max_seq_length,
        "num_samples": cfg.num_val_samples,
        "shuffle": False,
        "use_cache": cfg.use_cache,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
    }

    text_classification_model.setup_validation_data(
        file_path=os.path.join(cfg.data_dir, f'{cfg.eval_file_prefix}.tsv'), dataloader_params=dataloader_params_val,
    )

    # Setup optimizer and scheduler
    if 'sched' in cfg.optim:
        if cfg.pl.trainer.max_steps is None:
            if cfg.pl.trainer.gpus == 0:
                # training on CPU
                iters_per_batch = cfg.pl.trainer.max_epochs / float(
                    cfg.pl.trainer.num_nodes * cfg.pl.trainer.accumulate_grad_batches
                )
            else:
                iters_per_batch = cfg.pl.trainer.max_epochs / float(
                    cfg.pl.trainer.gpus * cfg.pl.trainer.num_nodes * cfg.pl.trainer.accumulate_grad_batches
                )
            cfg.optim.sched.iters_per_batch = iters_per_batch
        else:
            cfg.optim.sched.max_steps = cfg.pl.trainer.max_steps

    text_classification_model.setup_optimization(cfg.optim)
    trainer = pl.Trainer(**cfg.pl.trainer)
    trainer.fit(text_classification_model)


if __name__ == '__main__':
    main()
