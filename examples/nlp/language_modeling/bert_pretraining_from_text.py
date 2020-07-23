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


import pytorch_lightning as pl
from omegaconf import DictConfig

from nemo.collections.nlp.models.lm_model import BERTLMModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    train_data_file = os.path.join(cfg.data_dir, 'train.txt')
    valid_data_file = os.path.join(cfg.data_dir, 'valid.txt')
    vocab_size = None
    if cfg.config_file:
        vocab_size = json.load(open(cfg.config_file))['vocab_size']
    preprocessing_args = {
        'data_file': train_data_file,
        'tokenizer_name': cfg.tokenizer_name,
        'vocab_size': vocab_size,
        'sample_size': cfg.sample_size,
        'pretrained_model_name': cfg.pretrained_model_name,
        'tokenizer_model': cfg.tokenizer_model,
        'do_lower_case': cfg.do_lower_case,
    }
    bert_model = BERTLMModel(
        pretrained_model_name=cfg.pretrained_model_name,
        config_file=cfg.config_file,
        preprocessing_args=preprocessing_args,
    )
    bert_model.setup_training_data(
        train_data_layer_config={
            'dataset': train_data_file,
            'batch_size': cfg.batch_size,
            'max_seq_length': cfg.max_seq_length,
            'mask_probability': cfg.short_seq_prob,
            'short_seq_prob': cfg.short_seq_prob,
        },
    )
    bert_model.setup_validation_data(
        val_data_layer_config={
            'dataset': valid_data_file,
            'batch_size': cfg.batch_size,
            'max_seq_length': cfg.max_seq_length,
            'mask_probability': cfg.short_seq_prob,
            'short_seq_prob': cfg.short_seq_prob,
        },
    )

    # Setup optimizer and scheduler
    scheduler_args = {
        'monitor': 'val_loss',  # pytorch lightning requires this value
        'max_steps': cfg.max_steps,
    }

    scheduler_args["name"] = cfg.scheduler  # name of the scheduler
    scheduler_args["cfg"] = {
        "name": "auto",  # name of the scheduler config
        "params": {
            'warmup_ratio': cfg.warmup_ratio,
            'warmup_steps': cfg.warmup_steps,
            'min_lr': cfg.min_lr,
            'last_epoch': cfg.last_epoch,
        },
    }

    if cfg.max_steps is None:
        if cfg.gpus == 0:
            # training on CPU
            iters_per_batch = cfg.max_epochs / float(cfg.num_nodes * cfg.accumulate_grad_batches)
        else:
            iters_per_batch = cfg.max_epochs / float(cfg.gpus * cfg.num_nodes * cfg.accumulate_grad_batches)
        scheduler_args['iters_per_batch'] = iters_per_batch
    else:
        scheduler_args['iters_per_batch'] = None

    bert_model.setup_optimization(
        optim_config={
            'name': cfg.optimizer,  # name of the optimizer
            'lr': cfg.lr,
            'cfg': {
                "name": "auto",  # name of the optimizer config
                "params": {},  # Put cfg.opt_args here explicitly
            },
            'sched': scheduler_args,
        }
    )

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        val_check_interval=cfg.val_check_interval,
        amp_level=cfg.amp_level,
        precision=cfg.precision,
        gpus=cfg.gpus,
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        distributed_backend='ddp',
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        gradient_clip_val=cfg.gradient_clip_val,
    )
    trainer.fit(bert_model)


if __name__ == '__main__':
    main()
