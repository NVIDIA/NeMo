# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from megatron.core.optimizer import MegatronOptimizer, get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from nemo.tron.config import ConfigContainer


def setup_optimizer(cfg: ConfigContainer, model, no_weight_decay_cond=None, scale_lr_cond=None, lr_mult=1.0):
    optimizer = get_megatron_optimizer(cfg.optimizer_config, model, no_weight_decay_cond, scale_lr_cond, lr_mult)
    scheduler = _get_scheduler(cfg, optimizer)

    return optimizer, scheduler


def _get_scheduler(cfg: ConfigContainer, optimizer: MegatronOptimizer):
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=cfg.scheduler_config.lr_warmup_init,
        max_lr=cfg.optimizer_config.lr,
        min_lr=cfg.optimizer_config.min_lr,
        lr_warmup_steps=cfg.scheduler_config.lr_warmup_steps,
        lr_decay_steps=cfg.scheduler_config.lr_decay_steps,
        lr_decay_style=cfg.scheduler_config.lr_decay_style,
        start_wd=cfg.scheduler_config.start_weight_decay,
        end_wd=cfg.scheduler_config.end_weight_decay,
        wd_incr_steps=cfg.scheduler_config.wd_incr_steps,
        wd_incr_style=cfg.scheduler_config.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=cfg.scheduler_config.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=cfg.scheduler_config.override_opt_param_scheduler,
        wsd_decay_steps=cfg.scheduler_config.wsd_decay_steps,
        lr_wsd_decay_style=cfg.scheduler_config.lr_wsd_decay_style,
    )

    return scheduler
