# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import argparse
import os
from datetime import timedelta

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

from nemo.collections.multimodal.data.stable_diffusion.wds_sampler import WebDataloaderSamplerCallback
from nemo.collections.multimodal.data.stable_diffusion.webdataset import WebDatasetWithRawText
from nemo.collections.multimodal.models.stable_diffusion.ldm.ddpm_legacy import LatentDiffusion
from nemo.collections.multimodal.models.stable_diffusion.ldm_config import LatentDiffusionModelConfig
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import StatelessTimer, exp_manager


@hydra_runner(config_path='conf', config_name='sd_train.yaml')
def main(cfg):
    pl.seed_everything(42)

    # Tune for DDP
    if isinstance(cfg.trainer.strategy, str):
        strategy = cfg.trainer.strategy
    else:
        ddp_config = dict(cfg.trainer.strategy)
        if str(ddp_config.pop("allreduce_precision", "32")) == "16":  # can be bf16
            ddp_config["ddp_comm_hook"] = fp16_compress_hook
        ddp_config["timeout"] = timedelta(seconds=180)
        strategy = DDPStrategy(**ddp_config)
    del cfg.trainer.strategy

    batch_size = cfg.model.data.train.batch_size
    dataset = WebDatasetWithRawText(dataset_cfg=cfg.model.data, is_train=True,)
    data = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=cfg.model.data.num_workers, pin_memory=True, drop_last=False
    )
    global_bs = cfg.trainer.devices * cfg.trainer.num_nodes * batch_size

    callbacks = []
    if not cfg.model.data.webdataset.infinite_sampler:
        wds_sampler = WebDataloaderSamplerCallback(
            batch_size=batch_size, gradient_accumulation=cfg.trainer.accumulate_grad_batches
        )
        callbacks.append(wds_sampler)

    plugins = []
    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = pl.Trainer(**cfg.trainer, plugins=plugins, callbacks=callbacks, strategy=strategy)
    exp_manager(trainer, cfg.get("exp_manager", None))
    if (
        not cfg.model.data.webdataset.infinite_sampler
        and trainer._checkpoint_connector.resume_from_checkpoint_fit_path is not None
    ):
        # Reusming from previous training session
        wds_sampler.resume_flag = True

    model = LatentDiffusion(cfg.model, trainer).cuda()
    model.learning_rate = cfg.model.base_learning_rate * global_bs * cfg.trainer.accumulate_grad_batches

    trainer.fit(model, data)


if __name__ == '__main__':
    main()
