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
import pytorch_lightning as pl
import torch
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from torch._dynamo import disable, optimize
from torch._inductor import config as inductor_config

from nemo.collections.multimodal.models.imagen.imagen import MegatronImagen
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path='conf', config_name='base64-500m')
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
    with_distributed_adam = cfg.model.optim.get('name') == 'distributed_fused_adam'

    allow_tf32 = cfg.get('allow_tf32', True)
    if allow_tf32:
        logging.info('Allow TensorFloat32 operations on supported devices')
    else:
        logging.info('Disable TensorFloat32 operations.')
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    plugins = []
    ddp_overlap = cfg.model.get('ddp_overlap', True)
    if ddp_overlap:
        logging.info('Enable DDP Overlap.')
    else:
        logging.info('Use Megatron default configuration for async grad allreduce')

    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=not ddp_overlap,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=True,
        bucket_cap_mb=256,
    )

    if cfg.trainer.precision in [16, 'bf16']:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
        if megatron_amp_O2 and not with_distributed_adam:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    callbacks = []
    trainer = Trainer(plugins=plugins, strategy=strategy, callbacks=callbacks, **cfg.trainer)

    exp_manager(trainer, cfg.exp_manager)
    # update resume from checkpoint found by exp_manager
    if cfg.model.get("resume_from_checkpoint") is not None:
        resume_from_checkpoint = cfg.model.resume_from_checkpoint
    else:
        resume_from_checkpoint = trainer._checkpoint_connector.resume_from_checkpoint_fit_path

    logging.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')

    trainer._checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):

        cfg.model.precision = cfg.trainer.precision

    model = MegatronImagen(cfg.model, trainer)

    if cfg.model.get("inductor", False):
        # Temporary hack to get rid of TorchDynamo issue with DDP
        # TODO: remove these if https://github.com/pytorch/pytorch/issues/94574 fixed
        torch.arange = disable(torch.arange)
        torch.ones = disable(torch.ones)
        torch.zeros = disable(torch.zeros)

        # TODO: remove this if latest TorchDynamo fixed `t.uniform_(0, 1)` failure
        torch.Tensor.uniform_ = disable(torch.Tensor.uniform_)

        # Disable TorchDynamo for unsupported function
        pl.core.LightningModule.log = disable(pl.core.LightningModule.log)

        # TorchInductor with CUDA graph can lead to OOM
        inductor_config.triton.cudagraphs = cfg.model.inductor_cudagraphs
        model.model.model.unet = torch.compile(model.model.model.unet)

    trainer.fit(model)


if __name__ == '__main__':
    main()
