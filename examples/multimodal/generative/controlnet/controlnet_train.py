from datetime import timedelta

import pytorch_lightning as pl
import torch
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.strategies.ddp import DDPStrategy

from nemo.collections.multimodal.data.common.webdataset import WebDatasetCommon
from nemo.collections.multimodal.data.stable_diffusion.augmentation.augmentations import (
    construct_image_augmentations,
    identical_transform,
)
from nemo.collections.multimodal.models.controlnet.controlnet import MegatronControlNet
from nemo.collections.multimodal.models.controlnet.util import ImageLogger
from nemo.collections.multimodal.parts.stable_diffusion.utils import instantiate_from_config
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import StatelessTimer, exp_manager


@hydra_runner(config_path='conf', config_name='controlnet_v1-5.yaml')
def main(cfg):
    megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
    with_distributed_adam = cfg.model.optim.get('name') == 'distributed_fused_adam'

    plugins = []
    callbacks = []

    # Tune for DDP
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )

    # dataset, _ = build_train_valid_datasets(cfg.model, 0)

    if cfg.trainer.precision in [16, 'bf16']:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 65536.0),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
        if megatron_amp_O2 and not with_distributed_adam:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    if cfg.model.get('image_logger', None):
        callbacks.append(ImageLogger(**cfg.model.image_logger))

    trainer = pl.Trainer(**cfg.trainer, plugins=plugins, callbacks=callbacks, strategy=strategy)

    exp_manager(trainer, cfg.get("exp_manager", None))

    model = MegatronControlNet(cfg.model, trainer)
    trainer.fit(model)


if __name__ == '__main__':
    main()
