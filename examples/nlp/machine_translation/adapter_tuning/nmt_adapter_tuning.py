# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.plugins.environments.torchelastic_environment import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from nemo.collections.nlp.data.machine_translation.preproc_mt_data import MTDataPreproc
from nemo.collections.nlp.models.language_modeling.megatron_bart_model import MegatronBARTModel
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.models.machine_translation.megatron_nmt_model import MegatronNMTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager


@hydra_runner(config_path="conf", config_name="aayn_base_adapter_megatron")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_o2 = cfg.model.get('megatron_amp_O2', False)
    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )
    if cfg.trainer.precision in [16, 'bf16']:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
        if megatron_amp_o2:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer, callbacks=[ModelSummary(max_depth=3)])

    # tokenizers will be trained and and tarred training data will be created if needed
    # model config is then updated
    if cfg.model.preproc_out_dir is not None:
        MTDataPreproc(cfg=cfg.model, trainer=trainer)

    # tokenizers will be trained and and tarred training data will be created if needed
    # model config is then updated
    if cfg.model.preproc_out_dir is not None:
        MTDataPreproc(cfg=cfg.model, trainer=trainer)

    exp_manager(trainer, cfg.exp_manager)

    # update resume from checkpoint found by exp_manager
    if cfg.model.resume_from_checkpoint is not None:
        resume_from_checkpoint = cfg.model.resume_from_checkpoint
    else:
        resume_from_checkpoint = trainer._checkpoint_connector.resume_from_checkpoint_fit_path
    logging.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')

    trainer._checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)
    # Override timer callback to a stateless one
    for idx, callback in enumerate(trainer.callbacks):
        if isinstance(callback, Timer):
            trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time,)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    # create the base model with adapters
    if hasattr(cfg.model, 'pretrained_model_path') and cfg.model.pretrained_model_path is not None:

        base_model_cfg = MegatronNMTModel.restore_from(
            cfg.model.pretrained_model_path, trainer=trainer, return_config=True
        )

        # Class target for the new class being restored.
        base_model_cfg.target = "nemo.collections.nlp.models.machine_translation.megatron_nmt_model.MegatronNMTModel"

        with open_dict(base_model_cfg):
            # set model path and adapters
            base_model_cfg.pretrained_model_path = cfg.model.pretrained_model_path
            base_model_cfg.adapter_tuning = cfg.model.adapter_tuning

            # override data paths
            # Override data and global/micro batch size.
            base_model_cfg.train_ds = cfg.model.train_ds
            base_model_cfg.train_ds.micro_batch_size = cfg.model.micro_batch_size
            base_model_cfg.train_ds.global_batch_size = cfg.model.global_batch_size
            if hasattr(cfg.model, 'validation_ds'):
                base_model_cfg.validation_ds = cfg.model.validation_ds
            else:
                raise AttributeError(f"No validation dataset found in config.")
            if hasattr(cfg.model, 'test_ds'):
                base_model_cfg.test_ds = cfg.model.test_ds

            # Class target for the new class being restored.
            base_model_cfg.target = (
                "nemo.collections.nlp.models.machine_translation.megatron_nmt_model.MegatronNMTModel"
            )

        # create model
        model = MegatronNMTModel.restore_from(
            cfg.model.pretrained_model_path,
            trainer=trainer,
            override_config_path=base_model_cfg,
            save_restore_connector=NLPSaveRestoreConnector(),
        )

        trainer.fit(model)
        trainer.validate(model)
    else:
        raise ValueError("config should provide pretrained_model_path for adapter tuning.")


if __name__ == '__main__':
    main()
