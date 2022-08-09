# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.plugins.environments.torchelastic_environment import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from nemo.collections.common.parts import adapter_modules
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPPlugin,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager


@hydra_runner(config_path="conf", config_name="megatron_gpt_adapter_tuning_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    plugins = [NLPDDPPlugin(no_ddp_communication_hook=True, find_unused_parameters=False,)]
    if cfg.trainer.precision == 16:
        scaler = GradScaler(
            init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
            growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
            hysteresis=cfg.model.get('hysteresis', 2),
        )
        plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, **cfg.trainer)
    exp_manager(trainer, cfg.exp_manager)

    # Override timer callback to a stateless one
    for idx, callback in enumerate(trainer.callbacks):
        if isinstance(callback, Timer):
            trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time,)

    model_cfg = MegatronGPTModel.restore_from(
            cfg.model.get('gpt_model_file'), trainer=trainer, return_config=True
        )
    with open_dict(model_cfg):
            model_cfg.megatron_amp_O2 = False
            model_cfg.micro_batch_size = cfg.model.micro_batch_size
            model_cfg.global_batch_size = cfg.model.global_batch_size
            model_cfg.precision = cfg.trainer.precision
            model_cfg.data.data_prefix = cfg.model.data.data_prefix
            model_cfg.data.splits_string = cfg.model.data.splits_string

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    model = MegatronGPTModel.restore_from(
                cfg.model.get('gpt_model_file'),
                trainer=trainer,
                save_restore_connector=NLPSaveRestoreConnector(),
                override_config_path=model_cfg,
            )

    print(model.summarize())
    for name, module in model.named_modules():
        if hasattr(module, 'add_adapter'):
            print(f'Adding adapter to {name}')
            module.add_adapter(name='adapter_1', cfg=adapter_modules.LinearAdapterConfig(in_features=768, dim=50))
    model.freeze()
    for name, module in model.named_modules():
        if hasattr(module, 'add_adapter'):
            print(f'unfreezing to {name}')
            module.set_enabled_adapters(enabled=True) 
            module.unfreeze_enabled_adapters()
    print(model.summarize())
    trainer.fit(model)


if __name__ == '__main__':
    main()
