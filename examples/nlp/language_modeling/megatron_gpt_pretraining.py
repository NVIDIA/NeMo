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

from pathlib import Path

from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.plugins.environments.torchelastic_environment import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.parts.nlp_overrides import (
    NLPDDPPlugin,
    NLPNativeBfloat16PrecisionPlugin,
    NLPNativeMixedPrecisionPlugin,
    NLPPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    plugins = [NLPDDPPlugin(num_nodes=cfg.trainer.num_nodes)]
    if cfg.trainer.precision == 16:
        plugins.append(
            NLPNativeMixedPrecisionPlugin(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
            )
        )
    elif cfg.trainer.precision == 'bf16':
        plugins.append(NLPNativeBfloat16PrecisionPlugin())
    else:
        plugins.append(NLPPrecisionPlugin())

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, **cfg.trainer)

    exp_manager(trainer, cfg.exp_manager)

    # update resume from checkpoint found by exp_manager
    resume_from_checkpoint = trainer.resume_from_checkpoint
    if resume_from_checkpoint is not None:
        mp_rank = compute_model_parallel_rank(trainer.local_rank, cfg.model.tensor_model_parallel_size)
        resume_from_checkpoint = Path(resume_from_checkpoint)
        resume_from_checkpoint = resume_from_checkpoint.parent.parent.joinpath(f'mp_rank_{mp_rank:02d}').joinpath(
            resume_from_checkpoint.name
        )
        resume_from_checkpoint = str(resume_from_checkpoint)
        logging.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')

    trainer.checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)
    # Override timer callback to a stateless one
    for idx, callback in enumerate(trainer.callbacks):
        if isinstance(callback, Timer):
            trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time,)

    model = MegatronGPTModel(cfg.model, trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()
