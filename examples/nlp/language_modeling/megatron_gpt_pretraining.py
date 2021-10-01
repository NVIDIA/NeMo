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
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    NLPCheckpointConnector,
    NLPDDPPlugin,
    NLPNativeMixedPrecisionPlugin,
    NLPPrecisionPlugin,
    NLPSaveRestoreConnector,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    if cfg.trainer.precision == 16:
        trainer = Trainer(
            plugins=[
                NLPDDPPlugin(num_nodes=cfg.trainer.num_nodes),
                NLPNativeMixedPrecisionPlugin(
                    init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                    growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                ),
            ],
            **cfg.trainer,
        )
    else:
        trainer = Trainer(plugins=[NLPDDPPlugin(num_nodes=cfg.trainer.num_nodes), NLPPrecisionPlugin()], **cfg.trainer)

    # TODO: possibly add model parallel size arg to exp_manager
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

    trainer.checkpoint_connector = NLPCheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)

    model = MegatronGPTModel(cfg.model, trainer)

    model._save_restore_connector = NLPSaveRestoreConnector()

    trainer.fit(model)

    if cfg.model.get('nemo_file_path', None) is not None:
        model.save_to(cfg.model.nemo_file_path)


if __name__ == '__main__':
    main()
