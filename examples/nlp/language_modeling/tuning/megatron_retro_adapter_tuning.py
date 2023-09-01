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

import os
# from lightning_lite.plugins.environments import TorchElasticEnvironment
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector

from nemo.collections.nlp.parts.nlp_overrides import GradScaler, MegatronHalfPrecisionPlugin, NLPDDPStrategy, PipelineMixedPrecisionPlugin
from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
from nemo.collections.nlp.models.language_modeling.megatron_glue_model import MegatronT5GLUEModel
from nemo.collections.nlp.models.language_modeling.megatron_fused_retro import MegatronFusedRetrievalAdapterModel
from nemo.collections.common.parts import adapter_modules
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager


@hydra_runner(config_path="conf", config_name="retro_gpt_adapter_tuning_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_o2 = cfg.model.get('megatron_amp_O2', False)
    # with_distributed_adam = cfg.model.optim.get('name') == 'distributed_fused_adam'
    plugins = []
    # strategy = NLPDDPStrategy(
    #     no_ddp_communication_hook=True if megatron_amp_o2 else False,
    #     gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
    #     find_unused_parameters=False,
    # )

    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=False,
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

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)
    exp_manager(trainer, cfg.exp_manager)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    save_restore_connector = NLPSaveRestoreConnector()
    if os.path.isdir(cfg.get('restore_from_path')):
        save_restore_connector.model_extracted_dir = cfg.get('restore_from_path')
    frozen_model_cfg = MegatronFusedRetrievalAdapterModel.restore_from(
        cfg.get('restore_from_path'), trainer=trainer, return_config=True, save_restore_connector=save_restore_connector,
    )

    frozen_model_cfg.tokenizer = cfg.model.tokenizer
    frozen_model_cfg.data = cfg.model.data
    frozen_model_cfg.adapter_tuning = cfg.model.adapter_tuning
    frozen_model_cfg.optim = cfg.model.optim
    frozen_model_cfg.restore_from_path = cfg.model.restore_from_path
    frozen_model_cfg.eval = cfg.model.eval
    frozen_model_cfg.add_position_embedding = cfg.model.add_position_embedding
    frozen_model_cfg.micro_batch_size = cfg.model.micro_batch_size
    frozen_model_cfg.precision = trainer.precision

    frozen_model_cfg.task_templates = cfg["model"]["task_templates"]

    if "shape_file" in frozen_model_cfg:
        frozen_model_cfg.pop("shape_file")
    # frozen_model_cfg.tensor_model_parallel_size = 4

    model = MegatronFusedRetrievalAdapterModel(frozen_model_cfg, trainer)
    # model = MegatronRetrievalModel(cfg.model, trainer)
    trainer.fit(model)


if __name__ == '__main__':
    main()
