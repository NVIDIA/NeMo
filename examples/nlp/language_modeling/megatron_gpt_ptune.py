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

from pathlib import Path

from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.plugins.environments.torchelastic_environment import TorchElasticEnvironment
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_ptune_gpt_model import MegatronGPTPTuneModel
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.parts.nlp_overrides import GradScaler, NLPDDPPlugin
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager
import pathlib
import torch
import os


@hydra_runner(config_path="conf", config_name="megatron_ptune_gpt")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    plugins = [NLPDDPPlugin(num_nodes=cfg.trainer.num_nodes)]
    if cfg.trainer.precision == 16:
        scaler = GradScaler(
            init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
            growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
        )
        plugins.append(NativeMixedPrecisionPlugin(precision=16, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, **cfg.trainer)

    exp_manager(trainer, cfg.exp_manager)

    # update resume from checkpoint found by exp_manager
    resume_from_checkpoint = trainer.checkpoint_connector.resume_from_checkpoint_fit_path
    if resume_from_checkpoint is not None:
        # inject mp_rank into resume_from_checkpoint
        if cfg.model.tensor_model_parallel_size is not None and cfg.model.tensor_model_parallel_size > 1:
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

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision
    model = MegatronGPTPTuneModel(cfg.model, trainer)
    trainer.fit(model)

    if cfg.model.data.test_ds.file_path:
        logging.info("===========================================================================================")
        logging.info("Starting the testing of the trained model on test set...")
        trainer.test(model)
        logging.info("Testing finished!")
        logging.info("===========================================================================================")
        # extract the path of the best checkpoint from the training, you may update it to any checkpoint
        checkpoint_path = trainer.checkpoint_callback.best_model_path
        tensor_parallel_size = cfg.model.tensor_model_parallel_size
        pathobj = pathlib.Path(checkpoint_path)
        checkpoint_folder = str(pathobj.parent)
        checkpoint_name = str(pathobj.name)

        rank = trainer.accelerator.training_type_plugin.local_rank
        if tensor_parallel_size > 1:
            # inject model parallel rank
            checkpoint_path = os.path.join(checkpoint_folder, f'mp_rank_{rank:02d}', checkpoint_name)
        else:
            checkpoint_path = os.path.join(checkpoint_folder, checkpoint_name)

        # Load the checkpoint
        best_eval_model = MegatronGPTPTuneModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path, strict=False, trainer=trainer
        )
        logging.info(f'Best checkpoint path: {checkpoint_path}')
        logging.info("Running Test with best EVAL checkpoint!")
        # setup the test dataset
        #  best_eval_model.setup_test_data(test_data_config=cfg.model.data.test_ds)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        trainer.test(model=best_eval_model, ckpt_path=None, verbose=False)
        logging.info("Beset EVAL Testing finished!")
        logging.info("===========================================================================================")



if __name__ == '__main__':
    main()
