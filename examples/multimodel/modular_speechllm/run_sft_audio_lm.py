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

import os
import tempfile

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

from nemo.collections.asr.models import ASRModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import AppState, logging, model_utils
from nemo.utils.exp_manager import exp_manager
from nemo.utils.model_utils import inject_model_parallel_rank
from nemo.collections.multimodal.speechllm.models.speechllm_models import ModularAudioGPTLoRAModel

mp.set_start_method("spawn", force=True)

"""
This is the script to train an Adapter infused GPT Model for audio question answering.
A base GPT Model is required as a starting point. This script will then insert
Adapters into each Transformer layer and will train/update only these adapters
during training. The base GPT Model weights will remain frozen.

During training this script will only save the newly trained Adapter weights
in checkpoints. At the end of training a .nemo file of Adapter weights will 
be saved.

Usage:
    Assuming the base model is a 125m GPT Model, with TP=1, PP=1:
    a. run a training run for a base gpt nemo file:
        python megatron_gpt_adapter_tuning.py \
            model.data.train_ds=[PATH TO TRAINING JSONL FILE], \
            model.data.validation_ds=[PATH TO VALIDATION JSONL FILE]",\
            model.pretrained_audio_model="PATH TO ASR MODEL (.nemo FILE or NGC MODEL NAME)" \
            model.restore_from_path="PATH TO BASE GPT MODEL .nemo FILE" \
            name="NAME OF TRAINING RUN" \
            exp_manager.exp_dir="DIR TO SAVE CHECKPOINTS and .nemo FILE" \
            trainer.max_epochs=2
"""


@hydra_runner(config_path="conf", config_name="megatron_gpt_peft_tuning_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_o2 = cfg.model.get('megatron_amp_O2', False)
    with_distributed_adam = cfg.model.optim.get('name') == 'distributed_fused_adam'

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)
    # update resume from checkpoint found by exp_manager
    logging.info(f'Resuming training from checkpoint: {trainer.ckpt_path}')
    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    if hasattr(cfg, 'model_target'):
        imported_cls = model_utils.import_class_by_path(cfg.model_target)
    else:
        imported_cls = ModularAudioGPTLoRAModel
    model = imported_cls.restore_from_pretrained_models(cfg, trainer=trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()
