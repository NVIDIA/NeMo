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
import torch
import pytorch_lightning as pl

from dataclasses import is_dataclass
from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import FastPitchModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from nemo.core import adapter_mixins
from omegaconf import DictConfig, open_dict, OmegaConf

def update_model_config_to_support_adapter(config) -> DictConfig:
    with open_dict(config):
        enc_adapter_metadata = adapter_mixins.get_registered_adapter(config.input_fft._target_)
        if enc_adapter_metadata is not None:
            config.input_fft._target_ = enc_adapter_metadata.adapter_class_path

        dec_adapter_metadata = adapter_mixins.get_registered_adapter(config.output_fft._target_)
        if dec_adapter_metadata is not None:
            config.output_fft._target_ = dec_adapter_metadata.adapter_class_path

        pitch_predictor_adapter_metadata = adapter_mixins.get_registered_adapter(config.pitch_predictor._target_)
        if pitch_predictor_adapter_metadata is not None:
            config.pitch_predictor._target_ = pitch_predictor_adapter_metadata.adapter_class_path

        duration_predictor_adapter_metadata = adapter_mixins.get_registered_adapter(config.duration_predictor._target_)
        if duration_predictor_adapter_metadata is not None:
            config.duration_predictor._target_ = duration_predictor_adapter_metadata.adapter_class_path

        aligner_adapter_metadata = adapter_mixins.get_registered_adapter(config.alignment_module._target_)
        if aligner_adapter_metadata is not None:
            config.alignment_module._target_ = aligner_adapter_metadata.adapter_class_path

    return config


def add_global_adapter_cfg(model, global_adapter_cfg):
    # Convert to DictConfig from dict or Dataclass
    if is_dataclass(global_adapter_cfg):
        global_adapter_cfg = OmegaConf.structured(global_adapter_cfg)

    if not isinstance(global_adapter_cfg, DictConfig):
        global_adapter_cfg = DictConfig(global_adapter_cfg)

    # Update the model.cfg with information about the new adapter global cfg
    with open_dict(global_adapter_cfg), open_dict(model.cfg):
        if 'adapters' not in model.cfg:
            model.cfg.adapters = OmegaConf.create({})

        # Add the global config for adapters to the model's internal config
        model.cfg.adapters[model.adapter_global_cfg_key] = global_adapter_cfg

        # Update all adapter modules (that already exist) with this global adapter config
        model.update_adapter_cfg(model.cfg.adapters)


@hydra_runner(config_path="conf", config_name="fastpitch_align_44100")
def main(cfg):
    if hasattr(cfg.model.optim, 'sched'):
        logging.warning("You are using an optimizer scheduler while finetuning. Are you sure this is intended?")
    if cfg.model.optim.lr > 1e-3 or cfg.model.optim.lr < 1e-5:
        logging.warning("The recommended learning rate for finetuning is 2e-4")

    trainer = pl.Trainer(**cfg.trainer)
    n_speakers = cfg.model.n_speakers

    model = FastPitchModel(cfg=update_model_config_to_support_adapter(cfg.model), trainer=trainer)
    model.fastpitch.speaker_emb = torch.nn.Embedding(n_speakers, model.fastpitch.speaker_emb.embedding_dim)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)


    # Setup adapters
    with open_dict(cfg.model.adapter):
        # Extract the name of the adapter (must be give for training)
        adapter_name = cfg.model.adapter.pop("adapter_name", "adapter")
        adapter_module_name = cfg.model.adapter.pop("adapter_module_name", None)
        adapter_state_dict_name = cfg.model.adapter.pop("adapter_state_dict_name", None)
        cfg.model.adapter.pop("add_random_speaker", None)

        # augment adapter name with module name, if not provided by user
        if adapter_module_name is not None and ':' not in adapter_name:
            adapter_name = f'{adapter_module_name}:{adapter_name}'

        # Extract the global adapter config, if provided
        adapter_global_cfg = cfg.model.adapter.pop(model.adapter_global_cfg_key, None)
        if adapter_global_cfg is not None:
            add_global_adapter_cfg(model, adapter_global_cfg)


    model.load_adapters(adapter_state_dict_name, name=None, map_location='cpu')
    
    # Freeze 
    model.freeze()

    # Set model to eval mode.
    model = model.eval()
    # Then, Unfreeze just the adapter weights that were enabled above (no part of model)
    model.unfreeze_enabled_adapters()

    model.summarize()

    model.get_enabled_adapters()

    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)



if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
