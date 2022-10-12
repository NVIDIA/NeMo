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
from dataclasses import is_dataclass

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import FastPitchModel
from nemo.core import adapter_mixins
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


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
    exp_log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    n_speakers = cfg.model.n_speakers

    if cfg.model.adapter.add_random_speaker:
        cfg.model.n_speakers += 1

    model = FastPitchModel(cfg=update_model_config_to_support_adapter(cfg.model), trainer=trainer)
    model.fastpitch.speaker_emb = torch.nn.Embedding(n_speakers, model.fastpitch.speaker_emb.embedding_dim)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)

    # Add new speaker embedding
    if cfg.model.adapter.add_random_speaker and model.fastpitch.speaker_emb is not None:
        old_emb = model.fastpitch.speaker_emb

        # Choose random
        new_speaker_emb = torch.rand(1, old_emb.embedding_dim)
        # Choose existing
        # new_speaker_emb = old_emb.weight[0, :].unsqueeze(0).detach().clone()

        new_emb = torch.nn.Embedding(old_emb.num_embeddings + 1, old_emb.embedding_dim).from_pretrained(
            torch.cat([old_emb.weight.detach().clone(), new_speaker_emb], axis=0), freeze=True
        )
        model.fastpitch.speaker_emb = new_emb
        model.cfg.n_speakers += 1

    # Setup adapters
    with open_dict(cfg.model.adapter):
        # Extract the name of the adapter (must be give for training)
        adapter_name = cfg.model.adapter.pop("adapter_name", "adapter")
        adapter_module_name = cfg.model.adapter.pop("adapter_module_name", None)
        adapter_state_dict_name = cfg.model.adapter.pop("adapter_state_dict_name", None)
        cfg.model.adapter.pop("add_random_speaker", None)
        freeze_all = cfg.model.adapter.pop("freeze_all", False)
        freeze_encoder = cfg.model.adapter.pop("freeze_encoder", False)
        freeze_decoder = cfg.model.adapter.pop("freeze_decoder", False)

        # augment adapter name with module name, if not provided by user
        if adapter_module_name is not None and ':' not in adapter_name:
            adapter_name = f'{adapter_module_name}:{adapter_name}'

        # Extract the global adapter config, if provided
        adapter_global_cfg = cfg.model.adapter.pop(model.adapter_global_cfg_key, None)
        if adapter_global_cfg is not None:
            add_global_adapter_cfg(model, adapter_global_cfg)

    # model.add_adapter(name='encoder+decoder+duration_predictor+pitch_predictor+aligner:adapter', cfg=adapter_cfg)
    model.add_adapter(name=adapter_name, cfg=cfg.model.adapter)
    assert model.is_adapter_available()

    model.set_enabled_adapters(enabled=False)
    model.set_enabled_adapters(adapter_name, enabled=True)

    # Freeze
    if freeze_all:
        model.freeze()
    if freeze_encoder:
        model.fastpitch.encoder.freeze()
    if freeze_decoder:
        model.fastpitch.decoder.freeze()

    # Set model to training mode.
    model = model.train()
    # Then, Unfreeze just the adapter weights that were enabled above (no part of model)
    model.unfreeze_enabled_adapters()

    model.summarize()

    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)

    # Save the adapter state dict
    if adapter_state_dict_name is not None:
        state_path = exp_log_dir if exp_log_dir is not None else os.getcwd()
        ckpt_path = os.path.join(state_path, "checkpoints")
        if os.path.exists(ckpt_path):
            state_path = ckpt_path
        state_path = os.path.join(state_path, adapter_state_dict_name)

        # Save the adapter modules in a seperate file
        model.save_adapters(str(state_path))


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
