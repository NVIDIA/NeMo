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

"""
# Adapting the model

python train_ctc_adapter.py \
    --config-path="../conf/" \
    --config-name="adapt_ctc.yaml" \
    pretrained_model=null \
    nemo_model=null \
    model.adapter.adapter_name=<Unique adapter name> \
    model.adapter.in_features=<dimension of the layer outputs of the model> \
    model.adapter.dim=32 \
    model.train_ds.manifest_filepath=<Path to manifest> \
    model.train_ds.batch_size=16 \
    model.validation_ds.manifest_filepath=<Path to manifest> \
    model.validation_ds.batch_size=16 \
    model.optim.lr=0.01 \
    model.optim.weight_decay=0.0 \
    model.optim.sched.warmup_steps=1000 \
    trainer.max_steps=250 \
    trainer.devices=1 \
    exp_manager.exp_dir=<Some directory for experiment manager>

# Fine-tune a model

For documentation on fine-tuning this model, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations

# Pretrained Models

For documentation on existing pretrained models, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/results.html

"""

"""

python train_ctc_adapter.py \
    --config-path="../conf/" \
    --config-name="adapt_ctc.yaml" \
    model.pretrained_model=null \
    model.nemo_model="/home/smajumdar/PycharmProjects/nemo-eval/nemo_beta_eval/mls/pretrained/Conformer-RNNT-SPE/stt_en_conformer_transducer_large_mls.nemo" \
    model.adapter.adapter_name="tedlium" \
    model.adapter.in_features=512 \
    model.adapter.dim=32 \
    model.adapter.norm_position=post \
    model.train_ds.manifest_filepath=/home/smajumdar/PycharmProjects/nemo-eval/nemo_beta_eval/tedlium/tedlium_v2/manifests/manifest_dev.json \
    model.train_ds.batch_size=16 \
    model.validation_ds.manifest_filepath="/home/smajumdar/PycharmProjects/nemo-eval/nemo_beta_eval/tedlium/tedlium_v2/manifests/manifest_test.json" \
    model.validation_ds.batch_size=16 \
    model.optim.lr=0.5 \
    model.optim.weight_decay=0 \
    model.optim.sched.warmup_steps=100 \
    trainer.max_steps=300 \
    trainer.devices=1 \
    trainer.check_val_every_n_epoch=50 \
    exp_manager.create_wandb_logger=true \
    exp_manager.wandb_logger_kwargs.name="Conformer-MLS-Adapt-Tedlium" \
    exp_manager.wandb_logger_kwargs.project="Adapters-Local"

"""

import pytorch_lightning as pl

import os
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def update_encoder_config_to_support_adapter(cfg):
    if 'Adapter' not in cfg.encoder._target_:
        cfg.encoder._target_ = cfg.encoder._target_ + 'Adapter'


def update_model_cfg(original_cfg, new_cfg):
    with open_dict(new_cfg):
        # drop keys which dont exist in old config
        new_keys = list(new_cfg.keys())
        for key in new_keys:
            if key not in original_cfg:
                new_cfg.pop(key)

        # print("Original config :", OmegaConf.to_yaml(original_cfg))
        new_cfg = OmegaConf.merge(original_cfg, new_cfg)
        # print("Merged Config :", OmegaConf.to_yaml(new_cfg))
    return new_cfg


@hydra_runner(config_path="../conf", config_name="adapt_ctc.yaml")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if cfg.model.pretrained_model is None and cfg.model.nemo_model is None:
        raise ValueError("Either set `cfg.model.nemo_model` or `cfg.model.pretrained_model`")
    if cfg.model.pretrained_model is not None and cfg.model.nemo_model is not None:
        raise ValueError("Cannot set `cfg.model.nemo_model` and `cfg.model.pretrained_model`. Select one only.")

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if cfg.model.pretrained_model is not None:
        model_cfg = ASRModel.from_pretrained(cfg.model.pretrained_model, return_config=True)
        update_encoder_config_to_support_adapter(model_cfg)
        model = ASRModel.from_pretrained(cfg.model.pretrained_model, override_config_path=model_cfg, trainer=trainer)

    else:
        model_cfg = ASRModel.restore_from(cfg.model.nemo_model, return_config=True)
        update_encoder_config_to_support_adapter(model_cfg)
        model = ASRModel.restore_from(cfg.model.nemo_model, override_config_path=model_cfg, trainer=trainer)

    # Setup model for finetuning (train and validation only)
    cfg.model.train_ds = update_model_cfg(model.cfg.train_ds, cfg.model.train_ds)
    cfg.model.validation_ds = update_model_cfg(model.cfg.validation_ds, cfg.model.validation_ds)
    cfg.model.optim = update_model_cfg(model.cfg.optim, cfg.model.optim)

    # Call the dataloaders and optimizer + scheduler

    # extra support for train_ds
    # TODO: Fix pretrained checkpoints to not have this set to True by default.
    cfg.model.train_ds.is_tarred = False
    model.setup_training_data(cfg.model.train_ds)
    model.setup_validation_data(cfg.model.validation_ds)
    model.setup_optimization(cfg.model.optim)

    # Setup adapters
    with open_dict(cfg.model.adapter):
        adapter_name = cfg.model.adapter.pop("adapter_name")
    model.add_adapter(adapter_name, cfg=cfg.model.adapter)

    assert model.is_adapter_available()

    # Disable all other adapters, enable just the current adapter.
    model.set_enabled_adapters(enabled=False)  # disable all adapters prior to training
    model.set_enabled_adapters(adapter_name, enabled=True)  # enable just one adapter by name

    # First, Freeze all the weights of the model (not just encoder, everything)
    model.freeze()
    # Activate dropout() and other modules that depend on train mode.
    model.train()
    # Then, Unfreeze just the adapter weights that were enabled above (no part of encoder/decoder/joint/etc)
    model.unfreeze_enabled_adapters()

    # Finally, train model
    trainer.fit(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
