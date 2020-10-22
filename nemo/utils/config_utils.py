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

import copy
from dataclasses import is_dataclass

from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.core.config.modelPT import ModelPTConfig


def update_model_config(model_cls: ModelPTConfig, update_cfg: DictConfig):
    if not (is_dataclass(model_cls) or isinstance(model_cls, DictConfig)):
        raise ValueError("`model_cfg` must be a dataclass or a structured OmegaConf object")

    if not isinstance(update_cfg, DictConfig):
        update_cfg = OmegaConf.create(update_cfg)

    if is_dataclass(model_cls):
        model_cls = OmegaConf.structured(model_cls)

    # Update optional configs
    model_cls = _update_subconfig(model_cls, update_cfg, subconfig_key='train_ds')
    model_cls = _update_subconfig(model_cls, update_cfg, subconfig_key='validation_ds')
    model_cls = _update_subconfig(model_cls, update_cfg, subconfig_key='test_ds')
    model_cls = _update_subconfig(model_cls, update_cfg, subconfig_key='optim')

    # Add optim and sched additional keys to model cls
    model_cls = _add_subconfig_keys(model_cls, update_cfg, subconfig_key='optim')

    # Perform full merge of model config class and update config
    # Remove ModelPT artifact `target`
    if 'target' in update_cfg.model:
        # Assume artifact from ModelPT and pop
        if 'target' not in model_cls.model:
            with open_dict(update_cfg.model):
                update_cfg.model.pop('target')

    model_cfg = OmegaConf.merge(model_cls, update_cfg)

    return model_cfg


def _update_subconfig(model_cfg: DictConfig, update_cfg: DictConfig, subconfig_key: str):
    with open_dict(model_cfg.model):
        # If update config has the key, but model cfg doesnt have the key
        # Add the update cfg subconfig to the model cfg
        if subconfig_key in update_cfg.model and subconfig_key not in model_cfg.model:
            model_cfg.model[subconfig_key] = update_cfg.model[subconfig_key]

        # If update config does not the key, but model cfg has the key
        # Remove the model cfg subconfig in order to match layout of update cfg
        if subconfig_key not in update_cfg.model and subconfig_key in model_cfg.model:
            model_cfg.model.pop(subconfig_key)

    return model_cfg


def _add_subconfig_keys(model_cfg: DictConfig, update_cfg: DictConfig, subconfig_key: str):
    with open_dict(model_cfg.model):
        # Create copy of original model sub config

        if subconfig_key in update_cfg.model:
            if subconfig_key not in model_cfg.model:
                # create the key as a placeholder
                model_cfg.model[subconfig_key] = None

            subconfig = copy.deepcopy(model_cfg.model[subconfig_key])

            # Add the keys and update temporary values, will be updated during full merge
            subconfig = OmegaConf.merge(update_cfg.model[subconfig_key], subconfig)
            # Update sub config
            model_cfg.model[subconfig_key] = subconfig

    return model_cfg
