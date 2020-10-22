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


def update_model_config(model_cls: ModelPTConfig, update_cfg: DictConfig, drop_missing_subconfigs: bool = True):
    """
    Helper class that updates the default values of a ModelPT config class with the values
    in a DictConfig that mirrors the structure of the config class.

    Assumes the `update_cfg` is a DictConfig (either generated manually, via hydra or instantiated via yaml/model.cfg).
    This update_cfg is then used to override the default values preset inside the ModelPT config class.

    If `drop_missing_subconfigs` is set, the certain sub-configs of the ModelPT config class will be removed, iff
    they are not found in the mirrored `update_cfg`. The following sub-configs are subject to potential removal:
        -   `train_ds`
        -   `validation_ds`
        -   `test_ds`
        -   `optim` + nested `sched`.

    Args:
        model_cls: A subclass of ModelPTConfig, that details in entirety all of the parameters that constitute
            the NeMo Model.

        update_cfg: A DictConfig that mirrors the structure of the ModelPTConfig data class. Used to update the
            default values of the config class.

        drop_missing_subconfigs: Bool which determins whether to drop certain sub-configs from the ModelPTConfig
            class, if the corresponding sub-config is missing from `update_cfg`.

    Returns:
        A DictConfig with updated values that can be used to instantiate the NeMo Model along with supporting
        infrastructure.
    """
    if not (is_dataclass(model_cls) or isinstance(model_cls, DictConfig)):
        raise ValueError("`model_cfg` must be a dataclass or a structured OmegaConf object")

    if not isinstance(update_cfg, DictConfig):
        update_cfg = OmegaConf.create(update_cfg)

    if is_dataclass(model_cls):
        model_cls = OmegaConf.structured(model_cls)

    # Update optional configs
    model_cls = _update_subconfig(
        model_cls, update_cfg, subconfig_key='train_ds', drop_missing_subconfigs=drop_missing_subconfigs
    )
    model_cls = _update_subconfig(
        model_cls, update_cfg, subconfig_key='validation_ds', drop_missing_subconfigs=drop_missing_subconfigs
    )
    model_cls = _update_subconfig(
        model_cls, update_cfg, subconfig_key='test_ds', drop_missing_subconfigs=drop_missing_subconfigs
    )
    model_cls = _update_subconfig(
        model_cls, update_cfg, subconfig_key='optim', drop_missing_subconfigs=drop_missing_subconfigs
    )

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


def _update_subconfig(
    model_cfg: DictConfig, update_cfg: DictConfig, subconfig_key: str, drop_missing_subconfigs: bool
):
    """
    Updates the ModelPTConfig DictConfig such that:
    1)  If the sub-config key exists in the `update_cfg`, but does not exist in ModelPT config:
        - Add the sub-config from update_cfg to ModelPT config

    2) If the sub-config key does not exist in `update_cfg`, but exists in ModelPT config:
        - Remove the sub-config from the ModelPT config; iff the `drop_missing_subconfigs` flag is set.

    Args:
        model_cfg: A DictConfig instantiated from the ModelPTConfig subclass.
        update_cfg: A DictConfig that mirrors the structure of `model_cfg`, used to update its default values.
        subconfig_key: A str key used to check and update the sub-config.
        drop_missing_subconfigs: A bool flag, whether to allow deletion of the ModelPTConfig sub-config,
            if its mirror sub-config does not exist in the `update_cfg`.

    Returns:
        The updated DictConfig for the ModelPTConfig
    """
    with open_dict(model_cfg.model):
        # If update config has the key, but model cfg doesnt have the key
        # Add the update cfg subconfig to the model cfg
        if subconfig_key in update_cfg.model and subconfig_key not in model_cfg.model:
            model_cfg.model[subconfig_key] = update_cfg.model[subconfig_key]

        # If update config does not the key, but model cfg has the key
        # Remove the model cfg subconfig in order to match layout of update cfg
        if subconfig_key not in update_cfg.model and subconfig_key in model_cfg.model:
            if drop_missing_subconfigs:
                model_cfg.model.pop(subconfig_key)

    return model_cfg


def _add_subconfig_keys(model_cfg: DictConfig, update_cfg: DictConfig, subconfig_key: str):
    """
    For certain sub-configs, the default values specified by the ModelPTConfig class is insufficient.
    In order to support every potential value in the merge between the `update_cfg`, it would require
    explicit definition of all possible cases.

    An example of such a case is Optimizers, and their equivalent Schedulers. All optimizers share a few basic
    details - such as name and lr, but almost all require additional parameters - such as weight decay.
    It is impractical to create a config for every single optimizer + every single scheduler combination.

    In such a case, we perform a dual merge. The Optim and Sched Dataclass contain the bare minimum essential
    components. The extra values are provided via update_cfg.

    In order to enable the merge, we first need to update the update sub-config to incorporate the keys,
    with dummy temporary values (merge update config with model config). This is done on a copy of the
    update sub-config, as the actual override values might be overriden by the ModelPTConfig defaults.

    Then we perform a merge of this temporary sub-config with the actual override config in a later step
    (merge model_cfg with original update_cfg, done outside this function).

    Args:
        model_cfg: A DictConfig instantiated from the ModelPTConfig subclass.
        update_cfg: A DictConfig that mirrors the structure of `model_cfg`, used to update its default values.
        subconfig_key: A str key used to check and update the sub-config.

    Returns:
        A ModelPT DictConfig with additional keys added to the sub-config.
    """
    with open_dict(model_cfg.model):
        # Create copy of original model sub config
        if subconfig_key in update_cfg.model:
            if subconfig_key not in model_cfg.model:
                # create the key as a placeholder
                model_cfg.model[subconfig_key] = None

            subconfig = copy.deepcopy(model_cfg.model[subconfig_key])
            update_subconfig = copy.deepcopy(update_cfg.model[subconfig_key])

            # Add the keys and update temporary values, will be updated during full merge
            subconfig = OmegaConf.merge(update_subconfig, subconfig)
            # Update sub config
            model_cfg.model[subconfig_key] = subconfig

    return model_cfg
