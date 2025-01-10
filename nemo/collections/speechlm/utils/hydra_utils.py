# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


from typing import Any, Dict, List, Optional, Union

from megatron.core.optimizer import OptimizerConfig
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from nemo import lightning as nl
from nemo.core.classes.common import Serialization
from nemo.lightning import AutoResume


def get_object_list_from_config(cfg: Union[DictConfig, ListConfig, Dict]) -> List[Any]:
    """
    Instantiates a list of objects from a config object.
    Args:
        cfg: Config object (DictConfig or ListConfig).
    Returns:
        List of instantiated objects.
    """
    if isinstance(cfg, (DictConfig, Dict)) and "_target_" not in cfg:
        candidates = list(cfg.values())
    elif isinstance(cfg, (ListConfig, List)):
        candidates = cfg
    elif isinstance(cfg, (DictConfig, Dict)) and "_target_" in cfg:
        candidates = [cfg]
    else:
        raise ValueError(f"Unsupported config type: {type(cfg)}, with content: {cfg}")

    return [Serialization.from_config_dict(c) for c in candidates]


def get_resume_from_config(cfg: DictConfig) -> Optional[AutoResume]:
    if "restore_config" in cfg:
        if "_target_" not in cfg.restore_config:
            with open_dict(cfg):
                cfg.restore_config._target_ = "nemo.lightning.RestoreConfig"
        restore_config = get_object_list_from_config(cfg.restore_config)[0]
    else:
        restore_config = None

    resume = AutoResume(
        restore_config=restore_config,
        resume_from_directory=cfg.get("resume_from_directory", None),
        resume_from_path=cfg.get("resume_from_path", None),
        adapter_path=cfg.get("adapter_path", None),
        resume_if_exists=cfg.get("resume_if_exists", True),
        resume_past_end=cfg.get("resume_past_end", False),
        resume_ignore_no_checkpoint=cfg.get("resume_ignore_no_checkpoint", True),
    )
    return resume


def get_logger_from_config(cfg: DictConfig):
    logger = nl.NeMoLogger(**cfg)
    return logger
