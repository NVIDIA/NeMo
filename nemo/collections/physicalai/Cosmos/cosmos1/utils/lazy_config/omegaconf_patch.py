# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Union

from omegaconf import OmegaConf
from omegaconf.base import DictKeyType, SCMode
from omegaconf.dictconfig import DictConfig  # pragma: no cover


def to_object(cfg: Any) -> Union[Dict[DictKeyType, Any], List[Any], None, str, Any]:
    """
    Converts an OmegaConf configuration object to a native Python container (dict or list), unless
    the configuration is specifically created by LazyCall, in which case the original configuration
    is returned directly.

    This function serves as a modification of the original `to_object` method from OmegaConf,
    preventing DictConfig objects created by LazyCall from being automatically converted to Python
    dictionaries. This ensures that configurations meant to be lazily evaluated retain their intended
    structure and behavior.

    Differences from OmegaConf's original `to_object`:
    - Adds a check at the beginning to return the configuration unchanged if it is created by LazyCall.

    Reference:
    - Original OmegaConf `to_object` method: https://github.com/omry/omegaconf/blob/master/omegaconf/omegaconf.py#L595

    Args:
        cfg (Any): The OmegaConf configuration object to convert.

    Returns:
        Union[Dict[DictKeyType, Any], List[Any], None, str, Any]: The converted Python container if
        `cfg` is not a LazyCall created configuration, otherwise the unchanged `cfg`.

    Examples:
        >>> cfg = DictConfig({"key": "value", "_target_": "Model"})
        >>> to_object(cfg)
        DictConfig({"key": "value", "_target_": "Model"})

        >>> cfg = DictConfig({"list": [1, 2, 3]})
        >>> to_object(cfg)
        {'list': [1, 2, 3]}
    """
    if isinstance(cfg, DictConfig) and "_target_" in cfg.keys():
        return cfg

    return OmegaConf.to_container(
        cfg=cfg,
        resolve=True,
        throw_on_missing=True,
        enum_to_str=False,
        structured_config_mode=SCMode.INSTANTIATE,
    )
