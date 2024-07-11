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

"""Entry point, main file to run to launch jobs with the HP tool."""

import math

import hydra
import omegaconf
from autoconfig.search_config import search_config
from autoconfig.utils import convert_to_cli
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver(
    "divide_ceil", lambda x, y: int(math.ceil(x / y)), replace=True
)
OmegaConf.register_new_resolver("divide_floor", lambda x, y: int(x // y), replace=True)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: omegaconf.dictconfig.DictConfig) -> None:
    """
    Main function in the entire pipeline, it reads the config using hydra and calls search_config.
    :param omegaconf.dictconfig.DictConfig cfg: OmegaConf object, read using the @hydra.main decorator.
    :return: None
    """
    hydra_args = convert_to_cli(cfg)
    search_config(cfg=cfg, hydra_args=hydra_args)


if __name__ == "__main__":
    main()
