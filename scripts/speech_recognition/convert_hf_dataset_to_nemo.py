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

from dataclasses import dataclass, is_dataclass
from typing import Optional
from omegaconf import OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class HFDatasetConvertionConfig:
    pass


@hydra.main(config_name='hfds_config', config_path=None)
def main(cfg: HFDatasetConvertionConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    print(cfg)


ConfigStore.instance().store(name='hfds_config', node=HFDatasetConvertionConfig)

if __name__ == '__main__':
    main()
