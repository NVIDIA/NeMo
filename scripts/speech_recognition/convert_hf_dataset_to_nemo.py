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

import os
from dataclasses import dataclass, is_dataclass
from typing import Optional
from omegaconf import OmegaConf, open_dict
import hydra
from hydra.core.config_store import ConfigStore

from datasets import load_dataset, Dataset, IterableDataset


@dataclass
class HFDatasetConvertionConfig:
    path: str
    name: Optional[str] = None
    split: Optional[str] = None

    output_dir: str = "."
    resolved_output_dir: str = ''
    split_output_dir: str = None


def prepare_output_dirs(cfg: HFDatasetConvertionConfig):
    output_dir = os.path.abspath(cfg.output_dir)
    output_dir = os.path.join(output_dir, cfg.path)

    if cfg.name is not None:
        output_dir = os.path.join(output_dir, cfg.name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cfg.resolved_output_dir = output_dir


def process_dataset(dataset: IterableDataset, cfg: HFDatasetConvertionConfig):
    pass



@hydra.main(config_name='hfds_config', config_path=None)
def main(cfg: HFDatasetConvertionConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    prepare_output_dirs(cfg)

    # Load dataset in streaming mode
    dataset = load_dataset(
        path=cfg.path, name=cfg.name, split=cfg.split, cache_dir=cfg.resolved_output_dir, streaming=True
    )

    if isinstance(dataset, dict):
        print("Multiple splits found for dataset", cfg.path, ":", list(dataset.keys()))

    else:
        print("Single split found for dataset", cfg.path)

        process_dataset(dataset, cfg)


ConfigStore.instance().store(name='hfds_config', node=HFDatasetConvertionConfig)

if __name__ == '__main__':
    cfg = HFDatasetConvertionConfig(
        path='timit_asr', name=None, split='train', output_dir='/media/smajumdar/data/Datasets/Timit'
    )

    main(cfg)
