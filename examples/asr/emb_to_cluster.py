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

import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecClusteringModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


"""

"""


@hydra_runner(config_path="conf", config_name="config")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    cluster_model = EncDecClusteringModel(cfg=cfg.model, trainer=trainer)
    with open("new_pre_in.model", "w") as f:
        f.write("new_pre_in")
        f.write(str(cluster_model))

    print(cluster_model)
    num = 0
    for p in cluster_model.parameters():
        num += p.numel()
    print(num)

    trainer.fit(cluster_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        gpu = 1 if cfg.trainer.gpus != 0 else 0
        test_trainer = pl.Trainer(
            gpus=gpu,
            precision=trainer.precision,
            amp_level=trainer.accelerator_connector.amp_level,
            amp_backend=cfg.trainer.get("amp_backend", "native"),
        )
        if cluster_model.prepare_test(test_trainer):
            test_trainer.test(cluster_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
