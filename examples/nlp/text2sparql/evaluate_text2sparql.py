# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2020, MeetKai Inc.  All rights reserved.
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

import pytorch_lightning as pl
from omegaconf import DictConfig

from nemo.collections.nlp.models.neural_machine_translation import NeuralMachineTranslationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf", config_name="text2sparql_config")
def main(cfg: DictConfig) -> None:
    logging.info(f"Config:\n {cfg.pretty()}")
    trainer = pl.Trainer(gpus=cfg.trainer.gpus)
    nmt_model = NeuralMachineTranslationModel.restore_from(restore_path=cfg.model.nemo_path)
    nmt_model.setup_test_data(cfg.model.test_ds)
    results = trainer.test(nmt_model)

    with open(cfg.model.test_ds.filepath, "r") as f:
        lines = f.readlines()

    lines[0] = lines[0].strip() + f"\tpredictions\n"
    for i, res in enumerate(results[0]["texts"]):
        lines[i + 1] = lines[i + 1].strip() + f"\t{res}\n"

    savepath = os.path.join(cfg.exp_manager.exp_dir, os.path.basename(cfg.model.test_ds.filepath))
    with open(savepath, "w") as f:
        f.writelines(lines)
        logging.info(f"Predictions saved to {savepath}")


if __name__ == "__main__":
    main()
