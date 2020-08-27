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
import os
from omegaconf import DictConfig

from nemo.collections.nlp.models.question_answering.qa_model import QAModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config: {cfg.pretty()}')
    trainer = pl.Trainer(**cfg.trainer)
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    infer_datasets = [cfg.model.validation_ds, cfg.model.test_ds]
    for infer_dataset in infer_datasets:
        if infer_dataset.output_prediction_file is not None: 
            infer_dataset.output_prediction_file = os.path.join(log_dir, infer_dataset.output_prediction_file)
        if infer_dataset.output_nbest_file is not None: 
            infer_dataset.output_nbest_file = os.path.join(log_dir, infer_dataset.output_nbest_file)
    
    question_answering_model = QAModel(cfg.model, trainer=trainer)
    trainer.fit(question_answering_model)
    if cfg.model.nemo_path:
        question_answering_model.save_to(cfg.model.nemo_path)
    
    trainer.test(question_answering_model, ckpt_path=None)


if __name__ == '__main__':
    main()
