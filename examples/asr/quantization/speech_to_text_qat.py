# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from multiprocessing.sharedctypes import Value
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

@hydra_runner(config_path="conf", config_name="config")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecCTCModel(cfg=cfg.model, trainer=trainer)
    quantized_model = cfg.get("quantized_model", None)
    if not quantized_model:
        raise ValueError('quantized_model must be provided')
        return
    asr_model_cfg = EncDecCTCModel.restore_from(quantized_model, return_config=True)
    with open_dict(asr_model_cfg):
        asr_model_cfg.encoder.quantize = True

    asr_model = EncDecCTCModel.restore_from(restore_path=quantized_model, override_config_path=asr_model_cfg)

    asr_model.setup_optimization(optim_config=cfg.model.optim)
    asr_model.setup_training_data(train_data_config=cfg.model.train_ds)
    asr_model.setup_validation_data(val_data_config=cfg.model.validation_ds)

    trainer.fit(asr_model)
    save_model = cfg.get("save_model", "qat_model.nemo")
    asr_model.save_to(save_model)


if __name__ == "__main__":
    main()
