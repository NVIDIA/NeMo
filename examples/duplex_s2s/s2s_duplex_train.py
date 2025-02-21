# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import hydra
import torch
from lightning.pytorch import Trainer

from nemo.collections.duplex_s2s.data.datamodule import S2SDataModule
from nemo.collections.duplex_s2s.models.duplex_s2s_model import DuplexS2SModel


# During the training, the checkpoint format is standard PTL ckpt
# After the training -> convert to HF instead of .nemo ?
# Add a callback that does the above conversion at every checkpoint save


@hydra.main(config_path="../speechlm/conf", config_name="config")
def train(cfg):
    # TODO: decide on exp_manager or adopting NeMo 2.0 API with _setup function, or sth else ?
    # cfg = DuplexS2SModelConfig()
    model = DuplexS2SModel(cfg.model)
    # exp_manager / NeMo2 _setup provide:
    # * PEFT (possibly from HF)
    # * save/load checkpoint (exp_manager -> .nemo only)
    # * resume
    # * W&B loggers etc
    trainer = Trainer(**cfg.trainer)
    datamodule = S2SDataModule(cfg.data)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
