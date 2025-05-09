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
import os

import torch
from lightning.pytorch import Callback, Trainer
from omegaconf import OmegaConf, open_dict

from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset
from nemo.collections.speechlm2.parts.precision import HalfPrecisionForAudio
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class PROFILING(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch, batch_idx: int
    ) -> None:
        if batch_idx == 0:
            print("STARTING PROFILE")
            torch.cuda.profiler.cudart().cudaProfilerStart()

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int
    ) -> None:
        if batch_idx == 10:
            print("STOPPING PROFILE")
            torch.cuda.profiler.cudart().cudaProfilerStop()


@hydra_runner(config_path="conf", config_name="salm")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    precision = cfg.trainer.get("precision", "bf16-true")
    with open_dict(cfg):
        cfg.trainer.pop("precision", None)
    trainer = Trainer(
        precision=HalfPrecisionForAudio(precision),
        **resolve_trainer_cfg(cfg.trainer),
        # callbacks=[PROFILING()],
    )
    exp_manager(trainer, cfg.get("exp_manager", None))

    with trainer.init_module():
        model = SALM(OmegaConf.to_container(cfg.model, resolve=True))

    dataset = SALMDataset(tokenizer=model.tokenizer)
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
