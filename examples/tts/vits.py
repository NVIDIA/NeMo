# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from pytorch_lightning.plugins import DDPPlugin

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models.vits import Vits
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="vits")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = Vits(cfg=cfg.model, trainer=trainer)
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)

    """
    load_checkpoint = True
    if load_checkpoint:
        print('Loading from checkpoint')
        model, _, _, _ = utils.load_checkpoint("vits_lightning.ckpt", model)

    hps = utils.get_hparams()
    collate_fn = TextAudioCollate()
    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
    train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.train.batch_size,
            [32, 300, 400, 500, 600, 700, 800, 900, 1000],
            shuffle=True)
    
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True, collate_fn=collate_fn, batch_sampler=train_sampler)
    eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False, batch_size=hps.train.batch_size, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    
    trainer = Trainer(gpus=1, max_epochs=1)
    ljspeech = VITSDataModule()

    trainer.fit(model, train_loader, eval_loader)
    trainer.save_checkpoint("vits_lightning.ckpt")
    """


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
