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

import hydra
import pytorch_lightning as pl

from nemo.collections.asr.models import EncDecCTCModel
from nemo.utils import logging

"""
Basic run:
    python speech_to_text.py \
        AudioToTextDataLayer.manifest_filepath="/path/to/an4/train_manifest.json" \
        AudioToTextDataLayer_eval.manifest_filepath="/path/to/an4/test_manifest.json" \
        hydra.run.dir="." \
        pl.trainer.gpus=2 \
        pl.trainer.max_epochs=100

Add PyTorch Lightning Trainer arguments from CLI:
    python speech_to_text.py \
        ... \
        +pl.trainer.fast_dev_run=true

Hydra logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/.hydra)"
PTL logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/lightning_logs)"
"""


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    logging.info(f'Hydra config: {cfg.pretty()}')

    asr_model = EncDecCTCModel(
        preprocessor_config=cfg.preprocessor,
        encoder_config=cfg.encoder,
        decoder_config=cfg.decoder,
        spec_augment_config=cfg.spec_augment,
    )

    asr_model.setup_training_data(cfg.AudioToTextDataLayer)
    asr_model.setup_validation_data(cfg.AudioToTextDataLayer_eval)

    # Setup optimizer and scheduler
    if cfg.pl.trainer.max_steps is None:
        if cfg.pl.trainer.gpus == 0:
            # training on CPU
            iters_per_batch = cfg.pl.trainer.max_epochs / float(cfg.pl.trainer.num_nodes * cfg.accumulate_grad_batches)
        else:
            iters_per_batch = cfg.pl.trainer.max_epochs / float(
                cfg.pl.trainer.gpus * cfg.pl.trainer.num_nodes * cfg.accumulate_grad_batches
            )
        cfg.lr_scheduler.iters_per_batch = iters_per_batch
    else:
        cfg.lr_scheduler.max_steps = cfg.pl.trainer.max_steps

    asr_model.setup_optimization(cfg.optim_params)
    # TODO: Fix scheduler
    # asr_model.setup_lr_scheduler(cfg.lr_scheduler)

    trainer = pl.Trainer(**cfg.pl.trainer)
    trainer.fit(asr_model)


if __name__ == '__main__':
    main()
