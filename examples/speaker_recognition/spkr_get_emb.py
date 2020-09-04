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

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import ExtractSpeakerEmbeddingsModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

"""
To extract embeddings
    python examples/speaker_recognition/spkr_get_emb.py \
        model.test_ds.manifest_filepath="<validation_manifest_file>" \
        exp_manager.exp_name="<trained_model_name>"
        exp_manager.exp_dir="<path to model chckpoint directories>"
        hydra.run.dir="." \
        trainer.gpus=1 
"""

seed_everything(42)


@hydra_runner(config_path="conf", config_name="config")
def main(cfg):

    logging.info(f'Hydra config: {cfg.pretty()}')
    if cfg.trainer.gpus > 1:
        logging.info("changing gpus to 1 to minimize DDP issues while extracting embeddings")
        cfg.trainer.gpus = 1
        cfg.trainer.distributed_backend = None
    trainer = pl.Trainer(**cfg.trainer)
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    model_path = os.path.join(log_dir, '..', 'spkr.nemo')
    speaker_model = ExtractSpeakerEmbeddingsModel.restore_from(model_path)
    speaker_model.setup_test_data(cfg.model.test_ds)
    trainer.test(speaker_model)


if __name__ == '__main__':
    main()
