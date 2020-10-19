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
from omegaconf.listconfig import ListConfig
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import ExtractSpeakerEmbeddingsModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

"""
To extract embeddings
Place pretrained model in ${EXP_DIR}/${EXP_NAME} with spkr.nemo
    python spkr_get_emb.py --config-path='conf' --config-name='SpeakerNet_verification_3x2x512.yaml' \
        +model.test_ds.manifest_filepath="<test_manifest_file>" \
        +model.test_ds.sample_rate=16000 \
        +model.test_ds.labels=null \
        +model.test_ds.batch_size=1 \
        +model.test_ds.shuffle=False \
        +model.test_ds.time_length=8 \
        exp_manager.exp_name=${EXP_NAME} \
        exp_manager.exp_dir=${EXP_DIR} \
        trainer.gpus=1

See https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_recognition/Speaker_Recognition_Verification.ipynb for notebook tutorial
"""

seed_everything(42)


@hydra_runner(config_path="conf", config_name="config")
def main(cfg):

    logging.info(f'Hydra config: {cfg.pretty()}')
    if (isinstance(cfg.trainer.gpus, ListConfig) and len(cfg.trainer.gpus) > 1) or (
        isinstance(cfg.trainer.gpus, (int, str)) and int(cfg.trainer.gpus) > 1
    ):
        logging.info("changing gpus to 1 to minimize DDP issues while extracting embeddings")
        cfg.trainer.gpus = 1
        cfg.trainer.accelerator = None
    trainer = pl.Trainer(**cfg.trainer)
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    model_path = os.path.join(log_dir, '..', 'spkr.nemo')
    speaker_model = ExtractSpeakerEmbeddingsModel.restore_from(model_path)
    speaker_model.setup_test_data(cfg.model.test_ds)
    trainer.test(speaker_model)


if __name__ == '__main__':
    main()
