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
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.models import EncDecDiarLabelModel
from nemo.collections.asr.models.msdd_models import ClusterEmbedding
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

"""
Basic run (on GPU for 10 epochs for 2 class training):
EXP_NAME=sample_run
python ./speaker_reco.py --config-path='conf' --config-name='SpeakerNet_recognition_3x2x512.yaml' \
    trainer.max_epochs=10  \
    model.train_ds.batch_size=64 model.validation_ds.batch_size=64 \
    model.train_ds.manifest_filepath="<train_manifest>" model.validation_ds.manifest_filepath="<dev_manifest>" \
    model.test_ds.manifest_filepath="<test_manifest>" \
    trainer.gpus=1 \
    model.decoder.params.num_classes=2 \
    exp_manager.name=$EXP_NAME +exp_manager.use_datetime_version=False \
    exp_manager.exp_dir='./speaker_exps'

See https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Recognition_Verification.ipynb for notebook tutorial
"""

seed_everything(42)

@hydra_runner(config_path="conf", config_name="msdd_training.telephonic.yaml")
def main(cfg):

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    # if not cfg.msdd_model.end_to_end_train:
        # clustering_embedding = ClusterEmbedding(cfg_base=cfg, cfg_msdd_model=cfg.msdd_model)
        # clustering_embedding.prepare_cluster_embs_train()
        # cfg.msdd_model.base.diarizer.update(cfg.diarizer)
    # import ipdb; ipdb.set_trace()
    # msdd_model = EncDecDiarLabelModel(cfg=cfg, trainer=trainer)
    msdd_model = EncDecDiarLabelModel(cfg=cfg.msdd_model, trainer=trainer)
    trainer.fit(msdd_model)

if __name__ == '__main__':
    main()
