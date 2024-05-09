# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from collections import OrderedDict
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import EncDecSpeakerLabelModel, SpeechEncDecSelfSupervisedModel
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
    trainer.devices=1 \
    model.decoder.params.num_classes=2 \
    exp_manager.name=$EXP_NAME +exp_manager.use_datetime_version=False \
    exp_manager.exp_dir='./speaker_exps'

See https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Identification_Verification.ipynb for notebook tutorial

Optional: Use tarred dataset to speech up data loading.
   Prepare ONE manifest that contains all training data you would like to include. Validation should use non-tarred dataset.
   Note that it's possible that tarred datasets impacts validation scores because it drop values in order to have same amount of files per tarfile; 
   Scores might be off since some data is missing. 
   
   Use the `convert_to_tarred_audio_dataset.py` script under <NEMO_ROOT>/speech_recognition/scripts in order to prepare tarred audio dataset.
   For details, please see TarredAudioToClassificationLabelDataset in <NEMO_ROOT>/nemo/collections/asr/data/audio_to_label.py
"""

seed_everything(42)


def load_ssl_encoder(model, cfg):
    if cfg.get("init_from_ptl_ckpt", None) is not None:
        state_dict = torch.load(cfg.init_from_ptl_ckpt, map_location='cpu')['state_dict']
        logging.info(f"Loading encoder from PyTorch Lightning checkpoint: {cfg.init_from_ptl_ckpt}")
    elif cfg.get("init_from_nemo_model", None) is not None:
        ssl_model = SpeechEncDecSelfSupervisedModel.restore_from(cfg.init_from_nemo_model, map_location='cpu')
        state_dict = ssl_model.state_dict()
        logging.info(f"Loading encoder from NeMo model: {cfg.init_from_nemo_model}")
    elif cfg.get("init_from_pretrained_model", None) is not None:
        ssl_model = SpeechEncDecSelfSupervisedModel.from_pretrained(cfg.init_from_pretrained_model, map_location='cpu')
        state_dict = ssl_model.state_dict()
        logging.info(f"Loading encoder from pretrained model: {cfg.init_from_pretrained_model}")
    else:
        logging.info("No model checkpoint or pretrained model specified for encoder initialization.")
        return model

    encoder_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('encoder.'):
            encoder_state_dict[f'preprocessor.feature_extractor.{key}'] = value

    model.load_state_dict(encoder_state_dict, strict=False)
    logging.info("Loaded ssl encoder state dict.")

    return model


@hydra_runner(config_path="configs", config_name="ecapa_tdnn_ssl")
def main(cfg):

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    speaker_model = EncDecSpeakerLabelModel(cfg=cfg.model, trainer=trainer)

    if cfg.model.preprocessor.get("feature_extractor", None) is not None:
        speaker_model = load_ssl_encoder(speaker_model, cfg)

    # save labels to file
    if log_dir is not None:
        with open(os.path.join(log_dir, 'labels.txt'), 'w') as f:
            if speaker_model.labels is not None:
                for label in speaker_model.labels:
                    f.write(f'{label}\n')

    trainer.fit(speaker_model)

    if not trainer.fast_dev_run:
        model_path = os.path.join(log_dir, '..', 'spkr.nemo')
        speaker_model.save_to(model_path)

    torch.distributed.destroy_process_group()
    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if trainer.is_global_zero:
            trainer = pl.Trainer(devices=1, accelerator=cfg.trainer.accelerator, strategy=cfg.trainer.strategy)
            if speaker_model.prepare_test(trainer):
                trainer.test(speaker_model)


if __name__ == '__main__':
    main()
