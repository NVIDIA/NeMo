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

"""
This script demonstrates how to use pretrained .nemo speaker recognition model to finetune on your domain set
Usage:
python speaker_reco_finetune.py --pretrained_model='/path/to/.nemo_or_.ckpt/file' --finetune_config_file=/path/to/finetune/config/yaml/file'

in finetune config file make sure to change
- train manifest
- validation manifest
- decoder num classes
- learning rate
- num of epochs

for finetuning tips see: https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Recognition_Verification.ipynb
"""


from argparse import ArgumentParser

import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained_model", type=str, default="ecapa_tdnn", required=False, help="Pass your trained .nemo model",
    )
    parser.add_argument(
        "--finetune_config_file",
        type=str,
        required=True,
        help="path to speakernet config yaml file to load train, validation dataset and also for trainer parameters",
    )

    parser.add_argument(
        "--freeze_encoder",
        type=bool,
        required=False,
        default=True,
        help="True if speakernet encoder paramteres needs to be frozen while finetuning",
    )

    args = parser.parse_args()

    if args.pretrained_model.endswith('.nemo'):
        logging.info(f"Using local speaker model from {args.pretrained_model}")
        speaker_model = EncDecSpeakerLabelModel.restore_from(restore_path=args.pretrained_model)
    elif args.pretrained_model.endswith('.ckpt'):
        logging.info(f"Using local speaker model from checkpoint {args.pretrained_model}")
        speaker_model = EncDecSpeakerLabelModel.load_from_checkpoint(checkpoint_path=args.pretrained_model)
    else:
        logging.info("Using pretrained speaker recognition model from NGC")
        speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name=args.pretrained_model)

    finetune_config = OmegaConf.load(args.finetune_config_file)

    if 'test_ds' in finetune_config.model and finetune_config.model.test_ds is not None:
        finetune_config.model.test_ds = None
        logging.warning("Removing test ds")

    speaker_model.setup_finetune_model(finetune_config.model)
    finetune_trainer = pl.Trainer(**finetune_config.trainer)
    speaker_model.set_trainer(finetune_trainer)

    _ = exp_manager(finetune_trainer, finetune_config.get('exp_manager', None))
    speaker_model.setup_optimization(finetune_config.optim)

    if args.freeze_encoder:
        for param in speaker_model.encoder.parameters():
            param.requires_grad = False

    finetune_trainer.fit(speaker_model)


if __name__ == '__main__':
    main()
