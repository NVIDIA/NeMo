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
import glob
import torch
import pytorch_lightning as pl
from typing import Optional
from omegaconf import OmegaConf, MISSING
from dataclasses import dataclass

from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
# Transcribe audio

python speech_to_text_transcribe.py \
    model_path=null \
    pretrained_name=null \
    audio_dir=""

"""


@dataclass
class SpeechToTextTranscribeConfig:
    model_path: Optional[str] = None
    pretrained_name: Optional[str] = None
    audio_dir: str = MISSING
    output_filename: str = "speech_to_text_transcriptions.txt"
    batch_size: int = 32
    cuda: Optional[bool] = None
    audio_type: str = "wav"


@hydra_runner(config_name="SpeechToTextTranscribeConfig", schema=SpeechToTextTranscribeConfig)
def main(cfg: SpeechToTextTranscribeConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None !")

    # setup gpu
    if cfg.cuda is None:
        cfg.cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cfg.cuda else 'cpu')

    # setup model
    if cfg.model_path is not None:
        asr_model = EncDecCTCModel.restore_from(
            restore_path=cfg.model_path, map_location=device
        )  # type: EncDecCTCModel
    else:
        asr_model = EncDecCTCModel.from_pretrained(
            model_name=cfg.pretrained_name, map_location=device
        )  # type: EncDecCTCModel

    trainer = pl.Trainer(gpus=int(cfg.cuda))
    asr_model.set_trainer(trainer)

    # load paths to audio
    filepaths = list(glob.glob(os.path.join(cfg.audio_dir, f"*.{cfg.audio_type}")))
    logging.info(f"\nTranscribing {len(filepaths)} files...\n")

    # transcribe audio
    transcriptions = asr_model.transcribe(filepaths, batch_size=cfg.batch_size)
    logging.info(f"Finished transcribing {len(filepaths)} files !")

    logging.info(f"Writing transcriptions into file: {cfg.output_filename}")
    with open(cfg.output_filename, 'w', encoding='utf-8') as f:
        for line in transcriptions:
            f.write(f"{line}\n")

    logging.info("Finished writing predictions !")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
