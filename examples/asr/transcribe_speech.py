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

import contextlib
import glob
import os
from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import MISSING, OmegaConf

from nemo.collections.asr.metrics.rnnt_wer import RNNTDecodingConfig
from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils


"""
# Transcribe audio

python transcribe_speech.py \
    model_path=null \
    pretrained_name=null \
    audio_dir=""

"""


@dataclass
class TranscriptionConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: str = MISSING  # Path to a directory which contains audio files

    # General configs
    output_filename: str = "speech_to_text_transcriptions.txt"
    batch_size: int = 32
    cuda: Optional[bool] = None  # will switch to cuda if available, defaults to cpu otherwise
    amp: bool = False
    audio_type: str = "wav"

    # decoding strategy for RNNT models
    rnnt_decoding: RNNTDecodingConfig = RNNTDecodingConfig()


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None !")

    # setup gpu
    if cfg.cuda is None:
        cfg.cuda = torch.cuda.is_available()

    if type(cfg.cuda) == int:
        device_id = int(cfg.cuda)
    else:
        device_id = 0

    device = torch.device(f'cuda:{device_id}' if cfg.cuda else 'cpu')

    # setup model
    if cfg.model_path is not None:
        # restore model from .nemo file path
        model_cfg = ASRModel.restore_from(restore_path=cfg.model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info(f"Restoring model : {imported_class.__name__}")

        asr_model = imported_class.restore_from(restore_path=cfg.model_path, map_location=device)  # type: ASRModel
    else:
        # restore model by name
        asr_model = ASRModel.from_pretrained(model_name=cfg.pretrained_name, map_location=device)  # type: ASRModel

    trainer = pl.Trainer(gpus=int(cfg.cuda))
    asr_model.set_trainer(trainer)
    asr_model = asr_model.eval()

    # Setup decoding strategy
    if hasattr(asr_model, 'change_decoding_strategy'):
        asr_model.change_decoding_strategy(cfg.rnnt_decoding)

    # load paths to audio
    filepaths = list(glob.glob(os.path.join(cfg.audio_dir, f"*.{cfg.audio_type}")))
    logging.info(f"\nTranscribing {len(filepaths)} files...\n")

    # setup AMP (optional)
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast():
            yield

    # transcribe audio
    with autocast():
        with torch.no_grad():
            transcriptions = asr_model.transcribe(filepaths, batch_size=cfg.batch_size)
    logging.info(f"Finished transcribing {len(filepaths)} files !")

    logging.info(f"Writing transcriptions into file: {cfg.output_filename}")
    with open(cfg.output_filename, 'w', encoding='utf-8') as f:
        for line in transcriptions:
            f.write(f"{line}\n")

    logging.info("Finished writing predictions !")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
