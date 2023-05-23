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

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import LongClusteringDiarizer
from nemo.core.config import hydra_runner
from nemo.utils import logging

import torch

"""
This script demonstrates how to use run speaker diarization.
Usage:
  python offline_diar_infer.py \
    diarizer.manifest_filepath=<path to manifest file> \
    diarizer.out_dir='demo_output' \
    diarizer.speaker_embeddings.model_path=<pretrained modelname or path to .nemo> \
    diarizer.vad.model_path='vad_marblenet' \
    diarizer.speaker_embeddings.parameters.save_embeddings=False

Check out whole parameters in ./conf/offline_diarization.yaml and their meanings.
For details, have a look at <NeMo_git_root>/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb
"""
seed_everything(42)

@hydra_runner(config_path="../conf/inference", config_name="diar_infer_meeting.yaml")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    
    cfg.diarizer.speaker_embeddings.model_path = "nemo_models/titanet-l.nemo"
    cfg.diarizer.vad.model_path = "nemo_models/vad_multilingual_marblenet.nemo"
    cfg.diarizer.speaker_embeddings.parameters.save_embeddings=False
    

    input_path = '/home/gabi/git_repos/2023.04.27.08.11.36_aaa.wav'
    num_speakers = 2

    sd_model = LongClusteringDiarizer(cfg=cfg)
    sd_model.diarize(input_path= input_path)


if __name__ == '__main__':    
    main()
