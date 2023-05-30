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

seed_everything(42)

# no need for arguments
# for diarizing, write arguments in input_path and num_speakers
# if number of speakers is not known (oracle_num_speakers not known): num_speakers = None
# otherwise provide a natural number

@hydra_runner(config_path="../conf/inference", config_name="diar_infer_meeting.yaml")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    
    cfg.diarizer.speaker_embeddings.model_path = "nemo_models/titanet-l.nemo"
    cfg.diarizer.vad.model_path = "nemo_models/vad_multilingual_marblenet.nemo"
    cfg.diarizer.speaker_embeddings.parameters.save_embeddings=False
    

    input_path = '/home/gabi/NOSFE_Gojira_Podcast_16khz_mono.wav'
    num_speakers = None 

    sd_model = LongClusteringDiarizer(cfg=cfg)
    sd_model.diarize(input_path= input_path,
                     num_speakers = num_speakers)


if __name__ == '__main__':    
    main()
