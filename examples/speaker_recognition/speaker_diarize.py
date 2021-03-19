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

from pytorch_lightning import seed_everything

from nemo.collections.asr.models import ClusteringDiarizer
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
This script demonstrates how to use run speaker diarization.
Usage:
  python speaker_diarize.py \
  diarizer.paths2audio_files=<either list of audio file paths or file containing paths to audio files> \
  diarizer.path2groundtruth_rttm_files=<(Optional) either list of rttm file paths or file containing paths to rttm files> \
  diarizer.vad.model_path="vad_telephony_marblenet" \
  diarizer.vad.threshold=0.7 \
  diarizer.speaker_embeddings.model_path="speakerverification_speakernet" \
  diarizer.out_dir="demo"

Check out whole parameters in ./conf/speaker_diarization.yaml and their meanings.
For details, have a look at <NeMo_git_root>/tutorials/speaker_recognition/Speaker_Diarization_Inference.ipynb
"""

seed_everything(42)


@hydra_runner(config_path="conf", config_name="speaker_diarization.yaml")
def main(cfg):

    logging.info(f'Hydra config: {cfg.pretty()}')
    sd_model = ClusteringDiarizer(cfg=cfg)
    sd_model.diarize()


if __name__ == '__main__':
    main()
