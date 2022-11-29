# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.parts.utils.eval_utils import run_asr_inference

# from nemo.collections.asr.data.data_simulation import MultiSpeakerSimulator, RIRMultiSpeakerSimulator
from nemo.core.config import hydra_runner
from nemo.utils import logging


"""
This script serves as evaluator of ASR models
Usage:
  python python asr_evaluator.py \
asr_eval.pretrained_name="stt_en_conformer_transducer_large" \
asr_eval.inference_mode.mode="offline" \
.....

Check out parameters in ./conf/eval.yaml
"""


@hydra_runner(config_path="conf", config_name="eval.yaml")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    cfg = run_asr_inference(cfg.asr_eval)


if __name__ == "__main__":
    main()
