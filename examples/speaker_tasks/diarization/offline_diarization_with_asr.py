# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import shutil

from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map
from nemo.core.config import hydra_runner
from nemo.utils import logging


"""
This script demonstrates how to run offline speaker diarization with asr.
Usage:
python offline_diarization_with_asr.py \
    diarizer.manifest_filepath=<path to manifest file> \
    diarizer.out_dir='demo_asr_output' \
    diarizer.speaker_embeddings.model_path=<pretrained modelname or path to .nemo> \
    diarizer.asr.model_path=<pretrained modelname or path to .nemo> \
    diarizer.asr.parameters.asr_based_vad=True

Check out whole parameters in ./conf/offline_diarization_with_asr.yaml and their meanings.
For details, have a look at <NeMo_git_root>/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb

Currently Supported ASR models:

QuartzNet15x5Base-En
stt_en_conformer_ctc_large
"""


@hydra_runner(config_path="conf", config_name="offline_diarization_with_asr.yaml")
def main(cfg):

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    asr_diar_offline = ASR_DIAR_OFFLINE(**cfg.diarizer.asr.parameters)
    asr_diar_offline.root_path = cfg.diarizer.out_dir
    shutil.rmtree(asr_diar_offline.root_path, ignore_errors=True)

    AUDIO_RTTM_MAP = audio_rttm_map(cfg.diarizer.manifest_filepath)
    asr_diar_offline.AUDIO_RTTM_MAP = AUDIO_RTTM_MAP
    asr_model = asr_diar_offline.set_asr_model(cfg.diarizer.asr.model_path)

    word_list, word_ts_list = asr_diar_offline.run_ASR(asr_model)

    score = asr_diar_offline.run_diarization(cfg, word_ts_list,)
    total_riva_dict = asr_diar_offline.write_json_and_transcript(word_list, word_ts_list)

    if score is not None:
        metric, mapping_dict = score
        DER_result_dict = asr_diar_offline.gather_eval_results(metric, mapping_dict)

        WDER_dict = asr_diar_offline.get_WDER(total_riva_dict, DER_result_dict)
        effective_wder = asr_diar_offline.get_effective_WDER(DER_result_dict, WDER_dict)

        logging.info(
            f"\nDER  : {DER_result_dict['total']['DER']:.4f} \
            \nFA   : {DER_result_dict['total']['FA']:.4f} \
            \nMISS : {DER_result_dict['total']['MISS']:.4f} \
            \nCER  : {DER_result_dict['total']['CER']:.4f} \
            \nWDER : {WDER_dict['total']:.4f} \
            \neffective WDER : {effective_wder:.4f} \
            \nspk_counting_acc : {DER_result_dict['total']['spk_counting_acc']:.4f}"
        )


if __name__ == '__main__':
    main()
