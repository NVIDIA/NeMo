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

import argparse

from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE, get_file_lists
from nemo.utils import logging

"""
Currently Supported ASR models:

QuartzNet15x5Base

"""

CONFIG_URL = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/speaker_diarization.yaml"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained_speaker_model",
    type=str,
    help="Fullpath of the Speaker embedding extractor model (*.nemo).",
    required=True,
)
parser.add_argument(
    "--audiofile_list_path", type=str, help="Fullpath of a file contains the list of audio files", required=True
)
parser.add_argument(
    "--reference_rttmfile_list_path", default=None, type=str, help="Fullpath of a file contains the list of rttm files"
)
parser.add_argument("--oracle_vad_manifest", default=None, type=str, help="External VAD file for diarization")
parser.add_argument("--oracle_num_speakers", help="Either int or text file that contains number of speakers")
parser.add_argument("--threshold", default=50, type=int, help="Threshold for ASR based VAD")
parser.add_argument(
    "--diar_config_url", default=CONFIG_URL, type=str, help="Config yaml file for running speaker diarization"
)
parser.add_argument("--csv", default='result.csv', type=str, help="")
args = parser.parse_args()


params = {
    "time_stride": 0.02,  # This should not be changed if you are using QuartzNet15x5Base.
    "offset": -0.18,  # This should not be changed if you are using QuartzNet15x5Base.
    "round_float": 2,
    "window_length_in_sec": 1.5,
    "shift_length_in_sec": 0.75,
    "print_transcript": False,
    "lenient_overlap_WDER": True,
    "threshold": args.threshold,  # minimun width to consider non-speech activity
    "external_oracle_vad": True if args.oracle_vad_manifest else False,
    "diar_config_url": args.diar_config_url,
    "ASR_model_name": 'QuartzNet15x5Base-En',
}

asr_diar_offline = ASR_DIAR_OFFLINE(params)

asr_model = asr_diar_offline.set_asr_model(params['ASR_model_name'])

asr_diar_offline.create_directories()

audio_file_list = get_file_lists(args.audiofile_list_path)

transcript_logits_list = asr_diar_offline.run_ASR(asr_model, audio_file_list)

word_list, spaces_list, word_ts_list = asr_diar_offline.get_speech_labels_list(transcript_logits_list, audio_file_list)

oracle_manifest = (
    asr_diar_offline.write_VAD_rttm(asr_diar_offline.oracle_vad_dir, audio_file_list)
    if not args.oracle_vad_manifest
    else args.oracle_vad_manifest
)

asr_diar_offline.run_diarization(
    audio_file_list, oracle_manifest, args.oracle_num_speakers, args.pretrained_speaker_model
)

if args.reference_rttmfile_list_path:

    ref_rttm_file_list = get_file_lists(args.reference_rttmfile_list_path)

    diar_labels, ref_labels_list, DER_result_dict = asr_diar_offline.eval_diarization(
        audio_file_list, ref_rttm_file_list
    )

    logging.info(
        f"\nDER  : {DER_result_dict['total']['DER']:.4f} \
          \nFA   : {DER_result_dict['total']['FA']:.4f} \
          \nMISS : {DER_result_dict['total']['MISS']:.4f} \
          \nCER  : {DER_result_dict['total']['CER']:.4f} \
          \nspk_counting_acc : {DER_result_dict['total']['spk_counting_acc']:.4f}"
    )
else:
    diar_labels = asr_diar_offline.get_diarization_labels(audio_file_list)

total_riva_dict = asr_diar_offline.write_json_and_transcript(audio_file_list, diar_labels, word_list, word_ts_list,)
