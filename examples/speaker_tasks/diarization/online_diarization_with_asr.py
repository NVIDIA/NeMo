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


from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE, ASR_DIAR_ONLINE, write_txt
from nemo.collections.asr.models import OnlineDiarizer


@hydra_runner(config_path="conf", config_name="online_diarization_with_asr.yaml")
def main(cfg):
    diar = OnlineDiarizer(cfg)
    asr_diar = ASR_DIAR_ONLINE(diar, cfg=cfg.diarizer)

    if cfg.diarizer.asr.parameters.streaming_simulation:
        diar.uniq_id = cfg.diarizer.simulation_uniq_id
        asr_diar.get_audio_rttm_map(diar.uniq_id)
        diar.single_audio_file_path = diar.AUDIO_RTTM_MAP[diar.uniq_id]['audio_filepath']
        diar.rttm_file_path = diar.AUDIO_RTTM_MAP[diar.uniq_id]['rttm_filepath']
        asr_diar.rttm_file_path = diar.rttm_file_path
    else:
        diar.rttm_file_path = None

    diar._init_segment_variables()
    diar.device = asr_diar.device
    write_txt(f"{diar._out_dir}/print_script.sh", "")
    samplerate, sdata = wavfile.read(diar.single_audio_file_path)

    for index in range(int(np.floor(sdata.shape[0]/asr_diar.n_frame_len))):
        asr_diar.buffer_counter = index
        sample_audio = sdata[asr_diar.CHUNK_SIZE*(asr_diar.buffer_counter):asr_diar.CHUNK_SIZE*(asr_diar.buffer_counter+1)]
        asr_diar.callback_sim(sample_audio)

if __name__ == "__main__":
    main()

