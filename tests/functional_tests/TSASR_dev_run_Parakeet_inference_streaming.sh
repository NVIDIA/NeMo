# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/asr_chunked_inference/ctc/speech_to_text_tgt_spk_buffered_infer_ctc.py \
    model_path=/home/TestData/an4_tsasr/FastConformer-Hybrid-Transducer-TGT-CTC-BPE.nemo \
    dataset_manifest=/home/TestData/an4_tsasr/simulated_valid/tsasr_valid_tiny.json \
    output_filename=pred_text.json \
    chunk_len_in_secs=1.6 \
    total_buffer_in_secs=4.0 \
    model_stride=8 \
    override=True \
    rttm_mix_prob=0 \
    diar_model_path=/home/TestData/an4_tsasr/diar_sortformer_4spk-v1-tiny.nemo \
    