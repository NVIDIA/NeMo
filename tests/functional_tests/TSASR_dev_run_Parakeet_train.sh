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
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_tgt_spk_bpe.py \
    model.train_ds.manifest_filepath=/home/TestData/an4_tsasr/simulated_train/tsasr_train_tiny.json \
    model.validation_ds.manifest_filepath=/home/TestData/an4_tsasr/simulated_valid/tsasr_valid_tiny.json \
    model.test_ds.manifest_filepath=/home/TestData/an4_tsasr/simulated_valid/tsasr_valid_tiny.json \
    model.tokenizer.dir=/home/TestData/an4_tsasr/tokenizer_bpe_asr_phase1_en_v1024_beep \
    model.tokenizer.type=bpe \
    +model.diar_model_path=/home/TestData/an4_tsasr/diar_sortformer_4spk-v1-tiny.nemo