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
from nemo.collections.asr.parts.vad_utils import vad_tune_threshold_on_dev
import numpy as np 

thresholds = np.arange(0.7,1,0.1)
vad_pred_dir = "/home/fjia/code/NeMo-fei/examples/speaker_recognition/outputs_150/diarization/vad_outputs/overlap_smoothing_output_median_0.875"
groundtruth_RTTM_dir="/home/fjia/data/modified_callhome/RTTMS/CHI109/" 


if __name__ == "__main__":
    best_threhsold = vad_tune_threshold_on_dev(thresholds, vad_pred_dir, groundtruth_RTTM_dir)
    print(best_threhsold)