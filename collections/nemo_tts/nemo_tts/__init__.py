# Copyright 2019 AI Applications Design Team at NVIDIA. All Rights Reserved.
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
# ==============================================================================
from nemo.core import Backend

from .tacotron2_modules import (MakeGate, Tacotron2Loss, Tacotron2Postnet,
                                Tacotron2Decoder, Tacotron2DecoderInfer,
                                Tacotron2Encoder, TextEmbedding)
from .waveglow_modules import WaveGlowNM, WaveGlowInferNM, WaveGlowLoss
from .data_layers import AudioDataLayer
from .parts.helpers import (waveglow_log_to_tb_func,
                            waveglow_process_eval_batch,
                            waveglow_eval_log_to_tb_func,
                            tacotron2_log_to_tb_func,
                            tacotron2_process_eval_batch,
                            tacotron2_process_final_eval,
                            tacotron2_eval_log_to_tb_func)

name = "nemo_tts"
backend = Backend.PyTorch
__version__ = "0.9.0"
