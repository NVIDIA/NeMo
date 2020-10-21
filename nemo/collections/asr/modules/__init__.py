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

from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor,
    AudioToMFCCPreprocessor,
    CropOrPadSpectrogramAugmentation,
    SpectrogramAugmentation,
)
from nemo.collections.asr.modules.conv_asr import (
    ConvASRDecoder,
    ConvASRDecoderClassification,
    ConvASREncoder,
    SpeakerDecoder,
)

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.modules.lstm_decoder import LSTMDecoder

from nemo.collections.asr.modules.activations import Swish
from nemo.collections.asr.modules.conformer_modules import ConformerConvolution, ConformerFeedForward

from nemo.collections.asr.modules.subsampling import ConvSubsampling
from nemo.collections.asr.modules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttention_old,
    RelPositionalEncoding_old,
    RelPositionalEncoding,
    PositionalEncoding,
)