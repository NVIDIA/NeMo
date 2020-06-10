# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.collections.asr import models
from nemo.collections.asr.audio_preprocessing import *
from nemo.collections.asr.beam_search_decoder import BeamSearchDecoderWithLM
from nemo.collections.asr.contextnet import ContextNetDecoderForCTC, ContextNetEncoder
from nemo.collections.asr.data_layer import (
    AudioToSpeechLabelDataLayer,
    AudioToTextDataLayer,
    KaldiFeatureDataLayer,
    TarredAudioToTextDataLayer,
    TranscriptDataLayer,
)
from nemo.collections.asr.greedy_ctc_decoder import GreedyCTCDecoder
from nemo.collections.asr.jasper import (
    JasperDecoderForClassification,
    JasperDecoderForCTC,
    JasperDecoderForSpkrClass,
    JasperEncoder,
)
from nemo.collections.asr.las.misc import JasperRNNConnector
from nemo.collections.asr.losses import CTCLossNM

__all__ = [
    'AudioToTextDataLayer',
    'TarredAudioToTextDataLayer',
    'AudioToSpeechLabelDataLayer',
    'AudioPreprocessing',
    'AudioPreprocessor',
    'AudioToMFCCPreprocessor',
    'AudioToMelSpectrogramPreprocessor',
    'AudioToSpectrogramPreprocessor',
    'CropOrPadSpectrogramAugmentation',
    'MultiplyBatch',
    'SpectrogramAugmentation',
    'KaldiFeatureDataLayer',
    'TranscriptDataLayer',
    'GreedyCTCDecoder',
    'BeamSearchDecoderWithLM',
    'JasperEncoder',
    'JasperDecoderForCTC',
    'JasperDecoderForClassification',
    'JasperDecoderForSpkrClass',
    'JasperRNNConnector',
    'ContextNetEncoder',
    'ContextNetDecoderForCTC',
    'CTCLossNM',
    'CrossEntropyLossNM',
]
