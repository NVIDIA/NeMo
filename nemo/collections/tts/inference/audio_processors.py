# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from typing import Tuple

from nemo.collections.tts.inference.pipeline import AudioProcessor
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.core.classes.common import typecheck, Typing
from nemo.core.neural_types.elements import (
    AudioSignal,
    LengthsType,
    MelSpectrogramType,
)
from nemo.core.neural_types.neural_type import NeuralType


class MelSpectrogramProcessor(AudioProcessor, Typing):

    def __init__(self, preprocessor: AudioToMelSpectrogramPreprocessor):
        self.preprocessor = preprocessor

    @typecheck(
        input_types={
            "audio": NeuralType(('B', 'T'), AudioSignal()),
            "audio_len": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "spec": NeuralType(('B', 'C', 'T'), MelSpectrogramType()),
            "spec_len": NeuralType(tuple('B'), LengthsType()),
        },
    )
    def compute_spectrogram(self, audio: torch.tensor, audio_len: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        spec, spec_len = self.preprocessor(input_signal=audio, length=audio_len)
        return spec, spec_len
