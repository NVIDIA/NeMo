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

from torch import Tensor

from nemo.collections.tts.inference.pipeline import SpectrogramSynthesizer
from nemo.collections.tts.models import FastPitchModel
from nemo.core.classes.common import typecheck, Typing
from nemo.core.neural_types.elements import (
    Index,
    MelSpectrogramType,
    RegressionValuesType,
    TokenIndex,
)
from nemo.core.neural_types.neural_type import NeuralType


class FastPitchSpectrogramSynthesizer(SpectrogramSynthesizer, Typing):
    def __init__(self, model: FastPitchModel):
        self.model = model

    @property
    def device(self):
        return self.model.device

    @typecheck(
        input_types={
            "tokens": NeuralType(('B', 'T_text'), TokenIndex()),
            "speaker": NeuralType(('B'), Index(), optional=True),
            "pitch": NeuralType(('B', 'T_text'), RegressionValuesType(), optional=True),
        },
        output_types={"spec": NeuralType(('B', 'C', 'T'), MelSpectrogramType())},
    )
    def synthesize_spectrogram(self, tokens: Tensor, speaker: Tensor, pitch: Tensor) -> Tensor:
        spec, *_ = self.model.fastpitch.infer(text=tokens, speaker=speaker, pitch=pitch)
        return spec
