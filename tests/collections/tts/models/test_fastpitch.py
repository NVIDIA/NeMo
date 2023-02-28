# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
This file implemented unit tests for loading all pretrained FastPitch NGC checkpoints and generating Mel-spectrograms.
The test duration breakdowns are shown below. In general, each test for a single model is ~25 seconds on an NVIDIA RTX A6000.
"""
import pytest

from nemo.collections.tts.models import FastPitchModel

available_models = [model.pretrained_model_name for model in FastPitchModel.list_available_models()]


@pytest.fixture(params=available_models, ids=available_models)
@pytest.mark.run_only_on('GPU')
def pretrained_model(request):
    model_name = request.param
    model = FastPitchModel.from_pretrained(model_name=model_name)
    return model, model_name


@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
def test_inference(pretrained_model, language_specific_text_example):
    model, model_name = pretrained_model
    language_id = model_name.split("_")[1]
    if language_id not in ["en", "de", "es", "zh"]:
        pytest.fail(f"The name of model {model_name} does not follow the format, `tts_language_model_*`.")
    else:
        text = language_specific_text_example[language_id]
    parsed_text = model.parse(text)
    _ = model.generate_spectrogram(tokens=parsed_text)
