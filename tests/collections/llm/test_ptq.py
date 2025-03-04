# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.llm.quantization import QuantizationConfig, ExportConfig
from nemo.collections.llm.api import ptq
import pytest

HF_PATH = "/home/TestData/nlp/megatron_llama/llama-ci-hf"
OUTPUT_PATH = '/tmp/quantized_model'


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
def test_ptq():
    """
    Test if PTQ works for a model.
    """

    quantization_config = QuantizationConfig()
    export_config = ExportConfig(path=OUTPUT_PATH)
    ptq(HF_PATH, export_config, quantization_config=quantization_config)

