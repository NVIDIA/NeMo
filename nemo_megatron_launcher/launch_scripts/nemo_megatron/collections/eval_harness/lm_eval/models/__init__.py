# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from lm_eval.base import LM

from . import dummy, nemo_gpt3, nemo_gpt3_prompt

MODEL_REGISTRY = {
    "nemo-gpt3": nemo_gpt3.NeMo_GPT3LM_TP_PP,
    "nemo-gpt3-prompt": nemo_gpt3_prompt.NeMo_GPT3_PROMPTLM,
    "dummy": dummy.DummyLM,
}


def get_model(model_name: str) -> LM:
    return MODEL_REGISTRY[model_name]
