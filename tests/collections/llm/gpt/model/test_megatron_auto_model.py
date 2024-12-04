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


def test_all_models_in_registry():
    from nemo.collections import llm
    from nemo.collections.llm.gpt.model.megatron_auto_model import HF_TO_MCORE_REGISTRY

    model_class_names = list(
        filter(lambda x: 'model' in x.lower() and x[0].isupper() and not 'auto' in x.lower(), dir(llm))
    )
    mcore_models = set(cls.__name__ for cls in HF_TO_MCORE_REGISTRY.values())
    skipped = set(['GPTModel', 'T5Model'])
    for model_class_name in model_class_names:
        if model_class_name in skipped:
            continue
        assert model_class_name in mcore_models, f'Expected {model_class_name} to be in MegatronAutoModel registry'
