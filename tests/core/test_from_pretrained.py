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

import importlib
import inspect
import pytest

from nemo.core.classes import ModelPT
from nemo.utils.model_utils import resolve_subclass_pretrained_model_info


# @pytest.mark.with_downloads()
class TestFromPretrained:
    def build_model_import_path(self, domain):
        import_path = ['nemo', 'collections', domain, 'models']

        import_path = ".".join(import_path)

        return import_path

    def get_class_members_from_module(self, module):
        # classes = []
        class_members = inspect.getmembers(module, inspect.isclass)
        module_members = inspect.getmembers(module, inspect.ismodule)
        for module_member in module_members:
            import_path = module_member[1]
            new_module = importlib.import_module(import_path)
            class_members.append(self.get_class_members_from_module(new_module))
        return class_members

    @pytest.mark.parametrize('domain', ['asr', 'nlp', 'tts'])
    def test_from_pretrained(self, domain):

        import_path = self.build_model_import_path(domain)

        module = importlib.import_module(import_path)

        classes = self.get_class_members_from_module(module)
        # models = {}
        # for name, object in domain_models_module.__dict__.items():
        #     if inspect.isclass(object):
        #         # list_of_models = resolve_subclass_pretrained_model_info(object)
        #         # models[object] = list_of_models
        #         print(f'class: {object}')

        # for model in models:
        #     print(f'model: {model} list_of_models: {list_of_models}')
        assert 1 == 1

        # find all models

