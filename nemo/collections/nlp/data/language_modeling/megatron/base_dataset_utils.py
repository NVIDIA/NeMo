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

import math
import re
import os
from jinja2 import Environment, Template, exceptions
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.nodes import Name

def get_datasets_weights_and_num_samples(data_prefix, num_samples):

    # The data prefix should be in the format of:
    #   weight-1, data-prefix-1, weight-2, data-prefix-2, ..
    assert len(data_prefix) % 2 == 0
    num_datasets = len(data_prefix) // 2
    weights = [0] * num_datasets
    prefixes = [0] * num_datasets
    for i in range(num_datasets):
        weights[i] = float(data_prefix[2 * i])
        prefixes[i] = (data_prefix[2 * i + 1]).strip()
    # Normalize weights
    weight_sum = 0.0
    for weight in weights:
        weight_sum += weight
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]

    # Add 0.5% (the 1.005 factor) so in case the bleding dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    # TODO: check data leakage between train/val/test?
    datasets_train_valid_test_num_samples = []
    for weight in weights:
        # Comes here when we have seperate train,test and validation datasets.
        if isinstance(num_samples, int):
            datasets_train_valid_test_num_samples.append(int(math.ceil(num_samples * weight * 1.005)))
        else:
            datasets_train_valid_test_num_samples.append([int(math.ceil(val * weight * 1.005)) for val in num_samples])

    return prefixes, weights, datasets_train_valid_test_num_samples


def get_train_valid_test_split_(splits_string, size):
    """ Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(',') != -1:
        splits = [float(s) for s in splits_string.split(',')]
    elif splits_string.find('/') != -1:
        splits = [float(s) for s in splits_string.split('/')]
    else:
        splits = [float(splits_string)]
    if len(splits) != 3:
        raise ValueError(f"Invalid splits string: {splits_string}. Expected 3 comma separated values.")
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


class JinjaTemplating:
    def __init__(self):
        pass

    def is_jinja_template(self, template):
        # Check using regex for Jinja patterns
        if re.search(r'\{\{.*\}\}|\{%.*%\}|\{#.*#\}', self.template):
            return True

        # Try parsing with Jinja
        try:
            Template(template).render()
            return True
        except exceptions.TemplateSyntaxError:
            return False

    def load_prompt_file(self, file_path):
        # Check if the input is a valid file path
        if not os.path.isfile(file_path):
            print(f"{file_path} is not a valid file path.")
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except IOError as e:
            print(f"Error reading file: {e}")
            return None

    def find_template_variables(self, template_string):
        env = Environment()
        ast = env.parse(template_string)
        variable_names = {node.name for node in ast.find_all(Name)}
        return variable_names

    def compile_template(self, template):
        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined)
        jinja_env.globals["raise_exception"] = lambda message: TemplateError(message)
        # Check if target_template is template or path to template
        if os.path.exists(template):
            with open(template, 'r') as f:
                template = f.read()
        return jinja_env.from_string(template)


    def apply_template(self, template, data) -> str:
        return template.render(**data)