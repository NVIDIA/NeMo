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

import codecs
import importlib.util
import os
import subprocess
from distutils import cmd as distutils_cmd
from distutils import log as distutils_log
from itertools import chain

import setuptools

def req_file(filename, folder="/opt/NeMo/requirements"):
    with open(os.path.join(folder, filename), encoding='utf-8') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters
    # Example: `\n` at the end of each line
    return [x.strip() for x in content]


def setup_export():
    
    setuptools.setup(
        name="NeMo Export",
        # Versions should comply with PEP440.  For a discussion on single-sourcing
        # the version across setup.py and the project code, see
        # https://packaging.python.org/en/latest/single_source_version.html
        version="MINOR",
        description="NeMo Export - a module to export nemo checkpoints to TensorRT-LLM",
        long_description="NeMo Export - a module to export nemo checkpoints to TensorRT-LLM",
        # Author details
        author="NVIDIA",
        license='Apache2',
        packages=setuptools.find_packages(where="/opt/NeMo//nemo/export/"),
        install_requires=req_file("requirements_export.txt"),
        # Add in any packaged data.
        include_package_data=True,
        exclude=['tools', 'tests'],
        package_data={'': ['*.tsv', '*.txt', '*.far', '*.fst', '*.cpp', 'Makefile']},
        zip_safe=False,
    )


if __name__ == '__main__':
    setup_export()
