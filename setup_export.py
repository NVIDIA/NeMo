# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


"""Setup for pip package for export module."""
import setuptools


def setup_export():
    setuptools.setup(
        name="NeMo Export",
        # Versions should comply with PEP440.  For a discussion on single-sourcing
        # the version across setup.py and the project code, see
        # https://packaging.python.org/en/latest/single_source_version.html
        version=1,
        description="NeMo Export - a module to export nemo checkpoints to TensorRT-LLM",
        long_description="NeMo Export - a module to export nemo checkpoints to TensorRT-LLM",
        # Author details
        author="NVIDIA",
        license='Apache2',
        packages=[
            "nemo",
            "nemo.export",
            "nemo.export.trt_llm",
            "nemo.export.trt_llm.decoder",
            "nemo.export.trt_llm.nemo",
            "nemo.deploy",
        ],
        # Add in any packaged data.
        include_package_data=True,
        exclude=['tools', 'tests'],
        package_data={'': ['*.tsv', '*.txt', '*.far', '*.fst', '*.cpp', 'Makefile']},
        zip_safe=False,
    )


if __name__ == '__main__':
    setup_export()

