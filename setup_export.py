# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import importlib.util
import os
import setuptools

spec = importlib.util.spec_from_file_location('package_info', 'nemo/package_info.py')
package_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_info)


__contact_emails__ = package_info.__contact_emails__
__contact_names__ = package_info.__contact_names__
__download_url__ = package_info.__download_url__
__license__ = package_info.__license__
__repository_url__ = package_info.__repository_url__
__version__ = package_info.__version__


def req_file(filename, folder="requirements"):
    with open(os.path.join(folder, filename), encoding='utf-8') as f:
        content = f.readlines()
    return [x.strip() for x in content]


setuptools.setup(
    name="nemo_export",
    version=__version__,
    description="NeMo Export - a package to export NeMo checkpoints to TensorRT-LLM",
    long_description="NeMo Export - a package to export NeMo checkpoints to TensorRT-LLM",
    url=__repository_url__,
    download_url=__download_url__,
    author=__contact_names__,
    maintainer=__contact_names__,
    maintainer_email=__contact_emails__,
    license=__license__,
    packages=setuptools.find_packages(include=["nemo", "nemo.export*", "nemo.deploy"]),
    install_requires=req_file("requirements_infer_nim.txt"),
    zip_safe=False,
)
