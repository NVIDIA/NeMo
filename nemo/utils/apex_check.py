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


def is_megatron_supported():
    """ Helper function to determine if apex supports megatron
    """
    try:
        import apex
        _apex_installed = True
    except (ImportError, ModuleNotFoundError):
        _apex_installed = False

    try:
        import apex.transformer.utils
        _apex_recent_version = True
    except (ImportError, ModuleNotFoundError):
        _apex_recent_version = False

    return _apex_installed and _apex_recent_version
