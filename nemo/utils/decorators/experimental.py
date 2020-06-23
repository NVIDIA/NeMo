# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

__all__ = ['experimental']

from nemo.utils import logging


def experimental(cls):
    """ Decorator which indicates that module is experimental.
    Use it to mark experimental or research modules.
    """

    def wrapped(cls):
        class WrappedClass(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                logging.warning(
                    f'Module {cls} is experimental, not ready for production and is not supported. Use at your own risk.'
                )

        return WrappedClass

    return wrapped(cls=cls)
