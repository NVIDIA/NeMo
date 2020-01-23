# Copyright (C) tkornuta, NVIDIA AI Applications Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Tomasz Kornuta"

from io import StringIO
import unittest
from unittest.mock import patch

import nemo


def deprecated_function(func):
    """ Decorator function used for indicating that a function is depricated
    and going to be removed."""
    def wrapper(*args, **kwargs):
        # Display the depricated warning.
        nemo.logging.warning(
            "Function ``{}`` is depricated.".format(func.__name__))
        # Call the function.
        func(*args, **kwargs)
    return wrapper
