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


class deprecated(object):
    """ Decorator class used for indicating that a function is
    depricated and going to be removed.
    Tracks down which functions printed the warning and 
    will print it only once per function. 
    """

    # Static variable - list of names of functions that we already printed
    # the warning for.
    warned_functions = {}

    def __init__(self, version=None, alternative_function=None):
        """
        If there are decorator arguments, the function
        to be decorated is not passed to the constructor!
        """
        self.version = version
        self.alternative_function = alternative_function

    def __call__(self, func):
        """
        If there are decorator arguments, __call__() is only called
        once, as part of the decoration process! You can only give
        it a single argument, which is the function object.
        """

        def wrapper(*args, **kwargs):

            # Prepare the warning message.
            msg = "Function ``{}`` is depricated.".format(func.__name__)

            # Optionally, add version and alternative.
            if self.version is not None:
                msg = msg + \
                    " It is going to be removed in version {}.".format(
                        self.version)

            if self.alternative_function is not None:
                msg = msg + \
                    " Please use ``{}`` instead.".format(
                        self.alternative_function)
            # Display the depricated warning.
            nemo.logging.warning(msg)

            # Call the function.
            func(*args, **kwargs)

        return wrapper
