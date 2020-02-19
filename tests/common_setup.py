# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

import unittest

import nemo

logging = nemo.logging


class NeMoUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        """ Default setup - instantiates Neural Factory. """
        # Initialize the default Neural Factory - on GPU.
        self.nf = nemo.core.NeuralModuleFactory(placement=nemo.core.DeviceType.GPU)

        # Print standard header.
        logging.info("-" * 20 + " " + type(self).__name__ + "." + self._testMethodName + " " + "-" * 20)
