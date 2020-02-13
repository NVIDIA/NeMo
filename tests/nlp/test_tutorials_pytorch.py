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

from nemo import logging
from nemo.backends.pytorch.tutorials.chatbot.data import loadPrepareData
from tests.common_setup import NeMoUnitTest


class TestPytorchChatBotTutorial(NeMoUnitTest):
    def test_simple_train(self):
        datafile = "tests/data/dialog_sample.txt"
        logging.info(datafile)
        voc, pairs = loadPrepareData("cornell", datafile=datafile)
        self.assertEqual(voc.name, 'cornell')
        self.assertEqual(voc.num_words, 675)
