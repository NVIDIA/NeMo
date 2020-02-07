# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2019 NVIDIA. All Rights Reserved.
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

import json
import os
import shutil

from examples.nlp.scripts.get_squad import SquadDownloader

import nemo
import nemo.collections.nlp as nemo_nlp
from tests.common_setup import NeMoUnitTest

logging = nemo.logging


class TestSquad(NeMoUnitTest):
    def test_setup_squad(self):
        pretrained_bert_model = 'bert-base-uncased'
        tokenizer = nemo_nlp.data.NemoBertTokenizer(pretrained_bert_model)
        neural_factory = nemo.core.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch, local_rank=None, create_tb_writer=False
        )
        model = nemo_nlp.nm.trainables.huggingface.BERT(pretrained_model_name=pretrained_bert_model)
        hidden_size = model.hidden_size
        qa_head = nemo_nlp.nm.trainables.token_classification_nm.TokenClassifier(
            hidden_size=hidden_size, num_classes=2, num_layers=1, log_softmax=False
        )
        squad_loss = nemo_nlp.nm.losses.QuestionAnsweringLoss()
