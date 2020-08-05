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

import os
import tempfile
from unittest import TestCase

import pytest

import nemo.collections.nlp as nemo_nlp


def do_export(model, name: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate filename in the temporary directory.
        tmp_file_name = os.path.join(tmpdir, name + '.onnx')
        # Test export.
        model.export(tmp_file_name)


class TestHuggingFace(TestCase):
    @pytest.mark.unit
    def test_list_pretrained_models(self):
        pretrained_lm_models = nemo_nlp.modules.get_pretrained_lm_models_list()
        self.assertTrue(len(pretrained_lm_models) > 0)

    @pytest.mark.unit
    def test_get_pretrained_bert_model(self):
        model = nemo_nlp.modules.get_pretrained_lm_model('bert-base-uncased')
        assert isinstance(model, nemo_nlp.modules.BertEncoder)
        do_export(model, "bert-base-uncased")

    @pytest.mark.unit
    def test_get_pretrained_distilbert_model(self):
        model = nemo_nlp.modules.get_pretrained_lm_model('distilbert-base-uncased')
        assert isinstance(model, nemo_nlp.modules.DistilBertEncoder)
        do_export(model, "distilbert-base-uncased")

    @pytest.mark.unit
    def test_get_pretrained_roberta_model(self):
        model = nemo_nlp.modules.get_pretrained_lm_model('roberta-base')
        assert isinstance(model, nemo_nlp.modules.RobertaEncoder)
        do_export(model, "roberta-base-uncased")

    @pytest.mark.unit
    def test_get_pretrained_albert_model(self):
        model = nemo_nlp.modules.get_pretrained_lm_model('albert-base-v1')
        assert isinstance(model, nemo_nlp.modules.AlbertEncoder)
        do_export(model, "albert-base-v1")
