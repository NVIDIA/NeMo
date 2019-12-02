# Copyright (c) 2019 NVIDIA Corporation
import unittest
import os

from .context import nemo, nemo_nlp
from .common_setup import NeMoUnitTest


class TestBert(NeMoUnitTest):
    def test_list_pretrained_models(self):
        pretrained_models = nemo_nlp.huggingface.BERT.list_pretrained_models()
        self.assertTrue(len(pretrained_models) > 0)
