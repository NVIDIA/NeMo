# Copyright (c) 2019 NVIDIA Corporation
import unittest
from .context import nemo
from nemo.backends.pytorch.tutorials.chatbot.data import loadPrepareData
from .common_setup import NeMoUnitTest


class TestPytorchChatBotTutorial(NeMoUnitTest):

    def test_simple_train(self):
        datafile = "tests/data/dialog_sample.txt"
        print(datafile)
        voc, pairs = loadPrepareData("cornell", datafile=datafile)
        self.assertEqual(voc.name, 'cornell')
        self.assertEqual(voc.num_words, 675)


if __name__ == '__main__':
    unittest.main()
