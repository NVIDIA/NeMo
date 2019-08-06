# Copyright (c) 2019 NVIDIA Corporation
import unittest
from tests.context import nemo


class TestPytorchChatBotTutorial(unittest.TestCase):

    def test_simple_train(self):
        datafile = "tests/data/dialog_sample.txt"
        print(datafile)
        voc, pairs = nemo.backends.pytorch.tutorials.chatbot.data.loadPrepareData("cornell",
                                                                                  datafile=datafile)
        self.assertEqual(voc.name, 'cornell')
        self.assertEqual(voc.num_words, 675)


if __name__ == '__main__':
    unittest.main()
