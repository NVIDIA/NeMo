# Copyright (c) 2019 NVIDIA Corporation
import unittest
import os
from tests.context import nemo


class TestTrainers(unittest.TestCase):

    def test_checkpointing(self):
        path = 'optimizer.pt'
        optimizer = nemo.backends.pytorch.actions.PtActions(
            params={"learning_rate": 0.0003, "num_epochs": 1})
        optimizer.save_state_to(path)
        optimizer.step = 123
        optimizer.epoch_num = 324
        optimizer.restore_state_from(path)
        self.assertEqual(optimizer.step, 0)
        self.assertEqual(optimizer.epoch_num, 0)
        os.remove(path)
