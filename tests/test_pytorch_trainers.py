# Copyright (c) 2019 NVIDIA Corporation
import unittest
from tests.context import nemo


class TestPytorchTrainers(unittest.TestCase):

    def test_simple_train(self):
        print("Simplest train test")
        data_source = nemo.backends.pytorch.tutorials.RealFunctionDataLayer(n=10000,
                                                                            batch_size=128)
        trainable_module = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        loss = nemo.backends.pytorch.tutorials.MSELoss()
        x, y = data_source()
        y_pred = trainable_module(x=x)
        l = loss(predictions=y_pred, target=y)

        optimizer = nemo.backends.pytorch.actions.PtActions(
            params={"learning_rate": 0.0003, "num_epochs": 1})
        optimizer.train(tensors_to_optimize=[l])

    def test_simple_chained_train(self):
        print("Chained train test")
        data_source = nemo.backends.pytorch.tutorials.RealFunctionDataLayer(n=10000,
                                                                            batch_size=32)
        trainable_module1 = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        trainable_module2 = nemo.backends.pytorch.tutorials.TaylorNet(dim=2)
        trainable_module3 = nemo.backends.pytorch.tutorials.TaylorNet(dim=2)
        loss = nemo.backends.pytorch.tutorials.MSELoss()
        x, y = data_source()
        y_pred1 = trainable_module1(x=x)
        y_pred2 = trainable_module2(x=y_pred1)
        y_pred3 = trainable_module3(x=y_pred2)
        l = loss(predictions=y_pred3, target=y)

        optimizer = nemo.backends.pytorch.actions.PtActions(
            params={"learning_rate": 0.0003, "num_epochs": 1})
        optimizer.train(tensors_to_optimize=[l])


if __name__ == '__main__':
    unittest.main()
