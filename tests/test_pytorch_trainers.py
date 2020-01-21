# Copyright (c) 2019 NVIDIA Corporation
import unittest
from .context import nemo
from .common_setup import NeMoUnitTest


class TestPytorchTrainers(NeMoUnitTest):

    def test_simple_train(self):
        print("Simplest train test")
        data_source = nemo.backends.pytorch.tutorials.RealFunctionDataLayer(
            n=10000, batch_size=128)
        trainable_module = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        loss = nemo.backends.pytorch.tutorials.MSELoss()
        x, y = data_source()
        y_pred = trainable_module(x=x)
        loss_tensor = loss(predictions=y_pred, target=y)

        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            tensors_to_optimize=[loss_tensor],
            optimizer="sgd",
            optimization_params={"lr": 0.0003, "num_epochs": 1}
            )

    def test_simple_train_named_output(self):
        print('Simplest train test with using named output.')
        data_source = nemo.backends.pytorch.tutorials.RealFunctionDataLayer(
            n=10000,
            batch_size=128,
        )
        trainable_module = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        loss = nemo.backends.pytorch.tutorials.MSELoss()

        data = data_source()
        self.assertEqual(
            first=type(data).__name__,
            second='RealFunctionDataLayerOutput',
            msg='Check output class naming coherence.',
        )
        y_pred = trainable_module(x=data.x)
        loss_tensor = loss(predictions=y_pred, target=data.y)

        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            tensors_to_optimize=[loss_tensor],
            optimizer="sgd",
            optimization_params={"lr": 0.0003, "num_epochs": 1}
        )

    def test_simple_chained_train(self):
        print("Chained train test")
        data_source = nemo.backends.pytorch.tutorials.RealFunctionDataLayer(
            n=10000, batch_size=32)
        trainable_module1 = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        trainable_module2 = nemo.backends.pytorch.tutorials.TaylorNet(dim=2)
        trainable_module3 = nemo.backends.pytorch.tutorials.TaylorNet(dim=2)
        loss = nemo.backends.pytorch.tutorials.MSELoss()
        x, y = data_source()
        y_pred1 = trainable_module1(x=x)
        y_pred2 = trainable_module2(x=y_pred1)
        y_pred3 = trainable_module3(x=y_pred2)
        loss_tensor = loss(predictions=y_pred3, target=y)

        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            tensors_to_optimize=[loss_tensor],
            optimizer="sgd",
            optimization_params={"lr": 0.0003, "num_epochs": 1}
        )


if __name__ == '__main__':
    unittest.main()
