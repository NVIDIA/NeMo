# Copyright (c) 2019 NVIDIA Corporation
import unittest
from .context import nemo
from .common_setup import NeMoUnitTest


class TestNeuralFactory(NeMoUnitTest):

    def test_creation(self):
        neural_factory = nemo.core.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch, local_rank=None,
            create_tb_writer=False)
        instance = neural_factory.get_module(
            name="TaylorNet", collection="toys",
            params={"dim": 4})
        self.assertTrue(isinstance(
            instance, nemo.backends.pytorch.tutorials.TaylorNet))

    def test_simple_example(self):
        #######################################################################
        neural_factory = nemo.core.neural_factory.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch,
            local_rank=None,
            create_tb_writer=False)
        dl = neural_factory.get_module(
            name="RealFunctionDataLayer", collection="toys",
            params={"n": 10000, "batch_size": 128})
        fx = neural_factory.get_module(name="TaylorNet", collection="toys",
                                       params={"dim": 4})
        loss = neural_factory.get_module(name="MSELoss", collection="toys",
                                         params={})

        x, y = dl()
        y_pred = fx(x=x)
        loss_tensor = loss(predictions=y_pred, target=y)

        optimizer = neural_factory.get_trainer()
        optimizer.train([loss_tensor], optimizer="sgd",
                        optimization_params={"lr": 1e-3})
        #######################################################################
