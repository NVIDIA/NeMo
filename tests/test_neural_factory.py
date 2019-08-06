# Copyright (c) 2019 NVIDIA Corporation
import unittest
from tests.context import nemo


class TestNeuralFactory(unittest.TestCase):

    def test_creation(self):
        neural_factory = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                                       local_rank=None)
        instance = neural_factory.get_module(name="TaylorNet", collection="toys",
                                             params={"dim": 4})
        self.assertTrue(isinstance(
            instance, nemo.backends.pytorch.tutorials.TaylorNet))

    def test_simple_exampe(self):
        #########################################################################
        ###
        neural_factory = nemo.core.neural_factory.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch,
            local_rank=None)
        dl = neural_factory.get_module(
            name="RealFunctionDataLayer", collection="toys",
            params={"n": 10000, "batch_size": 128})
        fx = neural_factory.get_module(name="TaylorNet", collection="toys",
                                       params={"dim": 4})
        loss = neural_factory.get_module(name="MSELoss", collection="toys",
                                         params={})

        x, y = dl()
        y_pred = fx(x=x)
        l = loss(predictions=y_pred, target=y)

        optimizer = neural_factory.get_trainer(params={})
        optimizer.train([l])
        #########################################################################
