# Copyright (c) 2019 NVIDIA Corporation
import unittest
import os
import torch

from .context import nemo, nemo_nlp
from .common_setup import NeMoUnitTest


class TestDeployExport(NeMoUnitTest):
    def test_simple_module_export(self):

        class MyMod(torch.nn.Module):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self._dim1 = 10
                self._dim2 = 1
                self.fwd = torch.nn.Linear(self._dim1, self._dim2)

            def forward(self, x):
                return self.fwd(x)

        m = MyMod()

        traced_m = torch.jit.script(m)
        traced_m.save('m.pt')
        nf = nemo.core.NeuralModuleFactory(placement=nemo.core.DeviceType.CPU)
        trainable_module = nemo.backends.pytorch.tutorials.TaylorNet(dim=4,
                                                                     factory=nf)
        def new_call(self, *input, **kwargs):
            return torch.nn.Module.__call__(self, *input, **kwargs)

        type(trainable_module).__call__ = new_call
        #traced_m = torch.jit.trace(trainable_module, torch.rand(4, 1))
        traced_m = torch.jit.script(trainable_module)
        traced_m.save('trainable_module.pt')
