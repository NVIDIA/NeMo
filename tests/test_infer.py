# Copyright (c) 2019 NVIDIA Corporation
import torch

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.core.neural_types import *

from .context import nemo
from .common_setup import NeMoUnitTest


class AddsTen(NonTrainableNM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def inputs(self):
        return {
            "mod_in": NeuralType({0: AxisType(BatchTag),
                                  1: AxisType(BaseTag, dim=1)})
        }

    @property
    def outputs(self):
        return {
            "mod_out": NeuralType({0: AxisType(BatchTag),
                                   1: AxisType(BaseTag, dim=1)})
        }

    def forward(self, mod_in):
        return mod_in + 10


class SubtractsTen(NonTrainableNM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def inputs(self):
        return {
            "mod_in": NeuralType({0: AxisType(BatchTag),
                                  1: AxisType(BaseTag, dim=1)})
        }

    @property
    def outputs(self):
        return {
            "mod_out": NeuralType({0: AxisType(BatchTag),
                                   1: AxisType(BaseTag, dim=1)})
        }

    def forward(self, mod_in):
        return mod_in - 10


class TestInfer(NeMoUnitTest):
    def test_infer_caching(self):
        neural_factory = nemo.core.neural_factory.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch, create_tb_writer=False)

        data_source = nemo.backends.pytorch.common.ZerosDataLayer(
            size=1,
            dtype=torch.FloatTensor,
            batch_size=1,
            output_ports={
                "dl_out": NeuralType({0: AxisType(BatchTag),
                                      1: AxisType(BaseTag, dim=1)})})
        addten = AddsTen()
        minusten = SubtractsTen()

        zero_tensor = data_source()
        ten_tensor = addten(mod_in=zero_tensor)
        twenty_tensor = addten(mod_in=ten_tensor)
        thirty_tensor = addten(mod_in=twenty_tensor)

        evaluated_tensors = neural_factory.infer(
            tensors=[twenty_tensor, thirty_tensor],
            verbose=False,
            cache=True
        )
        self.assertEqual(evaluated_tensors[0][0].squeeze().data, 20)
        self.assertEqual(evaluated_tensors[1][0].squeeze().data, 30)

        new_ten_tensor = minusten(mod_in=twenty_tensor)
        evaluated_tensors = neural_factory.infer(
            tensors=[new_ten_tensor],
            verbose=False,
            use_cache=True
        )
        self.assertEqual(evaluated_tensors[0][0].squeeze().data, 10)

    def test_infer_errors(self):
        neural_factory = nemo.core.neural_factory.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch, create_tb_writer=False)

        data_source = nemo.backends.pytorch.common.ZerosDataLayer(
            size=1,
            dtype=torch.FloatTensor,
            batch_size=1,
            output_ports={
                "dl_out": NeuralType({0: AxisType(BatchTag),
                                      1: AxisType(BaseTag, dim=1)})})
        addten = AddsTen()
        minusten = SubtractsTen()

        zero_tensor = data_source()
        ten_tensor = addten(mod_in=zero_tensor)
        twenty_tensor = addten(mod_in=ten_tensor)
        thirty_tensor = addten(mod_in=twenty_tensor)

        with self.assertRaisesRegex(ValueError,
                                    "use_cache was set, but cache was empty"):
            evaluated_tensors = neural_factory.infer(
                tensors=[twenty_tensor, thirty_tensor],
                verbose=False,
                use_cache=True
            )

        new_ten_tensor = minusten(mod_in=twenty_tensor)
        evaluated_tensors = neural_factory.infer(
            tensors=[new_ten_tensor],
            verbose=False,
            cache=True
        )

        with self.assertRaisesRegex(ValueError,
                                    "cache was set but was not empty"):
            evaluated_tensors = neural_factory.infer(
                tensors=[twenty_tensor, thirty_tensor],
                verbose=False,
                cache=True
            )

        neural_factory.clear_cache()
        evaluated_tensors = neural_factory.infer(
            tensors=[new_ten_tensor],
            verbose=False,
            cache=True
        )

        with self.assertRaisesRegex(ValueError,
                                    "cache and use_cache were both set."):
            evaluated_tensors = neural_factory.infer(
                tensors=[twenty_tensor, thirty_tensor],
                verbose=False,
                cache=True,
                use_cache=True
            )
        self.assertEqual(evaluated_tensors[0][0].squeeze().data, 10)
