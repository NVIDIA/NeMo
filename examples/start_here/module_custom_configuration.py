# ! /usr/bin/python
# -*- coding: utf-8 -*-

# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from enum import Enum

from nemo.backends.pytorch.tutorials import MSELoss, RealFunctionDataLayer, TaylorNet
from nemo.core import DeviceType, NeuralModuleFactory, SimpleLossLoggerCallback
from nemo.utils import logging


# A custom enum.
class Status(Enum):
    success = 0
    error = 1


class CustomTaylorNet(TaylorNet):
    """Module which learns Taylor's coefficients. Extends the original module by a custom status enum."""

    def __init__(self, dim, status: Status):
        super().__init__(dim)
        logging.info("Status: {}".format(status))

    def _serialize_configuration(self):
        """
            A custom method serializing the configuration to a YAML file.

            Returns:
                a "serialized" dictionary with module configuration.
        """

        # Create the dictionary to be exported.
        init_to_export = {}

        # "Serialize" dim.
        init_to_export["dim"] = self._init_params["dim"]

        # Custom "serialization" of the status.
        if self._init_params["status"] == Status.success:
            init_to_export["status"] = 0
        else:
            init_to_export["status"] = 1

        # Return serialized parameters.
        return init_to_export

    @classmethod
    def _deserialize_configuration(cls, init_params):
        """
            A function that deserializes the module "configuration (i.e. init parameters).

            Args:
                init_params: List of init parameters loaded from the YAML file.

            Returns:
                A "deserialized" list with init parameters.
        """
        deserialized_params = {}

        # "Deserialize" dim.
        deserialized_params["dim"] = init_params["dim"]

        # Custom "deserialization" of the status.
        if init_params["status"] == 0:
            deserialized_params["status"] = Status.success
        else:
            deserialized_params["status"] = Status.error

        # Return deserialized parameters.
        return deserialized_params


# Run on CPU.
nf = NeuralModuleFactory(placement=DeviceType.CPU)

# Instantitate RealFunctionDataLayer defaults to f=torch.sin, sampling from x=[-1, 1]
dl = RealFunctionDataLayer(n=100, f_name="cos", x_lo=-1, x_hi=1, batch_size=32)

# Instantiate a simple feed-forward, single layer neural network.
fx = CustomTaylorNet(dim=4, status=Status.error)

# Instantitate loss.
mse_loss = MSELoss()

# Export the model configuration.
fx.export_to_config("/tmp/custom_taylor_net.yml")

# Create a second instance, using the parameters loaded from the previously created configuration.
# Please note that we are calling the overriden method from the CustomTaylorNet class.
fx2 = CustomTaylorNet.import_from_config("/tmp/custom_taylor_net.yml")

# Create a graph by connecting the outputs with inputs of modules.
x, y = dl()
# Please note that in the graph we are using the "second" instance.
p = fx2(x=x)
loss = mse_loss(predictions=p, target=y)

# SimpleLossLoggerCallback will print loss values to console.
callback = SimpleLossLoggerCallback(
    tensors=[loss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}')
)

# Invoke the "train" action.
nf.train([loss], callbacks=[callback], optimization_params={"num_epochs": 3, "lr": 0.0003}, optimizer="sgd")
