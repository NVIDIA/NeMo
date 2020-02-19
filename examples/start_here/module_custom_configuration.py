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
from os import path

from ruamel import yaml

import nemo
from nemo import logging
from nemo.core import DeviceType, NeuralModuleFactory, SimpleLossLoggerCallback
from nemo.core.neural_types import ChannelType, NeuralType


# A custom enum.
class Status(Enum):
    success = 0
    error = 1


class CustomTaylorNet(nemo.tutorials.TaylorNet):
    """Module which learns Taylor's coefficients. Extends the original module by a custom status enum."""

    @property
    def input_ports(self):
        return {"x": NeuralType(('B', 'D'), ChannelType())}

    @property
    def output_ports(self):
        return {"y_pred": NeuralType(('B', 'D'), ChannelType())}

    def __init__(self, dim, status):
        super().__init__(dim)
        logging.info("Status: {}".format(status))

    def export_to_config(self, config_file):
        """
            A custom method exporting configuration to a YAML file.

            Args:
                config_file: path (absolute or relative) and name of the config file (YML)
        """

        # Greate an absolute path.
        abs_path_file = path.expanduser(config_file)

        # Create the dictionary to be exported.
        to_export = {}

        # Add "header" with module "specification".
        to_export["header"] = self._create_config_header()

        # Add init parameters.
        to_export["init_params"] = self._init_params

        # Custom processing of the status.
        if to_export["init_params"]["status"] == Status.success:
            to_export["init_params"]["status"] = 0
        else:
            to_export["init_params"]["status"] = 1

        # All parameters are ok, let's export.
        with open(abs_path_file, 'w') as outfile:
            yaml.dump(to_export, outfile)

        logging.info(
            "Configuration of module {} ({}) exported to {}".format(self._uuid, type(self).__name__, abs_path_file)
        )

    @classmethod
    def import_from_config(cls, config_file, section_name=None, overwrite_params={}):
        """
            A custom method importing the YAML configuration file.
            Raises an ImportError exception when config file is invalid or
            incompatible (when called from a particular class).

            Args:
                config_file: path (absolute or relative) and name of the config file (YML)

                section_name: section in the configuration file storing module configuration (optional, DEFAULT: None)

                overwrite_params: Dictionary containing parameters that will be added to or overwrite (!) the default
                parameters loaded from the configuration file

            Returns:
                Instance of the created NeuralModule object.
        """

        # Validate the content of the configuration file (its header).
        loaded_config = cls._validate_config_file(config_file, section_name)

        # Get init parameters.
        init_params = loaded_config["init_params"]
        # Update parameters with additional ones.
        init_params.update(overwrite_params)

        # Custom processing of the status.
        if init_params["status"] == 0:
            init_params["status"] = Status.success
        else:
            init_params["status"] = Status.error

        # Create and return the object.
        obj = CustomTaylorNet(**init_params)
        logging.info(
            "Instantiated a new Neural Module of type `{}` using configuration loaded from the `{}` file".format(
                "CustomTaylorNet", config_file
            )
        )
        return obj


# Run on CPU.
nf = NeuralModuleFactory(placement=DeviceType.CPU)

# Instantitate RealFunctionDataLayer defaults to f=torch.sin, sampling from x=[-1, 1]
dl = nemo.tutorials.RealFunctionDataLayer(n=100, f_name="cos", x_lo=-1, x_hi=1, batch_size=128)

# Instantiate a simple feed-forward, single layer neural network.
fx = CustomTaylorNet(dim=4, status=Status.error)

# Instantitate loss.
mse_loss = nemo.tutorials.MSELoss()

# Export the model configuration.
fx.export_to_config("/tmp/custom_taylor_net.yml")

# Create a second instance, using the parameters loaded from the previously created configuration.
# Please note that we are calling the overriden method from the CustomTaylorNet class.
fx2 = CustomTaylorNet.import_from_config("/tmp/custom_taylor_net.yml")

# Create a graph by connecting the outputs with inputs of modules.
x, y = dl()
# Please note that in the graph are using the "second" instance.
p = fx2(x=x)
loss = mse_loss(predictions=p, target=y)

# SimpleLossLoggerCallback will print loss values to console.
callback = SimpleLossLoggerCallback(
    tensors=[loss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}')
)

# Invoke the "train" action.
nf.train([loss], callbacks=[callback], optimization_params={"num_epochs": 3, "lr": 0.0003}, optimizer="sgd")
