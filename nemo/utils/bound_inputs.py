# -*- coding: utf-8 -*-

# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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

from collections import namedtuple
from collections.abc import MutableMapping

from nemo.utils import logging
from nemo.utils.module_port import ModulePort

# Actually this throws error as module dependency is: core depends on utils :]
# from nemo.core.neural_types import NeuralType


class BoundInput(object):
    """ A helper class represenging a single bound input. """

    def __init__(self, type, modules=[]):
        """ 
        Initializes object.

        Args:
            type: a NeuralType object.
            modules: a list of ModulePort tuples (module name, port name).
        """
        self._type = type
        self._modules = modules

    def bind(self, modules):
        """ Binds the modules to this "graph input".

            Args:
                modules: List of ModulePort tuples to be added.
        """
        for module in modules:
            self._modules.append(module)

    @property
    def type(self):
        """ Returns NeuralType of that input. """
        return self._type

    @property
    def modules(self):
        """ Returns list of bound modules (i.e. (module name, port name) tupes) """
        return self._modules


class BoundInputs(MutableMapping):
    '''
        A specialized dictionary that contains bound inputs of a Neural Graph.
    '''

    def __init__(self):
        """ Initializes the mapping. """
        self._inputs = {}

    def __setitem__(self, key, value):
        """
            This method is used to "create" a bound input, i.e. copy definition from indicated module input port.

            Args:
                key: name of the input port of the Neural Graph.
                value: NeuralType that will be set.
        """
        if key in self._inputs.keys():
            raise KeyError("Overwriting definition of a previously bound port `{}` is not allowed".format(key))
        # Make sure that a proper NeuralType definition was passed here.
        # if type(value) is not NeuralType:
        #    raise TypeError("Port `{}` definition must be must be a NeuralType".format(key))

        # Ok, add definition to list of mapped (module, port)s.
        # Note: for now, there are no mapped modules, thus [].
        self._inputs[key] = BoundInput(type=value, modules=[])

    def __getitem__(self, key):
        """ Returns bound input. """
        return self._inputs[key]

    def __delitem__(self, key):
        raise NotImplementedError("Deleting a bound input port is not allowed")

    def __iter__(self):
        """ Iterates over the bound inputs. """
        return iter(self._inputs)

    def __len__(self):
        """ Return number of bound inputs. """
        return len(self._inputs)

    @property
    def definitions(self):
        """ Property returns definitions of the input ports by extracting them on the fly from list. """
        # Extract port definitions (Neural Types) from the inputs list.
        return {k: v.type for k, v in self._inputs.items()}
