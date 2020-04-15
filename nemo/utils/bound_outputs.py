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

from collections.abc import MutableMapping

from nemo.utils import logging


class BoundOutputs(MutableMapping):
    '''
        A specialized dictionary that contains bound outputs of a Neural Graph.
        In fact stores three lists of bound tensors:
            - "all" output tensors of all modules within a graph,
            - "default" output tensors with default keys taken from outputs of modules (might result in
            overwriting some keys), and
            - "manual" used for specifying the subset of output tensors, each with a new/different key
        When accessing the output tensors, it returns the "manual" tensors. If "manual" tensors are not defined,
        will return/work on "default" tensors.
    '''

    def __init__(self):
        """ Initializes three (empty) dictionaries. """
        # This dictionary stores list of output tensors of all modules, one key per
        # This will generate a "all output tensors" dictionary, where the key is name of "producer" module,
        # and the value contains all produced tensors.
        self._all_tensors = {}

        # This dictionary stores the output tensors collected during the "default" tensor recording.
        # As they are using the default port names, the second/next tensor published on the same port
        # will overwrite the old one (Warning).
        self._default_tensors = {}

        # This dictionary stores list of output tensors of module "manually" indicated by the user.
        # In this case tring to overwriting the existing ports with new tensors will be forbidden (Exception).
        self._manual_tensors = {}

    def __setitem__(self, key, value):
        """ This method is used to set the manual dictionary. """
        if key in self._manual_tensors.keys():
            raise KeyError("Overwriting of a port `{}` that was previously manually bound is not allowed".format(key))
        # Ok, set output.
        self._manual_tensors[key] = value

    def __getitem__(self, key):
        """ Returns output - depending whether there are some manual outputs or not. """
        if len(self._manual_tensors) > 0:
            return self._manual_tensors[key]
        else:  # Use default dict.
            return self._default_tensors[key]

    def __delitem__(self, key):
        raise NotImplementedError("Deleting a bound output port is not allowed")

    def __iter__(self):
        """ Iterates over the outputs - depending whether there are some manual outputs or not. """
        if len(self._manual_tensors) > 0:
            return iter(self._manual_tensors)
        else:  # Use default dict.
            return iter(self._default_tensors)

    def __len__(self):
        """ Return number of outputs - depending whether there are some manual outputs or not. """
        if len(self._manual_tensors) > 0:
            return len(self._manual_tensors)
        else:  # Use default dict.
            return len(self._default_tensors)

    def bind(self, tensors_list):
        """ Binds the output tensors.

            Args:
                tensors_list: List of tensors to be added.
        """
        for tensor in tensors_list:
            # Check the presence of the port name in "default" dictionary.
            name = tensor.name  # Use the default port name.
            if name in self._default_tensors.keys():
                logging.warning(
                    "Overwriting the already bound output port `{}` produced by `{}`".format(
                        name, self._default_tensors[name].producer_name
                    )
                )
            # Still, overwrite it.
            self._default_tensors[name] = tensor

            # Add tensor to "all" tensors dictionary.
            producer_name = tensor.producer_name
            if producer_name not in self._all_tensors.keys():
                self._all_tensors[producer_name] = []
            # Add tensor.
            self._all_tensors[producer_name].append(tensor)

    @property
    def all(self):
        """ Returns dictionary of all output tensors. """
        return self._all_tensors

    @property
    def definitions(self):
        """ Property returns definitions of the output ports by extracting them on the fly from the bound tensors. """
        # Get the right tensor dictionary.
        d = self._manual_tensors if len(self._manual_tensors) > 0 else self._default_tensors

        # Extract port definitions (Neural Types) straight from tensors.
        return {k: v.type for k, v in d.items()}
