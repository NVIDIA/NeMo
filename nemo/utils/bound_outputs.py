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
        A specialized dictionary that contains bound outputs.
        In fact stores two lists of bound tensors ("default" and "manual"), and accesses them following the logic:
        1) Record all output tensors as "default"
        2) If user doesn't set any outputs manually, use the default.
    '''

    def __init__(self):
        """ Initializes two dictionaries. """
        self._default_dict = {}
        self._manual_dict = {}

    def __setitem__(self, key, value):
        """ This method is used to set the manual dictionary. """
        if key in self._manual_dict.keys():
            raise KeyError("Overwriting of a port `{}` that was previously manually bound is not allowed".format(key))
        # Ok, set output.
        self._manual_dict[key] = value

    def __getitem__(self, key):
        """ Returns output - depending whether there are some manual outputs or not. """
        if len(self._manual_dict) > 0:
            return self._manual_dict[key]
        else:  # Use default dict.
            return self._default_dict[key]

    def __delitem__(self, key):
        raise NotImplemented("Deleting a bound output port is not allowed")

    def __iter__(self):
        """ Iterates over the outputs - depending whether there are some manual outputs or not. """
        if len(self._manual_dict) > 0:
            return iter(self._manual_dict)
        else:  # Use default dict.
            return iter(self._default_dict)

    def __len__(self):
        """ Return number of outputs - depending whether there are some manual outputs or not. """
        if len(self._manual_dict) > 0:
            return len(self._manual_dict)
        else:  # Use default dict.
            return len(self._default_dict)

    def add_defaults(self, tensors_list):
        """ Binds default output tensors.

            Args:
                tensors_list: List of tensors to be added.
        """
        for tensor in tensors_list:
            # Check the presence of the port name in default dictionary.
            name = tensor.name # Use the default port name.
            if name in self._default_dict.keys():
                logging.warning(
                    "Overwriting the already bound output port `{}` produced by `{}`".format(
                        name, self._default_dict[name].producer.name
                    )
                )
            # Still, overwrite it.
            self._default_dict[name] = tensor

    @property
    def definitions(self):
        """ Property returns definitions of the output ports by extracting them on the fly from the bound tensors. """
        # Get dict.
        d = self._manual_dict if len(self._manual_dict) > 0 else self._default_dict

        # Extract port definitions (Neural Types) straight from tensors.
        return {k: v.type for k, v in d.items()}
