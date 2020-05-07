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


class NmTensorNameRegistry:
    def __init__(self):
        """
            Constructor. Initializes the manager. Sets active graph to None.

            TODO: Should probably be a property of a graph
        """
        # Create the nmtensor_naming_dict
        # which contains a mapping of str to NMTensor.unique_name
        self._nmtensor_naming_dict = {"loss": "loss"}  # Reserve keyname of 'loss'
        # self._nmtensor_uniname_set = set(["loss"])
        self._nmtensor_uniname_dict = {"loss": None}

    # def summary(self):
    #     """ Prints a nice summary. """
    #     desc = ""
    #     for graph in self:
    #         desc = desc + "`{}`: {}\n".format(graph.name, graph)
    #     return desc

    @property
    def unique_names(self):
        return self._nmtensor_uniname_dict.keys()

    # def register(self, tensor: NmTensor):
    def register(self, tensor):
        """TODO
        """

        # Check if object is already in a set.
        if tensor.unique_name in self._nmtensor_uniname_dict:
            pass

        # Finally, add object to the set.
        self._nmtensor_uniname_dict[tensor.unique_name] = tensor

    # def rename_NmTensor(self, tensor: NmTensor, new_name: str):
    def rename_NmTensor(self, tensor, new_name: str):
        """ TODO
        """
        # Find old name if exists
        old_name = tensor.unique_name
        for custom_name, unique_name in self._nmtensor_naming_dict.items():
            if unique_name == tensor.unique_name:
                old_name = custom_name

        if old_name != tensor.unique_name:
            del self._nmtensor_naming_dict[old_name]

        if new_name in self._nmtensor_naming_dict:
            raise KeyError(f"{new_name} already exists in current graph. Please use a unique name")
        self._nmtensor_naming_dict[new_name] = tensor.unique_name

    def __getitem__(self, key):
        """
        Object getter function.

        Args:
            key: Object name.

        Returns:
            Object associated with the key.
        """
        # Search for an object with a given name.
        if key in self._nmtensor_naming_dict:
            key = self._nmtensor_naming_dict[key]

        if key in self._nmtensor_uniname_dict:
            return key

        raise KeyError("A NmTensor with name `{}` don't exists!".format(key))
