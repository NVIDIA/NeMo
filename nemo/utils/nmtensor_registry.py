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

from weakref import WeakValueDictionary


class NmTensorNameRegistry:
    def __init__(self):
        """
            Constructor. Initializes the NmTensorNameRegistry. Reserves the default 'loss' name.

            TODO: We should be recording the tensors of each graph rather than all the tensors.
        """
        # Create the nmtensor_naming_dict
        # which contains a mapping of str to NMTensor.unique_name
        self._nmtensor_naming_dict = {"loss": "loss"}  # Reserve keyname of 'loss'
        # Create a dict that maps unique_names to tensors for use with TrainingState.get_tensor()
        self._nmtensor_uniname_dict = WeakValueDictionary()

    @property
    def unique_names(self):
        """Returns the set of all NmTensors.unique_names + 'loss'
        """
        return list(self._nmtensor_uniname_dict.keys()) + ["loss"]

    def register(self, tensor: 'NmTensor'):
        """Helper function to register a newly created NmTensor by adding it to self.__nmtensor_uniname_dict.
        Should be called from NmTensor.__init__()

        args:
            tensor (NmTensor): The tensor to be registered.
        """

        # Check if object is already in a set.
        if tensor.unique_name in self._nmtensor_uniname_dict:
            pass

        # Finally, add object to the set.
        self._nmtensor_uniname_dict[tensor.unique_name] = tensor

    def rename_NmTensor(self, tensor: 'NmTensor', new_name: str):
        """Helper function that changes the naming dictionary to facilitate user name -> tensor.unique_name lookup.

        args:
            tensor (NmTensor): The tensor to be renamed.
            new_name (str): its new name.
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

    def __getitem__(self, key: str):
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

        if key in self._nmtensor_uniname_dict or key == "loss":
            return key

        raise KeyError("A NmTensor with name `{}` don't exists!".format(key))
