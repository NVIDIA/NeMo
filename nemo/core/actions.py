# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from abc import ABC, abstractmethod
from typing import List, Optional, Union

from nemo.core.neural_factory import Optimization
from nemo.core.neural_modules import ModuleType
from nemo.core.neural_types import NmTensor
from nemo.utils.app_state import AppState


def topological_sort_from_leaves(leaf_nmtensors: List[NmTensor], cached_training_state: 'TrainingState' = None):
    """A function that accepts a list of NmTensors that need to be computed and constructs a callchain DAG that starts
    from a datalayerNM and can be used to compute the NmTensors.

    args:
        leaf_nmtensors (List[NmTensors]): The tensors to be computed
        cached_training_state (TrainingState): A dictionary of already computed tensors.
            Defaults to None meaning an empty cache.

    returns:
        top_sorted_modules: the callchain DAG
    """

    def create_node(producer, producer_args):
        if producer_args is None:
            return tuple((producer, ()))
        return tuple((producer, tuple([(k, v) for k, v in producer_args.items()]),))

    def is_in_degree_zero(node, processed_nodes, cached_training_state):
        """A node has in degree of zero"""
        if node[1] == ():
            return True
        for _, nmtensor in node[1]:
            node = create_node(nmtensor.producer, nmtensor.producer_args)
            if node not in processed_nodes:
                if cached_training_state and cached_training_state.check_tensor_cached(nmtensor.unique_name):
                    continue
                return False
        return True

    hooks = leaf_nmtensors if isinstance(leaf_nmtensors, list) else [leaf_nmtensors]

    # ensures that no tensors are processed twice
    processed_nmtensors = set()

    indices_to_remove = []
    # Check for duplicates in hook
    for i, nmtensor in enumerate(hooks):
        if nmtensor in processed_nmtensors:
            indices_to_remove.append(i)
        else:
            processed_nmtensors.add(nmtensor)

    for i in reversed(indices_to_remove):
        hooks.pop(i)

    _top_sorted_modules = []
    all_nodes = {}

    # extract all nodes to all_nodes set
    hooks_lst = list(hooks)
    while len(hooks_lst) > 0:
        # take nmtensor from the end of the list
        nmtensor = hooks_lst.pop()
        producer_args = nmtensor.producer_args

        node = create_node(nmtensor.producer, producer_args)
        # Store nmtensor as an output of its producer
        # first make sure all keys are present per output port
        # and nm is inside all_nodes
        if node not in all_nodes:
            all_nodes[node] = {k: None for k in nmtensor.producer.output_ports}
        # second, populate output port with current nmtensor
        # where applicable
        all_nodes[node][nmtensor.output_port_name] = nmtensor
        processed_nmtensors.add(nmtensor)

        new_tensors = set()
        if producer_args is not None and producer_args != {}:
            for _, new_nmtensor in producer_args.items():
                if new_nmtensor not in processed_nmtensors:
                    new_tensors.add(new_nmtensor)

        if cached_training_state:
            for _, input_nmtensor in producer_args.items():
                if cached_training_state.check_tensor_cached(input_nmtensor.unique_name):
                    new_tensors.remove(input_nmtensor)

        # Order the new set so that all ranks have the callchain in the same order
        new_tensors = sorted(list(new_tensors), key=lambda x: str(x))
        for new_nmtensor in new_tensors:
            # put in the start of list
            hooks_lst.insert(0, new_nmtensor)

    all_node_with_output = []
    # Iterate over all_nodes to create new nodes that include its output
    # now all nodes have (module, input tensors, output tensors)
    for node in all_nodes:
        all_node_with_output.append(tuple((node[0], node[1], all_nodes[node])))

    processed_nodes = []
    while len(all_node_with_output) > 0:
        for node in all_node_with_output.copy():
            # if node's in_degree is zero it can be added to
            # _top_sorted_modules
            # this will also reduce in_degree of its children
            if is_in_degree_zero(node, processed_nodes, cached_training_state):
                _top_sorted_modules.append(node)
                processed_nodes.append((node[0], node[1]))
                all_node_with_output.remove(node)

    # Create top_sorted_modules aka callchain
    top_sorted_modules = []
    for i, mod in enumerate(_top_sorted_modules):
        top_sorted_modules.append((mod[0], dict(mod[1]), mod[2]))
        # Ensure that there is only one dataset in callchain
        if i > 0 and mod[0].type == ModuleType.datalayer:
            raise ValueError("There were more than one DataLayer NeuralModule inside your DAG.")

        if cached_training_state and mod[0].type == ModuleType.datalayer:
            raise ValueError("Could not compute tensor from current cached training state.")

    return top_sorted_modules


class TrainingState:
    def __init__(self, action: 'Actions'):
        """A class used to wrap the current training state of an Actions.train() function. This class holds a mapping
        of tensor.unique_name -> it's backend tensor (eg Pytorch Tensor) or None if the tensor has been been computed
        on the current step.

        args:
            action (Actions): The Actions object this state is associated with.
        """
        tensor_naming_registery = AppState().tensor_names
        self.tensor_dict = {}.fromkeys(tensor_naming_registery.unique_names, None)
        self._action = action

    def tensor_list(self):
        """Returns a list the unique names of all tensors.
        """
        return self.tensor_dict.keys()

    def clear_dict(self):
        """Clears the dictionary by setting all values to None. Used in-between training batches to clear it's state.
        """
        for name in self.tensor_dict:
            self.tensor_dict[name] = None

    def set_tensor(self, tensor: NmTensor, value: 'torch.Tensor'):
        """Sets the value of tensor

        args:
            tensor (NmTensor)
            value (torch.Tensor)
        """
        self.tensor_dict[tensor.unique_name] = value

    def check_tensor_cached(self, unique_name: str):
        """Checks to see the tensor value has been computed in the current step yet.

        args:
            unique_name (str): The NmTensor.unique_name that we want to check for.

        returns:
            (bool) whether the tensor with unique_name has been computed yet.
        """
        if self.tensor_dict[unique_name] is None:
            return False
        return True

    def get_tensor(self, name: Union[str, NmTensor], compute: bool = True):
        """Returns the value associated with a tensor. And optionally, computes the value of the tensor if not already
        set.

        args:
            name (str, NmTensor): The user-defined name for a tensor or the NmTensor itself.
            compute (bool): If True and the tensor has not already been computed, there will be an attempt to create a
                call DAG and then do a forward pass on this call DAG to compute the tensor. If False, it will return
                None if the tensor has not been computed yet.
                Defaults to True.

        returns:
            (torch.tensor or None) representing the computed value of the requested name. Returns None if compute is
            False and the tensor has not been computed yet.
        """
        if isinstance(name, NmTensor):
            unique_name = name.unique_name
        else:
            unique_name = AppState().tensor_names[name]
        tensor_value = self.tensor_dict[unique_name]
        if tensor_value is None and compute:
            nmtensor = AppState().tensor_names._nmtensor_uniname_dict[unique_name]
            callchain = topological_sort_from_leaves([nmtensor], cached_training_state=self)
            callchain.insert(0, ())
            self._action.nm_graph_forward_pass(callchain, self.tensor_dict)
            tensor_value = self.tensor_dict[unique_name]
        return tensor_value


class Actions(ABC):
    """Basic actions allowed on graphs of Neural Modules"""

    def __init__(self, local_rank, global_rank, optimization_level=Optimization.mxprO0):
        self._local_rank = local_rank
        self._global_rank = global_rank
        self._optim_level = optimization_level

    @property
    def local_rank(self):
        """Local rank during distributed execution. None if single GPU/CPU

        Returns:
            (int) rank or worker or None if not in distributed model
        """
        return self._local_rank

    @property
    def global_rank(self):
        """Global rank during distributed execution. None if single GPU/CPU

        Returns:
            (int) rank or worker or None if not in distributed model
        """
        return self._global_rank

    @abstractmethod
    def train(
        self,
        tensors_to_optimize: List[NmTensor],
        callbacks: Optional[List[Union['ActionCallback', 'NeMoCallback']]],
        lr_policy=None,
        batches_per_step=None,
        stop_on_nan_loss=False,
    ):
        """This action executes training and (optionally) evaluation.

        Args:
            tensors_to_optimize: which tensors to optimize. Typically this is
                single loss tesnor.
            callbacks: list of callback objects
            lr_policy: function which should take (initial_lr, step, epoch) and
                return learning rate
            batches_per_step: number of mini-batches to process before one
                optimizer step. (default: None, same as 1). Use this
                to simulate larger batch sizes on hardware which could not fit
                larger batch in memory otherwise. Effectively, this will make
                "algorithmic" batch size per GPU/worker = batches_per_step*
                batch_size
            stop_on_nan_loss: (default: False) If set to True, the training
                will stop if loss=nan or inf. If set to False, the training
                will continue.

        Returns:
            None
        """
        pass

    @abstractmethod
    def infer(self, tensors: List[NmTensor]):
        """This action executes inference. Nothing is optimized.
        Args:
          tensors: which tensors to evaluate.

        Returns:
          None
        """
        pass

    @abstractmethod
    def create_optimizer(self, optimizer, things_to_optimize, optimizer_params):
        """
        Creates an optimizer object to be use in the train() method.

        Args:
            optimizer: Specifies which optimizer to use.
            things_to_optimize: A list of neural modules or tensors to be
                optimized.
            optimizer_params: Specifies the parameters of the optimizer

        Returns:
            Optimizer
        """
        pass
