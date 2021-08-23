# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC
from contextlib import contextmanager
from enum import Enum
from typing import Dict, Iterator, List, Optional, Union

import torch

# TODO @blisc: Perhaps refactor instead of import guarding
try:
    from omegaconf import DictConfig

    _DISTILLATION_TYPE = None
    _DISTILLATION_LOSS_DICT = {}
    _DISTILLATION_CFG = DictConfig({})

except (ImportError, ModuleNotFoundError):

    _DISTILLATION_TYPE = None
    _DISTILLATION_LOSS_DICT = {}
    _DISTILLATION_CFG = {}


class DistillationType(Enum):
    STUDENT = 1
    TEACHER = 2


@contextmanager
def as_distill_type(distill_type: Optional[DistillationType]):
    if distill_type is not None and not isinstance(distill_type, DistillationType):
        raise TypeError("`distill_type` must be a valid enum value from `DistillationType` or None")

    global _DISTILLATION_TYPE

    original_type = _DISTILLATION_TYPE
    _DISTILLATION_TYPE = distill_type
    yield
    _DISTILLATION_TYPE = original_type


def set_distill_cfg(cfg: 'DictConfig'):
    global _DISTILLATION_CFG
    _DISTILLATION_CFG = cfg


def set_distill_loss_dict(loss_dict: dict):
    global _DISTILLATION_LOSS_DICT
    _DISTILLATION_LOSS_DICT = loss_dict


class DistillationMixin(ABC):
    """
    Mixin class to add Distillation support to Models or Pytorch/Neural Modules.
    """

    def __init__(self):
        super().__init__()
        self._distillation_registry = {}

    def setup_distillation_loss(self) -> Optional['torch.nn._Loss']:
        """
        Setup of distillation losses that are used during distillation training.

        If implemented by base class, in case the distillation config does not contain the 'loss' sub-config,
        the model itself can provide a default loss here, which will be used in place of the config.

        Returns:
            A dictionary of Loss object(s) that will be used as the distillation loss function.
            The dictionary must have at least 1 key - "primary" which is the primary loss function used
            for distillation.
            By default, this function returns KLDivergence loss (primary) and CosineEmbeddingLossWrapper (cosine).
            If None is returned, the distillation config must have an appropriate loss function defined.
        """
        # Lazy import to avoid circular dependency between imports
        from nemo.collections.common.losses import CosineEmbeddingLossWrapper, ScaledKLDivLoss

        temperature = self.distill_cfg.get('temperature', 1.0)
        primary = ScaledKLDivLoss(temperature, log_target=True, reduction='batchmean')
        cosine = CosineEmbeddingLossWrapper()
        loss_dict = {'primary': primary, 'cosine': cosine}
        return loss_dict

    def register_distillation_tensor(
        self, loss_key: str = None, tensor: torch.Tensor = None, loss_name: str = "primary",
    ):
        """
        Method to register a tensor to a certain loss (via the loss_name), binding the tensor to a keyword argument
        for that loss (via loss_key).

        At least one loss key must be present - `primary`. There can be any number of secondary loss keys.

        Args:
            loss_key: An optional string, used to bind the tensor via keyword argument to the loss.
                If None, assumed the loss is a binary function which does not require named keyword arguments.
            tensor: A pytorch tensor. If the distillation config has a key `preserve_distillation_memory`, and it
                is set to True, then the tensor will be moved onto the CPU. This may significantly impact training
                speed, so if memory is available, ensure the flag is set to False.
            loss_name: String name of the loss function that was provided in `setup_distillation_loss`.
                There must be at least one key - `primary`. There can be any number of secondary keys.
                By default, if nothing is provided, assumed as the `primary` key.
        """
        if not self.is_being_distilled():
            raise RuntimeError("Model is not being distilled, yet tensors are being registered for distillation!")

        if tensor is None:
            raise ValueError("Distillation `tensor` cannot be None !")

        # Check that the loss exists
        if loss_name not in self.distill_loss_dict:
            raise KeyError(
                f"Available distillation loss keys are : {list(self.distill_loss_dict.keys())}, "
                f"which did not match the provided key : {loss_name}"
            )

        # If module is wrapped, its registry may not be available, recreate it
        if not hasattr(self, '_distillation_registry'):
            self._distillation_registry = {}

        if loss_name not in self._distillation_registry:
            # If this is a binary argument loss function, create a list of tensors to register (unnamed args)
            # Positional arg losses can only take binary arguments and are designated by passing None to loss_key
            # For positional arg losses, create a list.
            if loss_key is None:
                self._distillation_registry[loss_name] = []
            else:
                # If loss_key is provided, consider it a kwargs based loss and instantiate a dict instead.
                self._distillation_registry[loss_name] = {}

        if loss_key is not None and loss_key in self._distillation_registry[loss_name]:
            raise ValueError(
                f"Distillation key '{loss_key}' already exists in distillation registry for "
                f"loss named : {loss_name}!"
            )

        # If the distillation config has `preserve_distillation_memory` set to True, force tensor to CPU.
        # Note that this might severely impact training speed.
        if self.distill_cfg.get('preserve_distillation_memory', False):
            tensor = tensor.cpu()

        if loss_key is None:
            # This is a positional binary loss
            self._distillation_registry[loss_name].append(tensor)
        else:
            # This is a kwarg based name
            self._distillation_registry[loss_name][loss_key] = tensor

    def distillation_registration_step(self, log_prob: torch.Tensor):
        """
        Helper method to register tensors inside of `training_step`. Subclasses should overwrite
        this method (and its signature) to suit their requirements.

        The default implementation assumes that the input is a tensor which represents log probabilities
        and the primary distillation loss is KLDivergence.

        Args:
            log_prob: A tensor of any shape (B, *) that represents log probabilities.
        """
        if self.is_student_model():
            loss_key = 'input'
        else:
            loss_key = 'target'

        # Register the tensor for the loss function
        self.register_distillation_tensor(loss_key=loss_key, tensor=log_prob)

    def reset_distillation_registry(self):
        """
        Recursively reset the registry of distillation tensors to clear up memory.
        """
        self._distillation_registry.clear()
        for _, m in self.named_modules():
            if hasattr(m, '_distillation_registry') and len(m._distillation_registry) > 0:
                m.reset_distillation_registry()

    def is_being_distilled(self) -> bool:
        """
        Utility method to check if the model is being distilled or not.

        Returns:
            True if the model distillation type is neither `student` or `teacher`, False otherwise.
        """
        return self.distill_type is not None

    def is_student_model(self) -> Optional[bool]:
        """
        Check whether the current instance of the model is undergoing teacher-student distillation,
        and if so, whether this instance is the teacher of the student.

        Returns: A bool if the current instance is a student model, or None stating that the model
            is not undergoing teacher-student distillation.
        """
        if self.is_being_distilled():
            return self.distill_type == DistillationType.STUDENT
        else:
            return None

    def validate_distillation_model(self, other_model: 'ModelPT'):
        """
        Optionally, perform validations on student model (self) and the teacher model (argument),
        such that this function must execute just after creation of the student and teacher models.
        It is called before training, at the moment after creation of both student and teacher models.

        If there is a fundamental incompatibility between the models, raise an appropriate error when
        overriding this method.

        Args:
            other_model: An instance of a ModelPT subclass that also inherits the DistillationMixin.
                When self is the `student` model, other_model is the `teacher` model and vice-versa.

        Returns:
            Nothing needs to be returned, however if there is some fundamental incompatibility between
            the student and teacher models, raise the appropriate warning/error in order to notify the user.
        """
        pass

    def prehook_primary_distillation_loss(self, loss_dict: dict, teacher_model: 'ModelPT'):
        """
        Pre-hook when computing the primary distillation loss. Modifications to the `loss_dict` here is utilized
        when computing the primary distillation loss.

        Note that the pre-hook is called only for the student model. Therefore, `self` refers to the student.

        Args:
            loss_dict: Dictionary of arguments that will be passed to the primary loss function as kwargs.
            teacher_model: The teacher model in the distillation training. To reference the student model, use `self`.
        """
        pass

    def prehook_additional_distillation_losses(
        self,
        loss_name: str,
        student_registry: Union[list, dict],
        teacher_registry: Union[list, dict],
        teacher_model: 'ModelPT',
    ):
        """
        Pre-hook when computing additional distillation losses. Modifications to the registry here is utilized
        when computing additional losses.

        Note that the pre-hook is called only for the student model. Therefore, `self` refers to the student.

        Args:
            loss_name: str name of the loss function which will be used.
            student_registry: Can be a list or a dictionary.
                When it is a list, items in it represent a tensor that will be paired with the corresponding teacher
                list, and passed to a loss function that accepts binary arguments as its input. Used primarily for
                similarity based losses.
                When it is a dictionary, represents a key-value entry that is merged with between student and teacher.
                Student and teacher should have unique keys, otherwise the merge step will overwrite either the student
                or teacher's values. The dictionary is then passed as a kwarg to the corresponding loss function.
            teacher_registry: Can be a list or a dictionary.
                When it is a list, items in it represent a tensor that will be paired with the corresponding teacher
                list, and passed to a loss function that accepts binary arguments as its input. Used primarily for
                similarity based losses.
                When it is a dictionary, represents a key-value entry that is merged with between student and teacher.
                Student and teacher should have unique keys, otherwise the merge step will overwrite either the student
                or teacher's values. The dictionary is then passed as a kwarg to the corresponding loss function.
            teacher_model: The teacher model in the distillation training. To reference the student model, use `self`.
        """
        pass

    @property
    def distillation_registry(self):
        """
        Returns:
            Returns the distillation registry or an empty dictionary if None exists
        """
        if hasattr(self, '_distillation_registry'):
            return self._distillation_registry
        else:
            return {}

    @property
    def distill_type(self):
        """
        Returns:
            Returns None if model is not being distilled, otherwise returns a DistillationType value.
        """
        global _DISTILLATION_TYPE
        return _DISTILLATION_TYPE

    @property
    def distill_cfg(self):
        """
        Returns:
            The global distillation config shared across all distillation modules.
        """
        global _DISTILLATION_CFG
        return _DISTILLATION_CFG

    @property
    def distill_loss_dict(self):
        """
        Returns:
            The global resolved dictionary of loss objects that are bound to their `loss_name`.
        """
        global _DISTILLATION_LOSS_DICT
        return _DISTILLATION_LOSS_DICT

    @classmethod
    def get_distillation_module_registry(
        cls, module: torch.nn.Module
    ) -> Dict[str, Dict[str, Union[List[Dict[str, torch.Tensor]], List[List[torch.Tensor]]]]]:
        """
        Given a module, will recursively extract in nested lists, all of the distillation registries that may exist.
        The keys of this dictionary are the flattened module names, the values are the internal distillation registry
        of each such module.

        Args:
            module: Any PyTorch Module that extends DistillationMixin.

        Returns:
            A nested dictionary with the following format:
                Dict[Key=module_flattented_name,
                     Value=Dict[Key=loss_name,
                                Value=<list of dictionaries (loss_key: tensor)>  # if keyword loss function
                                      OR
                                      <list of list of tensors>  # if binary loss function
                                ]
                     ]
        """
        module_registry = {}
        for name, m in module.named_modules():
            if hasattr(m, '_distillation_registry'):
                module_registry[name] = m._distillation_registry
        return module_registry

    @classmethod
    def flatten_distillation_module_registry(
        cls, registry: dict, loss_name: str, loss_key: Optional[str] = None,
    ) -> (Union[List[Dict[str, torch.Tensor]], List[List[torch.Tensor]]]):
        """
        Flatten the nested distillation registry obtained from a module using `get_distillation_module_registry()`.

        Args:
            registry: A nested dictionary obtained by using `get_distillation_module_registry()` on a module.
            loss_name: Name of the loss that will be be flattened out from the dictionary.
            loss_key: (Optional) The loss key that can be extracted for a given loss_name. If not provided,
                all of the loss_keys that belong to a certain loss_name will be returned.

        Returns:
            A flattened list of values represented as :
                List[
                    <dictionaries (loss_key: tensor)>  # if keyword loss function
                    OR
                    <list of tensors>  # if binary loss function
                    ]

            if the optional `loss_key` is provided and the loss_name refers to a dictionary of (loss_key: tensor),
            then the returned value is :
                List[List[tensors]]
            This is useful for cases where each module may have same key but different number of registered tensors.
        """
        flattented_registry = []
        for module_name, module_registry in registry.items():  # type: (str, dict)
            if loss_name in module_registry:
                loss_item = module_registry[loss_name]  # can be either another dict or a list

                # If user wants just one particular key from one particular loss
                # extract that loss value into a list of lists
                if type(loss_item) == dict and loss_key is not None:
                    flattented_registry.append(loss_item[loss_key])
                else:
                    # If the loss_key is not provided, then simply return either a list of lists
                    # or a list of dictionaries without further extraction.
                    flattented_registry.append(loss_item)

        return flattented_registry


class ScaledDistillationLossMixin:
    """
    Mixin class used for Distillation losses, so as to manipulate the gradient of the loss function.
    This mixin will reset the loss to 0 after it is done, so that no further gradient from the loss will be calculated
    from subsequent .backward() - say from Pytorch Lightning call.

    Refer `Distilling the Knowledge in a Neural Network` - Section 2.1 - https://arxiv.org/abs/1503.02531
    which notes gradient scaling required for KLDivergence Loss by temperature ^ 2.
    """

    def scale_gradients(self, parameters: Iterator[torch.nn.Parameter], loss: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradients, then scale them, while preserving the computation graph.

        Args:
            parameters: The parameters whose gradients need to be scaled
            loss:

        Returns:

        """
        loss.backward(retain_graph=True)

        # Scale the gradients from this loss' backward pass
        for p in parameters:
            if p.requires_grad and p.grad is not None:
                p.grad.data *= self.grad_scale

        return loss

    @property
    def grad_scale(self):
        raise NotImplementedError(
            "The class that inherits ScaledDistillationLossMixin must override the property `grad_scale`."
        )
