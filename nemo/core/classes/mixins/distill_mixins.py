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
from typing import Optional, Union

import torch
from omegaconf import DictConfig

from nemo.collections.common.losses import CosineSimilarityLoss

_DISTILLATION_TYPE = None
_DISTILLATION_LOSS_DICT = None
_DISTILLATION_CFG = DictConfig({})


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


def set_distill_cfg(cfg: DictConfig):
    global _DISTILLATION_CFG
    _DISTILLATION_CFG = cfg


def set_distill_loss_dict(loss_dict: dict):
    global _DISTILLATION_LOSS_DICT
    _DISTILLATION_LOSS_DICT = loss_dict


class DistillationMixin(ABC):
    def __init__(self):
        super().__init__()
        self._distillation_registry = {}

    def setup_distillation_loss(self) -> Optional['torch.nn._Loss']:
        """
        If implemented by base class, in case the distillation config does not contain the 'loss' subconfig,
        the model itself can provide a default loss which will be used in its place.

        Returns:
            A dictionary of Loss object(s) that will be used as the distillation loss function.
            The dictionary must have at least 1 key - "primary" which is the primary loss function used
            for distillation.
            By default, this function returns KLDivergence loss (primary) and CosineSimilarityLoss (cosine).
            If None is returned, the distillation config must have an appropriate loss function defined.
        """
        primary = torch.nn.KLDivLoss(log_target=True, reduction='batchmean')
        cosine = CosineSimilarityLoss()
        loss_dict = {'primary': primary, 'cosine': cosine}
        return loss_dict

    def register_distillation_tensor(
        self, loss_key: str = None, tensor: torch.Tensor = None, loss_name: str = "primary",
    ):
        if not self.is_being_distilled():
            raise RuntimeError("Model is not being distilled, yet tensors are being registered for distillation")

        if tensor is None:
            raise ValueError("Distillation `tensor` cannot be None !")

        if loss_name not in self.distill_loss_dict:
            raise KeyError(
                f"Available distillation loss keys are : {list(self.distill_loss_dict)}, "
                f"which did not match the provided key : {loss_name}"
            )

        if loss_name not in self._distillation_registry:
            # If this is a similarity match loss, create a list of tensors to register (unnamed args)
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

        The default implementation assumes that the input is a tensor which represents log probabilities.

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
        self._distillation_registry.clear()

    def is_being_distilled(self) -> bool:
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

        If there is a fundamental incompatibility between the models, raise an appropriate error when
        overriding this method.

        Args:
            other_model: An instance of a ModelPT subclass that also inherits the DistillationMixin.

        Returns:
            Nothing needs to be returned, however if there is some fundamental incompatibility between
            the student and teacher models, raise the appropriate warning/error in order to notify the user.
        """
        pass

    def prehook_primary_distillation_loss(self, loss_dict: dict):
        pass

    def prehook_additional_distillation_losses(
        self, loss_name: str, student_registry: Union[list, dict], teacher_registry: Union[list, dict]
    ):
        pass

    @property
    def distill_type(self):
        global _DISTILLATION_TYPE
        return _DISTILLATION_TYPE

    @property
    def distill_cfg(self):
        global _DISTILLATION_CFG
        return _DISTILLATION_CFG

    @property
    def distill_loss_dict(self):
        global _DISTILLATION_LOSS_DICT
        return _DISTILLATION_LOSS_DICT
