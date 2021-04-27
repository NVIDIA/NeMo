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
from enum import Enum
from typing import Optional

import torch
from omegaconf import DictConfig


class DistillationType(Enum):
    STUDENT = 1
    TEACHER = 2


class DistillationMixin(ABC):
    def __init__(self):
        super().__init__()
        self._DISTILLATION_TYPE: Optional[DistillationType] = None
        self.distillation_cfg = DictConfig({})
        self._distillation_registry_primary = {}

    def setup_distillation_loss(self) -> Optional['torch.nn._Loss']:
        """
        If implemented by base class, in case the distillation config does not contain the 'loss' subconfig,
        the model itself can provide a default loss which will be used in its place.

        Returns:
            An optional Loss object that will be used as the distillation loss function.
            If None is returned, the distillation config must have an appropriate loss function defined.
        """
        return None

    def register_distillation_tensor(self, loss_key: str, tensor: torch.Tensor):
        if not self.is_being_distilled():
            raise RuntimeError("Model is not being distilled, yet tensors are being registered for distillation")

        if loss_key in self._distillation_registry_primary:
            raise ValueError(f"Distillation key '{loss_key}' already exists in distillation registry!")

        self._distillation_registry_primary[loss_key] = tensor

    def reset_distillation_registry(self):
        self._distillation_registry_primary.clear()

    def is_being_distilled(self) -> bool:
        return self._DISTILLATION_TYPE is not None

    def is_student_model(self) -> Optional[bool]:
        """
        Check whether the current instance of the model is undergoing teacher-student distillation,
        and if so, whether this instance is the teacher of the student.

        Returns: A bool if the current instance is a student model, or None stating that the model
            is not undergoing teacher-student distillation.
        """
        if self.is_being_distilled():
            return self._DISTILLATION_TYPE == DistillationType.STUDENT
        else:
            return None

    def validate_distillation_model(self, teacher_model: 'ModelPT'):
        """
        Optionally, perform validations on student model (self) and the teacher model (argument),
        such that this function must execute just after creation of the student and teacher models.

        If there is a fundamental incompatibility between the models, raise an appropriate error when
        overriding this method.

        Args:
            teacher_model: An instance of a ModelPT subclass that also inherits the DistillationMixin.

        Returns:
            Nothing needs to be returned, however if there is some fundamental incompatibility between
            the student and teacher models, raise the appropriate warning/error in order to notify the user.
        """
        pass

    def prehook_primary_distillation_loss(self, loss_dict: dict):
        pass
