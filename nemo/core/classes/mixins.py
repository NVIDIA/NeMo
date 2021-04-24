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


class TeacherStudentType(Enum):
    STUDENT = 1
    TEACHER = 2


class TeacherStudentMixin(ABC):
    def __init__(self):
        super().__init__()
        self._TEACHER_STUDENT_TYPE: Optional[TeacherStudentType] = None
        self.distillation_cfg = DictConfig({})
        self._distillation_primary_registry = {}

    def is_being_distilled(self):
        return self._TEACHER_STUDENT_TYPE is not None

    def is_student_model(self) -> Optional[bool]:
        """
        Check whether the current instance of the model is undergoing teacher-student distillation,
        and if so, whether this instance is the teacher of the student.

        Returns: A bool if the current instance is a student model, or None stating that the model
            is not undergoing teacher-student distillation.
        """
        if self.is_being_distilled():
            return self._TEACHER_STUDENT_TYPE == TeacherStudentType.STUDENT
        else:
            return None

    def default_distillation_loss_config(self) -> Optional[DictConfig]:
        """
        If implemented by base class, in case the distillation config does not contain the 'loss' subconfig,
        the model itself can provide a default loss config which will be instantiated.

        Returns:
            An optional DictConfig that will be used to instantiate the distillation loss function.
            If None is returned, the distillation config must have an appropriate loss function defined.
        """
        return None

    def register_distillation_tensor(self, loss_key: str, tensor: torch.Tensor):
        if not self.is_being_distilled():
            raise RuntimeError("Model is not being distilled, yet tensors are being registered for distillation")

        if loss_key in self._distillation_primary_registry:
            raise ValueError(f"Distillation key '{loss_key}' already exists in distillation registry!")

        self._distillation_primary_registry[loss_key] = tensor

    def reset_distillation_registry(self):
        self._distillation_primary_registry.clear()
