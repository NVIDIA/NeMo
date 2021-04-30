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

from nemo.collections.common.losses import CosineSimilarityLoss


class DistillationType(Enum):
    STUDENT = 1
    TEACHER = 2


class DistillationMixin(ABC):
    def __init__(self):
        super().__init__()
        self._DISTILLATION_TYPE: Optional[DistillationType] = None
        self.distillation_cfg = DictConfig({})
        self._distillation_registry_primary = {}
        self._distillation_registry_similarity = []

    def setup_distillation_loss(self) -> Optional['torch.nn._Loss']:
        """
        If implemented by base class, in case the distillation config does not contain the 'loss' subconfig,
        the model itself can provide a default loss which will be used in its place.

        Returns:
            An optional Loss object that will be used as the distillation loss function.
            By default, this is the KLDivergence loss.
            If None is returned, the distillation config must have an appropriate loss function defined.
        """
        primary = torch.nn.KLDivLoss(log_target=True, reduction='batchmean')
        secondary = CosineSimilarityLoss()
        return (primary, secondary)

    def register_distillation_tensor(
        self, loss_key: str = None, tensor: torch.Tensor = None, similarity_match: bool = False
    ):
        if not self.is_being_distilled():
            raise RuntimeError("Model is not being distilled, yet tensors are being registered for distillation")

        if tensor is None:
            raise ValueError("Distillation `tensor` cannot be None !")

        if loss_key is not None and loss_key in self._distillation_registry_primary:
            raise ValueError(f"Distillation key '{loss_key}' already exists in distillation registry!")

        if not similarity_match:
            self._distillation_registry_primary[loss_key] = tensor
        else:
            self._distillation_registry_similarity.append(tensor)

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
        self._distillation_registry_primary.clear()
        self._distillation_registry_similarity.clear()

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

    def prehook_similarity_matching_loss(self, student_tensors: list, teacher_tensors: list):
        pass
