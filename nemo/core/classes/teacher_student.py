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

import copy
import inspect
import os
import shutil
import tarfile
import tempfile
from abc import abstractmethod
from dataclasses import is_dataclass
from os import path
from typing import Callable, Dict, List, Optional, Union

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only

from nemo.core import optim
from nemo.core.classes.common import Model
from nemo.core.classes.modelPT import ModelPT
from nemo.core.optim import prepare_lr_scheduler
from nemo.collections.common import losses
from nemo.utils import logging, model_utils
from nemo.utils.app_state import AppState
from nemo.utils.get_rank import is_global_rank_zero
from nemo.core.classes.mixins import TeacherStudentMixin


class TeacherStudentModelPT(ModelPT):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # Extract teacher model file
        if 'teacher_model_path' not in cfg:
            raise ValueError("Provided config must have a string path to a .nemo file which represents the model.")

        # extract teacher model path
        with open_dict(cfg):
            teacher_model_path = cfg.pop('teacher_model_path')

        # prevent data loaders from being loaded for either student of teacher model
        original_state_value = ModelPT._is_model_being_restored()
        ModelPT._set_model_restore_state(is_being_restored=True)
        logging.setLevel(logging.ERROR)

        # initialize model config (from here on out, self.cfg == self.student.cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        # Initialize teacher model and freeze it
        self.teacher = ModelPT.restore_from(teacher_model_path, map_location='cpu', strict=True)  # type: ModelPT
        self.teacher.set_trainer(self._trainer)
        self.teacher.freeze()

        # Initialize student model (student model must be of same class as teacher, unless overriden)
        if 'target' in cfg:
            target_cls = model_utils.import_class_by_path(cfg.target)
        else:
            target_cls = self.teacher.__class__

        self.student = target_cls(cfg=cfg, trainer=self._trainer)  # type: ModelPT

        if not isinstance(self.teacher, TeacherStudentMixin):
            raise TypeError(
                f"Teacher model ({self.teacher.__class__.__name__}) must inherit {TeacherStudentMixin.__name__}"
            )

        if not isinstance(self.student, TeacherStudentMixin):
            raise TypeError(
                f"Student model ({self.student.__class__.__name__}) must inherit {TeacherStudentMixin.__name__}"
            )

        # setup up delegation of TeacherStudentModelPT to self.student
        self._setup_delegates()

        # reset model restoration state
        ModelPT._set_model_restore_state(is_being_restored=original_state_value)
        logging.setLevel(logging.INFO)

        # setup loss function
        self._setup_loss_function()

        # restore delegated data loaders (of student model only)
        if self._cfg is not None and not self._is_model_being_restored():
            if 'train_ds' in self._cfg and self._cfg.train_ds is not None:
                self.setup_training_data(self._cfg.train_ds)

            if 'validation_ds' in self._cfg and self._cfg.validation_ds is not None:
                self.setup_multiple_validation_data(val_data_config=None)

            if 'test_ds' in self._cfg and self._cfg.test_ds is not None:
                self.setup_multiple_test_data(test_data_config=None)

        # ModelPT wrappers over subclass implementations
        self.training_step = model_utils.wrap_training_step(self.training_step)

    def _setup_loss_function(self):
        # self.loss = losses.SmoothedCrossEntropyLoss(
        #     pad_id=0, label_smoothing=self.cfg.get('teacher_student_label_smoothing', 0.0),
        # )
        with open_dict(self.cfg):
            self.loss_reduction = self.cfg.pop('teacher_student_loss_reduction', 'batchmean')

        if 'mean' in self.loss_reduction:  # always force batchmean for both `mean` and `batchmean`
            reduction = 'batchmean'
        else:
            reduction = self.loss_reduction

        self.loss = torch.nn.KLDivLoss(reduction=reduction, log_target=True)

    def _setup_delegates(self):
        # setup up delegation of TeacherStudentModelPT methods to self.student
        self.setup_multiple_validation_data = self.student.setup_multiple_validation_data
        self.setup_multiple_test_data = self.student.setup_multiple_test_data
        self.setup_optimization = self.student.setup_optimization

        # Misc methods
        self.extract_state_dict_from = self.student.extract_state_dict_from
        self.prepare_test = self.student.prepare_test
        self.set_trainer = self.student.set_trainer
        self.set_world_size = self.student.set_world_size
        self.get_validation_dataloader_prefix = self.student.get_validation_dataloader_prefix
        self.get_test_dataloader_prefix = self.student.get_test_dataloader_prefix

        # PTL dataloader delegates
        self.train_dataloader = self.student.train_dataloader
        self.val_dataloader = self.student.val_dataloader
        self.test_dataloader = self.student.test_dataloader

        # PTL step delegates
        self.validation_step = self.student.validation_step
        self.validation_step_end = self.student.validation_step_end
        self.validation_epoch_end = self.student.validation_epoch_end
        self.test_step = self.student.test_step
        self.test_step_end = self.student.test_step_end
        self.test_epoch_end = self.student.test_epoch_end
        self.multi_validation_epoch_end = self.student.multi_validation_epoch_end
        self.multi_test_epoch_end = self.student.multi_test_epoch_end

        # Forward all student PTL logging calls to self
        self.student.log = self.log
        self.student.log_dict = self.log_dict

        # Misc PTL delegates
        self.teardown = self.student.teardown
        self.configure_optimizers = self.student.configure_optimizers

    # Abstract methods must be explicitly overridden, even if just delegates
    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        return self.student.setup_training_data(train_data_config)

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        return self.student.setup_validation_data(val_data_config)

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        return self.student.setup_test_data(test_data_config)

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        self.teacher.eval()
        teacher_logprobs = self.teacher.get_logits(batch=batch, batch_nb=batch_nb)
        student_logprobs = self.student.get_logits(batch=batch, batch_nb=batch_nb)

        if teacher_logprobs.shape != student_logprobs.shape:
            raise TypeError(f"Teacher model provided logprobs of shape {teacher_logprobs.shape}\n"
                            f"Student model provided logprobs of shape {student_logprobs.shape}\n"
                            f"As the two shapes do not match, the student and teacher model are incompatible.")

        loss_value = self.loss(
            input=student_logprobs, target=teacher_logprobs
        )
        tensorboard_logs = {'train_loss': loss_value, 'learning_rate': self.student._optimizer.param_groups[0]['lr']}

        return {'loss': loss_value, 'log': tensorboard_logs}

    def save_to(self, save_path: str):
        """ Delegate save_to to student model """
        self.student.save_to(save_path=save_path)

    def restore_from(
        cls, **kwargs,
    ):
        """ Prevent restoration of teacher-student model - it is not preserved """
        raise NotImplementedError(
            f"Restoration of {TeacherStudentModelPT.__name__} is invalid,"
            f"only student model should be saved and restored."
        )

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        # No pretrained models of TeacherStudent model are possible.
        return []
