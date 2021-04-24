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
from typing import Dict, List, Union

from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer

from nemo.core.classes.mixins import TeacherStudentMixin, TeacherStudentType
from nemo.core.classes.modelPT import ModelPT
from nemo.utils import logging, model_utils


class TeacherStudentModelPT(ModelPT):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # Extract distillation config
        if 'distillation' not in cfg:
            raise ValueError(
                "Provided config must have a `distillation` subconfig, with at least one member,"
                "`model_path` or `model_name`."
            )

        # Perform checks teacher model file
        if 'model_path' not in cfg.distillation and 'model_name' not in cfg.distillation:
            raise ValueError(
                "Provided distillation config must have a string path to a .nemo file "
                "which represents the model (via `model_path`) or a name of a pretrained model"
                "which will be used as the teacher (via `model_name`)."
            )

        if 'model_path' in cfg.distillation and 'model_name' in cfg.distillation:
            raise ValueError("Only one of `model_path` or `model_name` should be passed to the teacher config !")

        # Extract teacher config completely from student config
        with open_dict(cfg):
            self.distillation_cfg = cfg.pop('distillation')

        # prevent data loaders from being loaded for either student of teacher model
        original_state_value = ModelPT._is_model_being_restored()
        ModelPT._set_model_restore_state(is_being_restored=True)
        logging.setLevel(logging.ERROR)

        # initialize model config (from here on out, self.cfg == self.student.cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        # Initialize teacher model and freeze it
        if 'model_path' in self.distillation_cfg and self.distillation_cfg.model_path is not None:
            self.teacher = ModelPT.restore_from(
                self.distillation_cfg.model_path, map_location='cpu', strict=True
            )  # type: ModelPT
        else:
            # Assume pretrained model name
            raise NotImplementedError("model_name is not supported yet as a teacher value.")

        # Freeze the
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
        else:
            self.teacher._TEACHER_STUDENT_TYPE = TeacherStudentType.TEACHER
            self.teacher.distillation_cfg = copy.deepcopy(self.distillation_cfg)

        if not isinstance(self.student, TeacherStudentMixin):
            raise TypeError(
                f"Student model ({self.student.__class__.__name__}) must inherit {TeacherStudentMixin.__name__}"
            )
        else:
            self.student._TEACHER_STUDENT_TYPE = TeacherStudentType.STUDENT
            self.student.distillation_cfg = copy.deepcopy(self.distillation_cfg)

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
        loss_config = self.distillation_cfg.get('loss', None)

        if loss_config is None:
            loss_config = self.student.default_distillation_loss_config()

        if loss_config is None:
            raise ValueError(
                "`teacher` config should have `loss` subconfig declaring the loss function "
                "for distillation. For example, KLDiv loss can be expressed as follows :"
                """
                 loss:
                      _target_: 'torch.nn.KLDivLoss'
                      log_target: true
                      reduction: 'batchmean'
                """
            )

        self.transfer_loss_primary = self.from_config_dict(loss_config)

        # TODO: use config to setup secondary loss(es)

    def forward_delegate(self, fn):
        fn_name = fn.__name__
        # preserve original self function
        setattr(self, fn_name + "_base", getattr(self, fn_name))

        # override self func
        setattr(self, fn_name, getattr(self.student, fn_name))
        delegated_fn = getattr(self, fn_name)
        return delegated_fn

    def _setup_delegates(self):
        # ModelPT Data loader methods
        self.setup_multiple_validation_data = self.forward_delegate(self.student.setup_multiple_validation_data)
        self.setup_multiple_test_data = self.forward_delegate(self.student.setup_multiple_test_data)
        self.setup_optimization = self.forward_delegate(self.student.setup_optimization)

        # Misc methods
        self.extract_state_dict_from = self.forward_delegate(self.student.extract_state_dict_from)
        self.prepare_test = self.forward_delegate(self.student.prepare_test)
        self.set_trainer = self.forward_delegate(self.student.set_trainer)
        self.set_world_size = self.forward_delegate(self.student.set_world_size)
        self.get_validation_dataloader_prefix = self.forward_delegate(self.student.get_validation_dataloader_prefix)
        self.get_test_dataloader_prefix = self.forward_delegate(self.student.get_test_dataloader_prefix)

        # PTL dataloader delegates
        self.train_dataloader = self.forward_delegate(self.student.train_dataloader)
        self.val_dataloader = self.forward_delegate(self.student.val_dataloader)
        self.test_dataloader = self.forward_delegate(self.student.test_dataloader)

        # PTL step delegates
        self.validation_step = self.forward_delegate(self.student.validation_step)
        self.validation_step_end = self.forward_delegate(self.student.validation_step_end)
        self.validation_epoch_end = self.forward_delegate(self.student.validation_epoch_end)
        self.test_step = self.forward_delegate(self.student.test_step)
        self.test_step_end = self.forward_delegate(self.student.test_step_end)
        self.test_epoch_end = self.forward_delegate(self.student.test_epoch_end)
        self.multi_validation_epoch_end = self.forward_delegate(self.student.multi_validation_epoch_end)
        self.multi_test_epoch_end = self.forward_delegate(self.student.multi_test_epoch_end)

        # Misc PTL delegates
        self.configure_optimizers = self.forward_delegate(self.student.configure_optimizers)

        # Backward delegate all student PTL logging calls to self
        self.student.log = self.log
        self.student.log_dict = self.log_dict

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        self.teacher.freeze()
        self.teacher.reset_distillation_registry()
        self.student.reset_distillation_registry()

        # Delegate train steps, dynamically replacing self with self.student / self.teacher to maintain model
        # level unawareness.
        self.teacher.__class__.training_step(self.teacher, batch=batch, batch_nb=batch_nb)
        self.student.__class__.training_step(self.student, batch=batch, batch_nb=batch_nb)

        # TODO: Maybe add hooks for student model to override
        # if teacher_logprobs.shape != student_logprobs.shape:
        #     raise TypeError(f"Teacher model provided logprobs of shape {teacher_logprobs.shape}\n"
        #                     f"Student model provided logprobs of shape {student_logprobs.shape}\n"
        #                     f"As the two shapes do not match, the student and teacher model are incompatible.")

        # Update the registry from both student and teacher models
        primary_loss_dict = self.teacher._distillation_primary_registry  # type: dict
        primary_loss_dict.update(self.student._distillation_primary_registry)  # type: dict

        # Compute primary distillation loss
        loss_value = self.transfer_loss_primary(**primary_loss_dict)
        tensorboard_logs = {'distillation_train_loss': loss_value}

        # TODO: Maybe add support for intermediate activation matching

        # Reset references to tensors which were registered
        self.teacher.reset_distillation_registry()
        self.student.reset_distillation_registry()

        return {'loss': loss_value, 'log': tensorboard_logs}

    def save_to(self, save_path: str):
        """ Delegate save_to to student model """
        self.student.save_to(save_path=save_path)

    @classmethod
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

    # Abstract methods must be explicitly overridden, even if just delegates
    # setup up delegation of TeacherStudentModelPT methods to self.student
    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        return self.student.setup_training_data(train_data_config)

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        return self.student.setup_validation_data(val_data_config)

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        return self.student.setup_test_data(test_data_config)

    def teardown(self, stage: str):
        self.teacher.teardown(stage)
        return self.student.teardown(stage)
