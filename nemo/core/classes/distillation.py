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
import torch
from typing import Dict, List, Union

from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer

from nemo.core import typecheck
from nemo.core.classes.mixins import DistillationMixin, DistillationType
from nemo.core.classes.modelPT import ModelPT
from nemo.utils import logging, model_utils


class DistillationModelPT(ModelPT):
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
        if 'target' in self.distillation_cfg and self.distillation_cfg.target is not None:
            target_cls = model_utils.import_class_by_path(self.distillation_cfg.target)
        else:
            target_cls = self.teacher.__class__

        self.student = target_cls(cfg=cfg, trainer=self._trainer)  # type: ModelPT

        # Test that the two classes implement the `TeacherStudentMixin`
        # If they do implement it, initialize their internal parameters.
        if not isinstance(self.teacher, DistillationMixin):
            raise TypeError(
                f"Teacher model ({self.teacher.__class__.__name__}) must inherit {DistillationMixin.__name__}"
            )
        else:
            # Add mixin type information and instantiate mixin
            self.teacher._DISTILLATION_TYPE = DistillationType.TEACHER
            self.teacher.distillation_cfg = copy.deepcopy(self.distillation_cfg)

        if not isinstance(self.student, DistillationMixin):
            raise TypeError(
                f"Student model ({self.student.__class__.__name__}) must inherit {DistillationMixin.__name__}"
            )
        else:
            # Add mixin type information and instantiate mixin
            self.student._DISTILLATION_TYPE = DistillationType.STUDENT
            self.student.distillation_cfg = copy.deepcopy(self.distillation_cfg)

        # setup up delegation of DistillationModelPT to self.student
        self._setup_delegates()

        # reset model restoration state
        ModelPT._set_model_restore_state(is_being_restored=original_state_value)
        logging.setLevel(logging.INFO)

        # setup loss function
        self._setup_distillation_loss()

        # restore delegated data loaders (of student model only)
        if self._cfg is not None and not self._is_model_being_restored():
            if 'train_ds' in self._cfg and self._cfg.train_ds is not None:
                self.setup_training_data(self._cfg.train_ds)

            if 'validation_ds' in self._cfg and self._cfg.validation_ds is not None:
                self.setup_multiple_validation_data(val_data_config=self._cfg.validation_ds)

            if 'test_ds' in self._cfg and self._cfg.test_ds is not None:
                self.setup_multiple_test_data(test_data_config=self._cfg.test_ds)

        # ModelPT wrappers over subclass implementations
        self.training_step = model_utils.wrap_training_step(self.training_step)

        # Validate that the student and teacher models are fundamentally compatibilities
        self.teacher.validate_distillation_model(other_model=self.student)
        self.student.validate_distillation_model(other_model=self.teacher)

    def _setup_distillation_loss(self):
        loss_config = self.distillation_cfg.get('loss', None)
        loss_obj = self.student.setup_distillation_loss()

        if loss_config is None and loss_obj is None:
            raise ValueError(
                "Distillation loss could not be setup. Either override `setup_distillation_loss()` in model"
                " OR `distillation` config should have `loss` subconfig declaring the loss function "
                "for distillation. For example, the default loss config can be expressed as follows :"
                """
                 loss:
                      primary:
                          _target_: 'torch.nn.KLDivLoss'
                          log_target: true
                          reduction: 'batchmean'
                      
                      similarity:
                          _target_: 'nemo.collections.common.losses.cosine_similarity.CosineSimilarityLoss'
                """
            )

        # prioritize config over the default loss implementation
        if loss_config is not None:
            if 'primary' not in loss_config:
                raise ValueError(
                    "`loss` config must have `primary` subsection to denote the primary "
                    "knowledge distillation loss."
                )

            self.transfer_loss_primary = self.from_config_dict(loss_config.primary)

            self.loss_dict = {'primary': self.transfer_loss_primary}

            for loss_key in loss_config.keys():
                if loss_key != 'primary':
                    self.loss_dict[loss_key] = self.from_config_dict(loss_config[loss_key])

        else:
            if type(loss_obj) == dict:
                # Assume implementation correctly returns a dictionary of losses
                if 'primary' not in loss_obj:
                    raise ValueError(
                        "setup_distillation_loss() must return a dictionary with at least one key - `primary` "
                        "to denote the primary knowledge distillation loss."
                    )

            elif isinstance(loss_obj, torch.nn.Module):
                # Consider the single loss returned as the primary loss and pack into dict
                loss_obj = {'primary': loss_obj}

            else:
                raise ValueError(
                    "setup_distillation_loss() should return a dictionary with at least one loss which "
                    "has the key - `primary`, and any number of additional losses."
                )

            self.transfer_loss_primary = loss_obj['primary']
            self.loss_dict = {'primary': self.transfer_loss_primary}

            for loss_key in loss_obj.keys():
                if loss_key != 'primary':
                    self.loss_dict[loss_key] = loss_obj[loss_key]

        # Attach the loss dict to teacher and student
        self.teacher._distillation_loss_dict = self.loss_dict
        self.student._distillation_loss_dict = self.loss_dict
        logging.info(f"Distillation losses registered : {list(self.loss_dict.keys())}")

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
        self.teacher.training_step(batch=batch, batch_nb=batch_nb)
        student_outputs = self.student.training_step(batch=batch, batch_nb=batch_nb)

        # Compute primary loss (required) and similarity losses (optional)
        primary_loss_value, additional_losses = self._compute_loss()
        self.log('distillation_train_loss', primary_loss_value)

        # Add similarity matching losses to primary loss with weighted term
        if additional_losses is not None and len(additional_losses) > 0:
            # For all additional losses
            similarity_loss_log_idx = 1
            similarity_loss_weight = self.distillation_cfg.get('similarity_loss_weight', 1.0)

            for ix, (additional_loss_name, additional_loss_val) in enumerate(additional_losses.items()):

                # if additional loss is a similarity loss (a positional loss)
                if len(additional_loss_val) > 1:
                    for jx, similarity_loss_val in enumerate(additional_loss_val):
                        similarity_loss_val = similarity_loss_weight * similarity_loss_val
                        primary_loss_value += similarity_loss_val

                        self.log(
                            f'{additional_loss_name}_subid_{similarity_loss_log_idx}', similarity_loss_val
                        )
                        similarity_loss_log_idx += 1

                else:
                    # if additional loss is a kwarg type loss
                    additional_loss_val = additional_loss_val[0]  # only 1 item for dictionary based losses
                    primary_loss_value += additional_loss_val

                    self.log(f'{additional_loss_name}', additional_loss_val)

        # Clearup resources
        if additional_losses is not None:
            del additional_losses

        # Reset references to tensors which were registered
        self.teacher.reset_distillation_registry()
        self.student.reset_distillation_registry()

        # Optionally, add student train loss along with distillation loss
        if (
            'student_train_loss_weight' in self.distillation_cfg
            and self.distillation_cfg.student_train_loss_weight > 0.0
        ):
            if student_outputs is None:
                raise RuntimeError(
                    "During distillation, student did not return any loss value for its " "`training_step`"
                )

            student_train_loss = student_outputs['loss']
            student_train_loss_weight = self.distillation_cfg.student_train_loss_weight

            distillation_loss_weight = self.distillation_cfg.get('distillation_loss_weight', 1.0)
            primary_loss_value = (
                distillation_loss_weight * primary_loss_value + student_train_loss_weight * student_train_loss
            )

        return {'loss': primary_loss_value}

    def _compute_loss(self):
        # Update the registry from both student and teacher models
        teacher_registry = self.teacher._distillation_registry
        student_registry = self.student._distillation_registry

        # Extract the primary loss dictionary
        primary_loss_dict = teacher_registry.pop('primary')  # type: dict
        primary_loss_dict.update(student_registry.pop('primary'))  # type: dict

        # Check that tensors were registered for distillation loss calculation
        if len(primary_loss_dict) == 0:
            raise RuntimeError(
                "No tensors were registered in order to compute the distillation loss.\n"
                "Use self.register_distillation_tensor(loss_key, tensor) to register tensors!"
            )

        # Call prehook_primary_distillation_loss() of student
        self.student.prehook_primary_distillation_loss(loss_dict=primary_loss_dict)

        # Compute primary distillation loss
        loss_value = self.transfer_loss_primary(**primary_loss_dict)

        # Clear memory of primary loss
        del primary_loss_dict

        # For all subsequent loss tensors that may have been registered
        additional_losses = {}

        for loss_name in teacher_registry.keys():
            if loss_name not in student_registry:
                raise KeyError(f"Student distillation registry did not contain loss named {loss_name}")

            # Assign new key name for this additional loss
            additional_loss_key = f"distillation_additional_loss_{loss_name}_id_{len(additional_losses)}"

            # Additional loss can hold 1 or more loss results
            additional_losses[additional_loss_key] = []

            # Extract the loss object from both student and teacher
            # It is not yet known if its a positional loss (nameless) or a kwarg loss
            teacher_loss_obj = teacher_registry.get(loss_name)
            student_loss_obj = student_registry.get(loss_name)

            if type(teacher_loss_obj) == list:
                # This is a binary positional loss

                # Extract list of tensors from teacher and student whose similarity must be matched 1:1
                teacher_tensor_list = teacher_loss_obj
                student_tensor_list = student_loss_obj

                if len(teacher_tensor_list) != len(student_tensor_list):
                    raise ValueError(
                        f"Tensors were registered for similarity loss ({loss_name}), but "
                        f"number of registered teacher tensors ({len(teacher_tensor_list)}) "
                        f"does not match the number of registered student tensors ("
                        f"{len(student_tensor_list)})!"
                    )

                # Call prehook_similarity_matching_loss() of student
                self.student.prehook_additional_distillation_losses(
                    loss_name=loss_name, student_registry=student_tensor_list, teacher_registry=teacher_tensor_list,
                )

                # Disable typechecking and compute losses via direct function call
                binary_positional_loss_fn = self.loss_dict[loss_name]

                # Disable type checking for positional loss function call
                with typecheck.disable_checks():
                    for student_tensor, teacher_tensor in zip(student_tensor_list, teacher_tensor_list):
                        additional_loss = binary_positional_loss_fn(student_tensor, teacher_tensor)
                        additional_losses[additional_loss_key].append(additional_loss)

            else:
                # This is a kwarg loss

                # Extract dict of tensors from teacher and student
                teacher_tensor_dict = teacher_loss_obj
                student_tensor_dict = student_loss_obj

                # Call prehook_similarity_matching_loss() of student
                self.student.prehook_additional_distillation_losses(
                    loss_name=loss_name, student_registry=student_tensor_dict, teacher_registry=teacher_tensor_dict,
                )

                additional_loss_dict = teacher_tensor_dict
                additional_loss_dict.update(student_tensor_dict)

                # Compute additional distillation loss
                additional_loss = self.transfer_loss_primary(**additional_loss_dict)

                additional_losses[additional_loss_key].append(additional_loss)

        return loss_value, additional_losses

    def save_to(self, save_path: str):
        """ Delegate save_to to student model """
        self.student.save_to(save_path=save_path)

    @classmethod
    def restore_from(
        cls, **kwargs,
    ):
        """ Prevent restoration of teacher-student model - it is not preserved """
        raise NotImplementedError(
            f"Restoration of {DistillationModelPT.__name__} is invalid,"
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
    # setup up delegation of DistillationModelPT methods to self.student
    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        return self.student.setup_training_data(train_data_config)

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        return self.student.setup_validation_data(val_data_config)

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        return self.student.setup_test_data(test_data_config)

    def teardown(self, stage: str):
        self.teacher.teardown(stage)
        return self.student.teardown(stage)
