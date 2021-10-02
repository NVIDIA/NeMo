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

from math import ceil
from typing import Union

from nemo.core.classes.mixins import DistillationMixin
from nemo.utils import logging


class CTCDistillationMixin(DistillationMixin):
    """
    Distillation Mixin specialization for CTC based ASR models.
    """

    def setup_distillation_loss(self):
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

    def distillation_registration_step(self, decoder: DistillationMixin):
        """
        Helper method to register tensors inside of `training_step`.

        Args:
            decoder: A ConvASRDecoder decoder which implements DistillationMixin and registers
                a `target` and `input` tensors during distillation.
        """
        if not isinstance(decoder, DistillationMixin):
            raise RuntimeError(
                f"Decoder {decoder.__class__.__name__}` does not implement DistillationMixin."
                f"PLease extend it and return `target` and `input` tensors for KLDivergence loss."
            )

        # Extract the registry from the decoder, then flatten its `primary` loss dictionary.
        # Flatten method always returns a list, even for single items, so remove the first element.
        registry = decoder.get_distillation_module_registry(decoder)
        registry = self.flatten_distillation_module_registry(registry, loss_name='primary')[0]

        for loss_key, tensor in registry.items():
            # Register the tensor for the loss function
            self.register_distillation_tensor(loss_key=loss_key, tensor=tensor)

    def _validate_distillation_encoder_match(self, other_model: 'EncDecCTCModel'):
        """
        Utility method to check if asr model encoder is of certain type, and if so, try to perform encoder specific
        distillation.

        Args:
            other_model: When `self` = student model, `other_model` = teacher model.
                When `self` = teacher model, `other_model` = student model.
        """
        # Specialize for ConvASREncoder type encoder.
        if 'ConvASREncoder' in self.cfg.encoder._target_ and 'ConvASREncoder' in other_model.cfg.encoder._target_:
            self._distillation_encoder_match = 'ConvASREncoder'
            logging.info(
                "Teacher and student models have a ConvASREncoder module as their encoder !\n"
                "Encoder distillation will be attempted if teacher-student models shapes are compatible."
            )
        else:
            self._distillation_encoder_match = None
            logging.info("Teacher and student models do not have compatible encoders.")

    def _validate_distillation_decoder_match(self, other_model: 'EncDecCTCModel'):
        """
        Utility method to check if asr model decoder is of certain type, and if so, try to perform decoder specific
        distillation.

        Args:
            other_model: When `self` = student model, `other_model` = teacher model.
                When `self` = teacher model, `other_model` = student model.
        """
        # Check if number of weight matrices of both student and teacher decoders is the same.
        teacher_decoder_params = list(other_model.decoder.parameters())
        student_decoder_params = list(self.decoder.parameters())

        loss_cfg = self.distill_cfg.get('loss', {})
        if len(teacher_decoder_params) == len(student_decoder_params) and 'cosine' in loss_cfg:
            # If number of weight matrices is same, attempt to perform decoder weight matrix distillation
            self._distillation_decoder_match = True

            for tp, sp in zip(teacher_decoder_params, student_decoder_params):
                if tp.data.shape != sp.data.shape:
                    # Weight matrices mismatched, failed to perform decoder weight matrix distillation
                    self._distillation_decoder_match = False
                    break

            if self._distillation_decoder_match:
                logging.info("Decoder parameters match exactly between student and teacher models")
            else:
                logging.info("Decoder parameters do not match exactly between student and teacher models, "
                             "due to parameter shape mismatch")
        else:
            # Number of weight matrices are different, cannot perform weight matrix distillation
            self._distillation_decoder_match = False
            logging.info("Decoder parameters do not match exactly between student and teacher models")

    def prehook_additional_distillation_losses(
        self,
        loss_name: str,
        student_registry: Union[list, dict],
        teacher_registry: Union[list, dict],
        teacher_model: 'EncDecCTCModel',
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
        if loss_name == 'cosine':
            if self._distillation_encoder_match == 'ConvASREncoder' and self.distill_cfg.get('distill_encoder', False):
                self._distill_additional_loss_conv_asr_encoder(student_registry, teacher_model, teacher_registry)

    def _distill_additional_loss_conv_asr_encoder(self, student_registry, teacher_model, teacher_registry):
        """
        Utility method to perform ConvASREncoder specific joint loss distillation training.

        For ConvASREncoder models, will extract the intermediate outputs per Block between teacher and student.
        Then performing near linear mapping between teacher and student to compute cosine embedding loss.

        Args:
            student_registry:
            teacher_model:
            teacher_registry:

        Returns:

        """
        # ConvASREncoder compatible teacher and student models
        # Get student encoder's registered tensors
        student_encoder_registry = self.get_distillation_module_registry(self.encoder)
        # Get teacher encoder's registered tensors
        teacher_encoder_registry = self.get_distillation_module_registry(teacher_model.encoder)

        # Flatten student registry, extracting just the nested list of tensors registered to cosine loss.
        student_encoder_tensor_list = self.flatten_distillation_module_registry(
            student_encoder_registry, loss_name='cosine'
        )
        # Flatten student registry, extracting just the nested list of tensors registered to cosine loss.
        teacher_encoder_tensor_list = self.flatten_distillation_module_registry(
            teacher_encoder_registry, loss_name='cosine'
        )

        # The above nested lists describe a nested
        # List of blocks (# of jasper blocks) of List of Sub blocks (only last sub block) of tensors
        # Flatten the tensor lists (across the individual sub-modules)
        student_encoder_tensor_list = [mod[0] for mod in student_encoder_tensor_list]  # only last sub-block available
        teacher_encoder_tensor_list = [mod[0] for mod in teacher_encoder_tensor_list]  # only last sub-block available

        # Distribute the teacher layers across the student layers
        num_student_layers = len(student_encoder_tensor_list)  # num student blocks
        num_teacher_layers = len(teacher_encoder_tensor_list)  # num teacher blocks
        stride = num_teacher_layers / float(num_student_layers)

        # for each student block
        for s_idx, student_t in enumerate(student_encoder_tensor_list):
            # find closest teacher block index (clamp at max number of teacher blocks)
            t_idx = max(0, int(ceil(s_idx * stride)))
            t_idx = min(t_idx, num_teacher_layers - 1)

            # select teacher block
            teacher_t = teacher_encoder_tensor_list[t_idx]

            # If student and teacher block shapes match, then perform distillation
            if student_t.shape == teacher_t.shape:
                student_registry.append(student_t)
                teacher_registry.append(teacher_t)
