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

from typing import Dict, List, Optional, Union
from math import ceil
from nemo.core.classes.mixins import DistillationMixin
from nemo.utils import logging


class CTCDistillationMixin(DistillationMixin):

    def _validate_distillation_encoder_match(self, other_model: 'EncDecCTCModel'):
        if 'ConvASREncoder' in self.cfg.encoder._target_ and 'ConvASREncoder' in other_model.cfg.encoder._target_:
            self._distillation_encoder_match = 'ConvASREncoder'
            logging.info("Teacher and student models have a ConvASREncoder module as their encoder !")
        else:
            self._distillation_encoder_match = None
            logging.info("Teacher and student models do not have compatible encoders.")

    def _validate_distillation_decoder_match(self, other_model: 'EncDecCTCModel'):
        teacher_decoder_params = list(other_model.decoder.parameters())
        student_decoder_params = list(self.decoder.parameters())

        if len(teacher_decoder_params) == len(student_decoder_params):
            self._distillation_decoder_match = True
            for tp, sp in zip(teacher_decoder_params, student_decoder_params):
                if tp.data.shape != sp.data.shape:
                    self._distillation_decoder_match = False
                    break

            logging.info(
                "Decoder parameters match exactly between student and teacher models. Initializing "
                "student decoder with teacher parameters."
            )
        else:
            self._distillation_decoder_match = False
            logging.info("Decoder parameters do not match exactly between student and teacher models")

    def prehook_additional_distillation_losses(
        self,
        loss_name: str,
        student_registry: Union[list, dict],
        teacher_registry: Union[list, dict],
        teacher_model: 'EncDecCTCModel',
    ):
        if self._distillation_encoder_match == 'ConvASREncoder' and self.distill_cfg.get('distill_encoder', False):
            self._distill_additional_loss_conv_asr_encoder(student_registry, teacher_model, teacher_registry)

    def _distill_additional_loss_conv_asr_encoder(self, student_registry, teacher_model, teacher_registry):
        # ConvASREncoder compatible teacher and student models
        # Get teacher encoder's registered tensors
        # Get student encoder's registered tensors
        student_encoder_registry = self.get_distillation_module_registry(self.encoder)
        teacher_encoder_registry = self.get_distillation_module_registry(teacher_model.encoder)
        student_encoder_tensor_list = self.flatten_distillation_module_registry(
            student_encoder_registry, loss_name='cosine'
        )
        teacher_encoder_tensor_list = self.flatten_distillation_module_registry(
            teacher_encoder_registry, loss_name='cosine'
        )
        # flatten the tensor lists (across the individual sub-modules)
        student_encoder_tensor_list = [mod[0][0] for mod in student_encoder_tensor_list]
        teacher_encoder_tensor_list = [mod[0][0] for mod in teacher_encoder_tensor_list]
        num_student_layers = len(student_encoder_tensor_list)  # num student blocks
        num_teacher_layers = len(teacher_encoder_tensor_list)  # num teacher blocks
        stride = max(1, int(ceil(num_teacher_layers / num_student_layers)))

        for s_idx, student_t in enumerate(student_encoder_tensor_list):
            t_idx = min(s_idx * stride, num_teacher_layers - 1)
            teacher_t = teacher_encoder_tensor_list[t_idx]
            if student_t.shape == teacher_t.shape:
                student_registry.append(student_t)
                teacher_registry.append(teacher_t)
