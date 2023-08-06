# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""
A NeMo wrapper for any module so that it can be used in parallel feature extraction 
for speech llm models
"""

from math import ceil
from typing import Dict, List, Optional, Union

import torch
import numpy as np
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.parts.mixins import ASRModuleMixin
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.modules.ssl_modules.quantizers import RandomProjectionVectorQuantizer
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin, set_access_cfg
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    AudioSignal,
    LabelsType,
    LengthsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging

__all__ = ['FeatExWrapperModel']

class FeatExWrapperModel(ModelPT, AccessMixin):
    """
    A NeMo wrapper for any module so that it can be used in parallel feature extraction 
    """

    def __init__(
            self,
            cfg: DictConfig,
            trainer: Trainer = None,
    ):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices
        
        super().__init__(cfg=cfg, trainer=trainer)

        self.preprocessor = ModelPT.from_config_dict(self._cfg.preprocessor)        # [B, D, T]
        self.encoder = ModelPT.from_config_dict(self._cfg.quantizer)

        if (self._cfg.quantizer.get('init_path', None) is None) and (self._cfg.quantizer.get('save_quantizer_params_path', None) is not None):
            torch.save(self.encoder.state_dict(), self._cfg.quantizer.save_quantizer_params_path)
            # proj_matrix = self.encoder.proj.weight.detach().cpu().numpy()
            # codebooks = self.encoder.codebooks.detach().cpu().numpy()
            # np.savez(self._cfg.quantizer.save_quantizer_params_path, proj_matrix=proj_matrix, codebooks=codebooks)


    def forward(self, input_signal=None, input_signal_length=None,):
        processed_signal, processed_signal_length = self.preprocessor(input_signal=input_signal, length=input_signal_length,)
        emb, codes, codes_length = self.encoder.forward(input_signal=processed_signal, input_signal_length=processed_signal_length,)
        return {'emb':emb, 'codes':codes, 'codes_len':codes_length}


    def list_available_models(cls) -> List[PretrainedModelInfo]:
        pass

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        pass

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        pass
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, _, _, sample_id = batch

        encoded = self.forward(input_signal=signal, input_signal_length=signal_len)
        assert len(encoded['codes'].shape) == 2, "Expected 2D codes, got {}".format(encoded['codes'].shape)
        del signal

        sample_id = sample_id.cpu().detach().numpy()
        result = []
        for i, id in enumerate(sample_id):
            result.append((id, {'codes':encoded['codes'][i][:encoded['codes_len'][i]].cpu().detach().numpy()}))
        return result
