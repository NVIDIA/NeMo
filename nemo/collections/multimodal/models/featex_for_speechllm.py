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

from abc import ABC, abstractmethod
from typing import List

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
from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel
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


__all__ = ['RQFeatExModel', 'EmbFeatExModel', 'KmeansFeatExModel']


class FeatExBaseModel(ModelPT, AccessMixin):
    """
    Base class for parallel feature extraction 
    """  

    def list_available_models(cls) -> List[PretrainedModelInfo]:
        pass

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        pass

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        pass

    @abstractmethod
    def predict_step(self, batch):
        pass


class RQFeatExModel(FeatExBaseModel):
    """
    NeMo wrapper for RQ quantization module for parallel feature extraction 
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


class EmbFeatExModel(FeatExBaseModel):
    """
    NeMo wrapper for parallel feature extraction 
    this works only for full context size
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

        if not cfg.init_from_nemo_model.endswith('.nemo'):
            logging.error(f"init_from_nemo_model should be a .nemo file, got {cfg.init_from_nemo_model}")
            raise ValueError(f"init_from_nemo_model should be a .nemo file, got {cfg.init_from_nemo_model}")
    
        init_model = ModelPT.restore_from(restore_path=cfg.init_from_nemo_model, map_location='cpu')

        self.preprocessor = init_model.preprocessor        # [B, D, T]
        self.emb_capture_location = cfg.emb_capture_location
        if self.emb_capture_location == 'post_conv_subsample':
            logging.info(f"Extracting features from post_conv_subsample module")
            self.model = init_model.encoder.pre_encode
        elif self.emb_capture_location == 'encoder':
            logging.info(f"Extracting features from encoder module from layer {cfg.emb_capture_layer}")
            self.emb_capture_layer = cfg.emb_capture_layer
            self.model = init_model     # ptl automatically sets the model to eval mode, so dont need to do explicitly
            # NOTE: if you are using an SSL model, you have to explicitly set self.apply_masking=False
            
            # set params for access registry
            self.model.encoder.access_cfg['save_encoder_tensors'] = True
            self.model.encoder.access_cfg['detach'] = True
            self.model.encoder.set_access_enabled(access_enabled=True)

            # for ssl models
            if hasattr(self.model, 'apply_masking'):
                self.model.apply_masking = False

        else:
            logging.error(f"emb_capture_location should be either post_conv_subsample or encoder, got {cfg.emb_capture_location}")
            raise ValueError(f"emb_capture_location should be either post_conv_subsample or encoder, got {cfg.emb_capture_location}")

    def forward(self, input_signal=None, input_signal_length=None,):
        if self.emb_capture_location == 'post_conv_subsample':
            processed_signal, processed_signal_length = self.preprocessor(input_signal=input_signal, length=input_signal_length,)
            processed_signal = torch.transpose(processed_signal, 1, 2)    # [B, D, T] -> [B, T, D]
            audio_signal, audio_signal_length = self.model(x=processed_signal, lengths=processed_signal_length)
        elif self.emb_capture_location == 'encoder':
            if isinstance(self.model, SpeechEncDecSelfSupervisedModel):
                _, _, audio_signal, audio_signal_length = self.model.forward(input_signal=input_signal, input_signal_length=input_signal_length)
            else:
                audio_signal, audio_signal_length = self.model.forward(input_signal=input_signal, input_signal_length=input_signal_length)
            registry  = self.model.encoder.get_module_registry(self.model.encoder)
            audio_signal = registry[f"layers.{self.emb_capture_layer}"]['encoder'][0]
        else:
            logging.error(f"emb_capture_location should be either post_conv_subsample or encoder, got {self.emb_capture_location}")
            raise ValueError(f"emb_capture_location should be either post_conv_subsample or encoder, got {self.emb_capture_location}")
        
        return {'emb':torch.transpose(audio_signal, 1, 2), 'emb_len':audio_signal_length}   # emb is [B, D, T]
    

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)        # reset also clears access_enabled flag, so enable
            AccessMixin.set_access_enabled(access_enabled=True)


        signal, signal_len, transcript, transcript_len, sample_id = batch
        encoded = self.forward(input_signal=signal, input_signal_length=signal_len)
        
        assert len(encoded['emb'].shape) == 3, "Expected 3D codes, got {}".format(encoded['emb'].shape)
        del signal

        sample_id = sample_id.cpu().detach().numpy()
        result = []
        for i, id in enumerate(sample_id):
            result.append((id, {'emb':encoded['emb'][i][:, :encoded['emb_len'][i]].detach().cpu().numpy()}))
        return result


class KmeansFeatExModel(EmbFeatExModel):
    """
    NeMo wrapper for (EmbFeatExModel + kmeans) for parallel feature extraction 
    this works only for full context size
    """

    def __init__(
            self,
            cfg: DictConfig,
            trainer: Trainer = None,
    ):

        super().__init__(cfg=cfg, trainer=trainer)

        if cfg.centroids_filepath is None:
            logging.error(f"centroids_filepath should be provided")
            raise ValueError(f"centroids_filepath should be provided")
        
        centroids = torch.from_numpy(np.load(cfg.centroids_filepath)['centroids'])   # [num_classes, hidden_dim]
        self.centroids = nn.Parameter(centroids, requires_grad=False)

    def forward(self, input_signal=None, input_signal_length=None,):
        encoded = super().forward(input_signal=input_signal, input_signal_length=input_signal_length)
        audio_signal = encoded['emb']    # [B, D, T]
        audio_signal = audio_signal.transpose(1, 2)    # [B, D, T] -> [B, T, D]
        audio_signal_length = encoded['emb_len']

        # get token ids (xid) of shape (B, T)
        #(B, T, D) -> (B, T, D, num_classes)
        xid = audio_signal.unsqueeze(-1) - self.centroids.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        xid = xid.norm(dim=-2).argmin(dim=-1)

        return {'codes': xid, 'codes_len':audio_signal_length}
    

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)        # reset also clears access_enabled flag, so enable
            AccessMixin.set_access_enabled(access_enabled=True)

        signal, signal_len, transcript, transcript_len, sample_id = batch
        encoded = self.forward(input_signal=signal, input_signal_length=signal_len)
        
        assert len(encoded['codes'].shape) == 2, "Expected 2D codes, got {}".format(encoded['codes'].shape)
        del signal

        sample_id = sample_id.cpu().detach().numpy()
        result = []
        for i, id in enumerate(sample_id):
            codes = encoded['codes'][i][:encoded['codes_len'][i]].detach().cpu().numpy()
            # expand dims to make it compatible with acoustic tokens shape (N_codebooks, T)
            codes = np.expand_dims(codes, axis=0)
            result.append((id, {'codes': codes}))
        return result