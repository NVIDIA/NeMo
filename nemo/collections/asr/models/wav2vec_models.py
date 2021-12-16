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

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch import nn

from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel
from nemo.core.classes.common import PretrainedModelInfo, typecheck


class Wav2VecEncoderModel(SpeechEncDecSelfSupervisedModel):
    """
    Model class for Wav2Vec style model feature encoder as in 
    Baevski et al. See: https://arxiv.org/pdf/2006.11477.pdf
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        self.feat_penalty = (
            cfg.model_defaults.feature_penalty
        )  # Check if we add L2 regularization of feature encodings
        if self.feat_penalty:
            # Run time editing of forward function
            self.loss.pen = 0.0
            self.loss.contrastive_loss = self.loss.forward
            self.loss.forward = lambda **kwargs: self.loss.contrastive_loss(**kwargs) + self.loss.pen

        # Dropouts and norms
        self.dropout_features = nn.Dropout(cfg.model_defaults.dropout_features)
        self.dropout_features_q = nn.Dropout(cfg.model_defaults.dropout_features_q)

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        """
        Forward pass of the model.
        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.
            Note: processed_signal and processed_signal_length are dummy values and will throw error if passed.
        Returns:
            A tuple of 3 elements -
            1) Processed audio signal features of shape [B, C, T].
            2) Masks applied to features of shape [B, C, T].
            3) Decoder outputs of shape [B, T, C].
        """

        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if has_processed_signal:
            raise ValueError(f"{self} does not allow DALI preprocessing. Please provide raw audio inputs.")

        # B, C, T
        features, feature_length = self.preprocessor(  # Feature convolution is aliased as a NeMo preprocessor
            input_signal=input_signal, length=input_signal_length,
        )

        if self.feat_penalty:  # Apply L2 regularization to feature projections
            self.loss.pen = features.float().pow(2).mean() * self.feat_penalty

        # B, C, T
        unmasked_features = features.clone()  # These will be used for the loss function

        features = self.dropout_features(features)
        unmasked_features = self.dropout_features_q(unmasked_features)

        # SpeAug is alias for masking operation
        # B, C, T
        features = self.spec_augmentation(input_spec=features, length=feature_length)

        # Indexes locations of mask and padding
        feature_masks = torch.logical_and(features < 1e-5, features > -1e-5).float()
        for idx, proc_len in enumerate(feature_length):
            feature_masks[idx, :, proc_len:] = 0.0

        # B, C, T, For compatibility, encoder outputs length as second var.
        logits, _ = self.encoder(features, feature_length)

        # B, T, C
        logits = self.decoder_ssl(encoder_output=logits)

        return unmasked_features, feature_masks, logits

    def get_features(self, input_signal, input_signal_length):
        """Returns only feature encodings"""
        with torch.no_grad():
            return self.preprocessor(input_signal=input_signal, length=input_signal_length)
