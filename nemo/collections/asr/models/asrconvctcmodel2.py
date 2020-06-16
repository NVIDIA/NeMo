# -*- coding: utf-8 -*-

from typing import Dict, Optional

# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================
import torch
from pytorch_lightning import LightningModule
from nemo.collections.asr.helpers import monitor_asr_train_progress2
from nemo.collections.asr.losses import CTCLoss2
from nemo.core.apis import NeuralModelAPI, NeuralModuleAPI


class ASRConvCTCModel(LightningModule, NeuralModelAPI):
    @classmethod
    def from_cloud(cls, name: str):
        pass

    def __init__(
        self,
        preprocessor_params: Dict,
        encoder_params: Dict,
        decoder_params: Dict,
        spec_augment_params: Optional[Dict] = None,
    ):
        super().__init__()
        self.preprocessor = NeuralModuleAPI.from_config(preprocessor_params)
        self.encoder = NeuralModuleAPI.from_config(encoder_params)
        self.decoder = NeuralModuleAPI.from_config(decoder_params)
        self.loss = CTCLoss2(num_classes=self.decoder._num_classes - 1)
        self.__train_dl = None
        self.__val_dl = None
        if spec_augment_params is not None:
            self.spec_augmentation = NeuralModuleAPI.from_config(spec_augment_params)
        else:
            self.spec_augmentation = None

    def forward(self, input_signal, input_signal_length):
        # # Non-typed old-fashioned way
        # processed_signal, processed_signal_len = self.preprocessor(
        #     input_signal=input_signal, length=input_signal_length,
        # )
        # if self.spec_augmentation is not None:
        #     processed_signal = self.spec_augmentation(input_spec=processed_signal)
        # encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_len)
        # log_probs = self.decoder(encoder_output=encoded)
        # greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        # return log_probs, encoded_len, greedy_predictions

        # Typed way -- good for "production-ready"
        processed_signal, processed_signal_len = self.preprocessor.typed_forward(
            input_signal=input_signal, length=input_signal_length,
        )
        if self.spec_augmentation is not None:
            processed_signal = self.spec_augmentation.typed_forward(input_spec=processed_signal)
        encoded, encoded_len = self.encoder.typed_forward(audio_signal=processed_signal, length=processed_signal_len)
        # log_probs = self.decoder.typed_forward(encoder_output=processed_signal)
        log_probs = self.decoder.typed_forward(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        return log_probs, encoded_len, greedy_predictions

    def save_to(self, save_path: str, optimize_for_deployment=False):
        print("TODO: Implement Me")

    def restore_from(cls, restore_path: str):
        print("TODO: Implement Me")

    def training_step(self, batch, batch_nb):
        self.train()
        audio_signal, audio_signal_len, transcript, transcript_len = batch
        log_probs, encoded_len, predictions = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss.loss_function(
            log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len
        )
        wer, prediction, reference = monitor_asr_train_progress2(tensors=[predictions, transcript, transcript_len],
                                                                 labels=self.decoder.vocabulary)
        tensorboard_logs = {'train_loss': loss_value, 'wer': wer}
        return {'loss': loss_value, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return self.__train_dl.data_loader

    def setup_training_data(self, train_data_layer_params):
        """
        Setups data loader to be used in training
        Args:
            train_data_layer_params: training data layer parameters.
        Returns:

        """
        self.__train_dl = NeuralModuleAPI.from_config(train_data_layer_params)

    def validation_step(self, batch, batch_idx):
        self.eval()
        audio_signal, audio_signal_len, transcript, transcript_len = batch
        log_probs, encoded_len, _ = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss.loss_function(
            log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len
        )
        tensorboard_logs = {'val_loss': loss_value}
        return {'val_loss': loss_value, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def val_dataloader(self):
        return self.__val_dl.data_loader

    def setup_validation_data(self, val_data_layer_params):
        """
        Setups data loader to be used in validation
        Args:
            val_data_layer_params: validation data layer parameters.
        Returns:

        """
        self.__val_dl = NeuralModuleAPI.from_config(val_data_layer_params)
