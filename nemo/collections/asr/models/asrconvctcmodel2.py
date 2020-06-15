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

from nemo.collections.asr.data_layer import AudioToTextDataLayer2
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

        if spec_augment_params is not None:
            self.spec_augmentation = NeuralModuleAPI.from_config(spec_augment_params)
        else:
            self.spec_augmentation = None

    def forward(self, input_signal, input_signal_length):
        processed_signal, processed_signal_len = self.preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )
        if self.spec_augmentation is not None:
            processed_signal = self.spec_augmentation(input_spec=processed_signal)
        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_len)
        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        return log_probs, encoded_len, greedy_predictions

    def training_step(self, batch, batch_nb):
        audio_signal, audio_signal_len, transcript, transcript_len = batch
        log_probs, encoded_len, _ = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss.loss_function(
            log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len
        )

        tensorboard_logs = {'train_loss': loss_value}
        return {'loss': loss_value, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        data_loader = AudioToTextDataLayer2(
            manifest_filepath='/Users/okuchaiev/Data/an4_dataset/an4_train.json',
            labels=[
                " ",
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "x",
                "y",
                "z",
                "'",
            ],
            batch_size=32,
            trim_silence=True,
            max_duration=16.7,
            shuffle=True,
        )
        return data_loader.data_loader
