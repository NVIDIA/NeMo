# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


__all__ = ['CTCModel', 'JasperNet', 'QuartzNet']

from abc import ABC
from typing import Dict, Optional

from nemo.collections.asr.losses.ctc import CTCLossNM
from nemo.collections.asr.metrics.wer import monitor_asr_train_progress
from nemo.collections.asr.models.asr_model import IASRModel
from nemo.core.classes.common import INMSerialization
from nemo.core.neural_types import NeuralType


class EncDecCTCModel(IASRModel):
    """Encoder decoder CTC-based models."""

    def transcribe(self, path2audio_file: str) -> str:
        pass

    def setup_training_data(self, train_data_layer_params: Optional[Dict]):
        pass

    def setup_test_data(self, test_data_layer_params: Optional[Dict]):
        pass

    def setup_optimization(self, optim_params: Optional[Dict]):
        pass

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        pass

    def export(self, **kwargs):
        pass

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.preprocessor.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        # TODO: Write me
        return None

    def __init__(
        self,
        preprocessor_params: Dict,
        encoder_params: Dict,
        decoder_params: Dict,
        spec_augment_params: Optional[Dict] = None,
    ):
        super().__init__()
        self.preprocessor = INMSerialization.from_config_dict(preprocessor_params)
        self.encoder = INMSerialization.from_config_dict(encoder_params)
        self.decoder = INMSerialization.from_config_dict(decoder_params)
        self.loss = CTCLossNM(num_classes=self.decoder.num_classes_with_blank - 1)
        if spec_augment_params is not None:
            self.spec_augmentation = INMSerialization.from_config_dict(spec_augment_params)
        else:
            self.spec_augmentation = None

        self.__train_dl = None
        self.__val_dl = None

    # TODO: typing decorator should go here
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
        self.train()
        audio_signal, audio_signal_len, transcript, transcript_len = batch
        log_probs, encoded_len, predictions = self.forward(
            input_signal=audio_signal, input_signal_length=audio_signal_len
        )
        loss_value = self.loss.loss_function(
            log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len
        )
        wer, prediction, reference = monitor_asr_train_progress(
            tensors=[predictions, transcript, transcript_len], labels=self.decoder.vocabulary
        )
        tensorboard_logs = {'train_loss': loss_value, 'wer': wer}
        return {'loss': loss_value, 'log': tensorboard_logs}


class JasperNet(EncDecCTCModel):
    pass


class QuartzNet(EncDecCTCModel):
    pass
