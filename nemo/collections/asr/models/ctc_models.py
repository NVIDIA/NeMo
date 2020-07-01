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

from typing import Dict, Optional

import torch

from nemo.collections.asr.data.audio_to_text import AudioToTextDataset
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import monitor_asr_train_progress
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.parts.features import WaveformFeaturizer
from nemo.core.classes.common import Serialization, logging, typecheck
from nemo.core.classes.optimizers import get_optimizer, parse_optimizer_args
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = ['EncDecCTCModel', 'JasperNet', 'QuartzNet']


@experimental
class EncDecCTCModel(ASRModel):
    """Encoder decoder CTC-based models."""

    def transcribe(self, path2audio_file: str) -> str:
        pass

    @staticmethod
    def __setup_dataloader_from_config(config: Optional[Dict]):
        featurizer = WaveformFeaturizer(sample_rate=config['sample_rate'], int_values=config.get('int_values', False))
        dataset = AudioToTextDataset(
            manifest_filepath=config['manifest_filepath'],
            labels=config['labels'],
            featurizer=featurizer,
            max_duration=config.get('max_duration', None),
            min_duration=config.get('min_duration', None),
            max_utts=config.get('max_utts', 0),
            blank_index=config.get('blank_index', -1),
            unk_index=config.get('unk_index', -1),
            normalize=config.get('normalize_transcripts', False),
            trim=config.get('trim_silence', True),
            load_audio=config.get('load_audio', True),
            parser=config.get('parser', 'en'),
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=config['shuffle'],
            num_workers=config.get('num_workers', 0),
        )

    def setup_training_data(self, train_data_layer_params: Optional[Dict]):
        if 'shuffle' not in train_data_layer_params:
            train_data_layer_params['shuffle'] = True
        self.__train_dl = self.__setup_dataloader_from_config(config=train_data_layer_params)

    def setup_validation_data(self, val_data_layer_params: Optional[Dict]):
        if 'shuffle' not in val_data_layer_params:
            val_data_layer_params['shuffle'] = False
        self.__val_dl = self.__setup_dataloader_from_config(config=val_data_layer_params)

    def setup_test_data(self, test_data_layer_params: Optional[Dict]):
        if 'shuffle' not in test_data_layer_params:
            test_data_layer_params['shuffle'] = False
        self.__test_dl = self.__setup_dataloader_from_config(config=test_data_layer_params)

    def setup_optimization(self, optim_params: Optional[Dict]):
        optim_params = optim_params or {}  # In case null was passed as optim_params

        # Check if caller provided optimizer name, default to Adam otherwise
        optimizer_name = optim_params.get('optimizer', 'adam')

        # Check if caller has optimizer kwargs, default to empty dictionary
        optimizer_args = optim_params.get('opt_args', [])
        optimizer_args = parse_optimizer_args(optimizer_args)

        # We are guarenteed to have lr since it is required by the argparser
        # But maybe user forgot to pass it to this function
        lr = optim_params.get('lr', None)

        if 'lr' is None:
            raise ValueError('`lr` must be passed when setting up the optimization !')

        # Actually instantiate the optimizer
        optimizer = get_optimizer(optimizer_name)
        self.__optimizer = optimizer(self.parameters(), lr=lr, **optimizer_args)

        # TODO: Remove after demonstration
        logging.info("Optimizer config = %s", str(self.__optimizer))

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
        if hasattr(self.preprocessor, '_sample_rate'):
            audio_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            audio_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), audio_eltype),
            "input_signal_length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
        }

    def __init__(
        self,
        preprocessor_params: Dict,
        encoder_params: Dict,
        decoder_params: Dict,
        spec_augment_params: Optional[Dict] = None,
    ):
        super().__init__()
        self.preprocessor = Serialization.from_config_dict(preprocessor_params)
        self.encoder = Serialization.from_config_dict(encoder_params)
        self.decoder = Serialization.from_config_dict(decoder_params)
        self.loss = CTCLoss(num_classes=self.decoder.num_classes_with_blank - 1)
        if spec_augment_params is not None:
            self.spec_augmentation = Serialization.from_config_dict(spec_augment_params)
        else:
            self.spec_augmentation = None

        # This will be set by setup_training_data
        self.__train_dl = None
        # This will be set by setup_validation_data
        self.__val_dl = None
        # This will be set by setup_test_data
        self.__test_dl = None
        # This will be set by setup_optimization
        self.__optimizer = None

    @typecheck()
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

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        self.train()
        audio_signal, audio_signal_len, transcript, transcript_len = batch
        log_probs, encoded_len, predictions = self.forward(
            input_signal=audio_signal, input_signal_length=audio_signal_len
        )
        # loss_value = self.loss.loss_function(
        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        wer, prediction, reference = monitor_asr_train_progress(
            tensors=[predictions, transcript, transcript_len], labels=self.decoder.vocabulary
        )
        tensorboard_logs = {'train_loss': loss_value, 'wer': wer}
        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        self.eval()
        audio_signal, audio_signal_len, transcript, transcript_len = batch
        log_probs, encoded_len, _ = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        # loss_value = self.loss.loss_function(
        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        tensorboard_logs = {'val_loss': loss_value}
        return {'val_loss': loss_value, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def configure_optimizers(self):
        return self.__optimizer

    def train_dataloader(self):
        return self.__train_dl

    def val_dataloader(self):
        return self.__val_dl


@experimental
class JasperNet(EncDecCTCModel):
    pass


@experimental
class QuartzNet(EncDecCTCModel):
    pass
