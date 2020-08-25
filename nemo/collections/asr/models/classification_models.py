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

import copy
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.asr.data.audio_to_text import AudioLabelDataset
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.parts.features import WaveformFeaturizer
from nemo.collections.asr.parts.perturb import process_augmentations
from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.common.metrics import TopKClassificationAccuracy, compute_topk_accuracy
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = ['EncDecClassificationModel', 'MatchboxNet']


@experimental
class EncDecClassificationModel(ASRModel):
    """Encoder decoder CTC-based models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor = EncDecClassificationModel.from_config_dict(self._cfg.preprocessor)
        self.encoder = EncDecClassificationModel.from_config_dict(self._cfg.encoder)
        self.decoder = EncDecClassificationModel.from_config_dict(self._cfg.decoder)
        self.loss = CrossEntropyLoss()
        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecClassificationModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None
        if hasattr(self._cfg, 'crop_or_pad_augment') and self._cfg.crop_or_pad_augment is not None:
            self.crop_or_pad = EncDecClassificationModel.from_config_dict(self._cfg.crop_or_pad_augment)
        else:
            self.crop_or_pad = None

        # Setup metric objects
        self._accuracy = TopKClassificationAccuracy()

    def transcribe(self, paths2audio_files: str) -> str:
        raise NotImplementedError("Classification models do not transcribe audio.")

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if config.get('manifest_filepath') is None:
            return

        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=augmentor
        )
        dataset = AudioLabelDataset(
            manifest_filepath=config['manifest_filepath'],
            labels=config['labels'],
            featurizer=featurizer,
            max_duration=config.get('max_duration', None),
            min_duration=config.get('min_duration', None),
            trim=config.get('trim_silence', True),
            load_audio=config.get('load_audio', True),
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=config['shuffle'],
            num_workers=config.get('num_workers', 0),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True
        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False
        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False
        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        model = PretrainedModelInfo(
            pretrained_model_name="MatchboxNet-3x1x64-v1",
            location="https://nemo-public.s3.us-east-2.amazonaws.com/nemo-1.0.0alpha-tests/MatchboxNet-3x1x64-v1.nemo",
            description="MatchboxNet model trained on Google Speech Commands dataset (v1, 30 classes) "
            "which obtains 97.32% accuracy on test set.",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="MatchboxNet-3x2x64-v1",
            location="https://nemo-public.s3.us-east-2.amazonaws.com/nemo-1.0.0alpha-tests/MatchboxNet-3x2x64-v1.nemo",
            description="MatchboxNet model trained on Google Speech Commands dataset (v1, 30 classes) "
            "which obtains 97.68% accuracy on test set.",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="MatchboxNet-3x1x64-v2",
            location="https://nemo-public.s3.us-east-2.amazonaws.com/nemo-1.0.0alpha-tests/MatchboxNet-3x1x64-v2.nemo",
            description="MatchboxNet model trained on Google Speech Commands dataset (v2, 35 classes) "
            "which obtains 97.12% accuracy on test set.",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="MatchboxNet-3x1x64-v2",
            location="https://nemo-public.s3.us-east-2.amazonaws.com/nemo-1.0.0alpha-tests/MatchboxNet-3x2x64-v2.nemo",
            description="MatchboxNet model trained on Google Speech Commands dataset (v2, 30 classes) "
            "which obtains 97.29% accuracy on test set.",
        )
        result.append(model)
        return result

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
        return {"outputs": NeuralType(('B', 'D'), LogitsType())}

    @typecheck()
    def forward(self, input_signal, input_signal_length):
        processed_signal, processed_signal_len = self.preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )
        # Crop or pad is always applied
        if self.crop_or_pad is not None:
            processed_signal, processed_signal_len = self.crop_or_pad(
                input_signal=processed_signal, length=processed_signal_len
            )
        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal)
        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_len)
        logits = self.decoder(encoder_output=encoded)
        return logits

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        self.training_step_end()
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels)

        tensorboard_logs = {
            'train_loss': loss_value,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
        }

        correct_counts, total_counts = self._accuracy(logits=logits, labels=labels)

        for ki in range(correct_counts.shape[-1]):
            correct_count = correct_counts[ki]
            total_count = total_counts[ki]
            top_k = self._accuracy.top_k[ki]

            tensorboard_logs['training_batch_accuracy_top@{}'.format(top_k)] = correct_count / float(total_count)

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels)
        correct_counts, total_counts = self._accuracy(logits=logits, labels=labels)
        return {'val_loss': loss_value, 'val_correct_counts': correct_counts, 'val_total_counts': total_counts}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels)
        correct_counts, total_counts = self._accuracy(logits=logits, labels=labels)
        return {'test_loss': loss_value, 'test_correct_counts': correct_counts, 'test_total_counts': total_counts}

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct_counts = torch.stack([x['val_correct_counts'] for x in outputs])
        total_counts = torch.stack([x['val_total_counts'] for x in outputs])

        topk_scores = compute_topk_accuracy(correct_counts, total_counts)

        tensorboard_log = {'val_loss': val_loss_mean}
        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            tensorboard_log['val_epoch_top@{}'.format(top_k)] = score

        return {'log': tensorboard_log}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        correct_counts = torch.stack([x['test_correct_counts'].unsqueeze(0) for x in outputs])
        total_counts = torch.stack([x['test_total_counts'].unsqueeze(0) for x in outputs])

        topk_scores = compute_topk_accuracy(correct_counts, total_counts)

        tensorboard_log = {'test_loss': test_loss_mean}
        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            tensorboard_log['test_epoch_top@{}'.format(top_k)] = score

        return {'log': tensorboard_log}

    def change_labels(self, new_labels: List[str]):
        """
        Changes labels used by the decoder model. Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another dataset.

        If new_labels == self.decoder.vocabulary then nothing will be changed.

        Args:

            new_labels: list with new labels. Must contain at least 2 elements. Typically, \
            this is set of labels for the dataset.

        Returns: None

        """
        if new_labels is not None and not isinstance(new_labels, ListConfig):
            new_labels = ListConfig(new_labels)

        if self._cfg.labels == new_labels:
            logging.warning(
                f"Old labels ({self._cfg.labels}) and new labels ({new_labels}) match. Not changing anything"
            )
        else:
            if new_labels is None or len(new_labels) == 0:
                raise ValueError(f'New labels must be non-empty list of labels. But I got: {new_labels}')

            decoder_config = self.decoder.to_config_dict()
            new_decoder_config = copy.deepcopy(decoder_config)
            new_decoder_config['params']['num_classes'] = len(new_labels)
            del self.decoder
            self.decoder = EncDecClassificationModel.from_config_dict(new_decoder_config)

            # Update config
            self._cfg.labels = new_labels

            OmegaConf.set_struct(self._cfg.decoder, False)
            self._cfg.decoder = new_decoder_config
            OmegaConf.set_struct(self._cfg.decoder, True)

            if 'train_ds' in self._cfg and self._cfg.train_ds is not None:
                self._cfg.train_ds.labels = new_labels

            if 'validation_ds' in self._cfg and self._cfg.validation_ds is not None:
                self._cfg.validation_ds.labels = new_labels

            if 'test_ds' in self._cfg and self._cfg.test_ds is not None:
                self._cfg.test_ds.labels = new_labels

            logging.info(f"Changed decoder output to {self.decoder.num_classes} labels.")


@experimental
class MatchboxNet(EncDecClassificationModel):
    pass
