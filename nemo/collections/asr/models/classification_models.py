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
import os
from typing import Dict, List, Optional, Union

import onnx
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.asr.data.audio_to_text import AudioLabelDataset
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.parts.features import WaveformFeaturizer
from nemo.collections.asr.parts.perturb import process_augmentations
from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.common.metrics import TopKClassificationAccuracy
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.export_utils import attach_onnx_to_onnx

__all__ = ['EncDecClassificationModel', 'MatchboxNet']


class EncDecClassificationModel(ASRModel, Exportable):
    """Encoder decoder CTC-based models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self._update_decoder_config(self.cfg.decoder)

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
        self._accuracy = TopKClassificationAccuracy(dist_sync_on_step=True)

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
            pin_memory=config.get('pin_memory', False),
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
    def list_available_models(cls) -> Optional[List[PretrainedModelInfo]]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        model = PretrainedModelInfo(
            pretrained_model_name="MatchboxNet-3x1x64-v1",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/MatchboxNet-3x1x64-v1.nemo",
            description="MatchboxNet model trained on Google Speech Commands dataset (v1, 30 classes) which obtains 97.32% accuracy on test set.",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="MatchboxNet-3x2x64-v1",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/MatchboxNet-3x2x64-v1.nemo",
            description="MatchboxNet model trained on Google Speech Commands dataset (v1, 30 classes) which obtains 97.68% accuracy on test set.",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="MatchboxNet-3x1x64-v2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/MatchboxNet-3x1x64-v2.nemo",
            description="MatchboxNet model trained on Google Speech Commands dataset (v2, 35 classes) which obtains 97.12% accuracy on test set.",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="MatchboxNet-3x1x64-v2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/MatchboxNet-3x1x64-v2.nemo",
            description="MatchboxNet model trained on Google Speech Commands dataset (v2, 30 classes) which obtains 97.29% accuracy on test set.",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="MatchboxNet-3x1x64-v2-subset-task",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/MatchboxNet-3x1x64-v2-subset-task.nemo",
            description="MatchboxNet model trained on Google Speech Commands dataset (v2, 10+2 classes) which obtains 98.2% accuracy on test set.",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="MatchboxNet-3x2x64-v2-subset-task",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/MatchboxNet-3x2x64-v2-subset-task.nemo",
            description="MatchboxNet model trained on Google Speech Commands dataset (v2, 10+2 classes) which obtains 98.4% accuracy on test set.",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="MatchboxNet-VAD-3x2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/MatchboxNet_VAD_3x2.nemo",
            description="Voice Activity Detection MatchboxNet model trained on google speech command (v2) and freesound background data, which obtains 0.992 accuracy on testset from same source and 0.852 TPR for FPR=0.315 on testset (ALL) of AVA movie data",
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

        self._accuracy(logits=logits, labels=labels)
        top_k = self._accuracy.compute()
        for i, top_i in enumerate(top_k):
            tensorboard_logs[f'training_batch_accuracy_top@{i}'] = top_i

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels)
        acc = self._accuracy(logits=logits, labels=labels)
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k
        return {
            'val_loss': loss_value,
            'val_correct_counts': correct_counts,
            'val_total_counts': total_counts,
            'val_acc': acc,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels)
        acc = self._accuracy(logits=logits, labels=labels)
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k
        return {
            'test_loss': loss_value,
            'test_correct_counts': correct_counts,
            'test_total_counts': total_counts,
            'test_acc': acc,
        }

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct_counts = torch.stack([x['val_correct_counts'] for x in outputs])
        total_counts = torch.stack([x['val_total_counts'] for x in outputs])

        self._accuracy.correct_counts_k = correct_counts
        self._accuracy.total_counts_k = total_counts
        topk_scores = self._accuracy.compute()

        tensorboard_log = {'val_loss': val_loss_mean}
        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            tensorboard_log['val_epoch_top@{}'.format(top_k)] = score

        return {'log': tensorboard_log}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        correct_counts = torch.stack([x['test_correct_counts'].unsqueeze(0) for x in outputs])
        total_counts = torch.stack([x['test_total_counts'].unsqueeze(0) for x in outputs])

        self._accuracy.correct_counts_k = correct_counts
        self._accuracy.total_counts_k = total_counts
        topk_scores = self._accuracy.compute()

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

            # Update config
            self._cfg.labels = new_labels

            decoder_config = self.decoder.to_config_dict()
            new_decoder_config = copy.deepcopy(decoder_config)
            self._update_decoder_config(new_decoder_config)
            del self.decoder
            self.decoder = EncDecClassificationModel.from_config_dict(new_decoder_config)

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

    def _update_decoder_config(self, cfg):
        """
        Update the number of classes in the decoder based on labels provided.

        Args:
            cfg: The config of the decoder which will be updated.
        """
        OmegaConf.set_struct(cfg, False)

        labels = self.cfg.labels

        if 'params' in cfg:
            cfg.params.num_classes = len(labels)
        else:
            cfg.num_classes = len(labels)

        OmegaConf.set_struct(cfg, True)

    def export(
        self,
        output: str,
        input_example=None,
        output_example=None,
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        onnx_opset_version: int = 12,
        try_script: bool = False,
        set_eval: bool = True,
        check_trace: bool = True,
        use_dynamic_axes: bool = True,
    ):
        if input_example is not None or output_example is not None:
            logging.warning(
                "Passed input and output examples will be ignored and recomputed since"
                " EncDecClassificationModel consists of two separate models (encoder and decoder) with different"
                " inputs and outputs."
            )

        encoder_onnx = self.encoder.export(
            os.path.join(os.path.dirname(output), 'encoder_' + os.path.basename(output)),
            None,  # computed by input_example()
            None,
            verbose,
            export_params,
            do_constant_folding,
            keep_initializers_as_inputs,
            onnx_opset_version,
            try_script,
            set_eval,
            check_trace,
            use_dynamic_axes,
        )

        decoder_onnx = self.decoder.export(
            os.path.join(os.path.dirname(output), 'decoder_' + os.path.basename(output)),
            None,  # computed by input_example()
            None,
            verbose,
            export_params,
            do_constant_folding,
            keep_initializers_as_inputs,
            onnx_opset_version,
            try_script,
            set_eval,
            check_trace,
            use_dynamic_axes,
        )

        output_model = attach_onnx_to_onnx(encoder_onnx, decoder_onnx, "EDC")
        onnx.save(output_model, output)


class MatchboxNet(EncDecClassificationModel):
    pass
