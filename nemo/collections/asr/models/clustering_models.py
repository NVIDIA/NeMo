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
import json
import os
import tempfile
from math import ceil
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from tqdm.auto import tqdm

from nemo.collections.asr.data import feature_to_label_dataset
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.parts.feature_loader import ExternalFeatureLoader
from nemo.collections.asr.parts.perturb import process_augmentations
from nemo.collections.common.losses import CrossEntropyLoss, SmoothedCrossEntropyLoss
from nemo.collections.common.metrics import TopKClassificationAccuracy
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    AudioSignal,
    LabelsType,
    LengthsType,
    LogitsType,
    LogprobsType,
    MaskType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging

__all__ = ['EncDecClusteringModel']


class EncDecClusteringModel(ASRModel, ExportableEncDecModel):
    """Base class for encoder decoder CTC-based models."""

    # TODO list available models

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set byt LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_gpus

        super().__init__(cfg=cfg, trainer=trainer)
        # self.preprocessor = EncDecClusteringModel.from_config_dict(self._cfg.preprocessor)
        self.encoder = EncDecClusteringModel.from_config_dict(self._cfg.encoder)

        # Conformer
        # with open_dict(self._cfg):
        #     if "feat_in" not in self._cfg.decoder or (
        #         not self._cfg.decoder.feat_in and hasattr(self.encoder, '_feat_out')
        #     ):
        #         self._cfg.decoder.feat_in = self.encoder._feat_out
        #     if "feat_in" not in self._cfg.decoder or not self._cfg.decoder.feat_in:
        #         raise ValueError("param feat_in of the decoder's config is not set!")

        self.decoder = EncDecClusteringModel.from_config_dict(self._cfg.decoder)
        # self.log_softmax = TokenClassifier(
        #     hidden_size=self.decoder.hidden_size,
        #     num_classes=self.decoder.vocab_size,
        # )

        # self.loss = CrossEntropyLoss(logits_ndim=3)
        self.loss = SmoothedCrossEntropyLoss(pad_id=-1)

        
        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecClusteringModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        # Setup metric objects
        self._accuracy = TopKClassificationAccuracy(dist_sync_on_step=True, top_k=[1])

        self.restricted=self._cfg.decoder.restricted

    # TODO add trancribe function like in asr
    @torch.no_grad()
    def transcribe(
        self,
        paths2audio_files: List[str],
        batch_size: int = 4,
        logprobs: bool = False,
        return_hypotheses: bool = False,
    ) -> List[str]:

        pass

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']
        device = 'gpu' if torch.cuda.is_available() else 'cpu'

        feature_loader = ExternalFeatureLoader(
            file_path=config['manifest_filepath'], sample_rate=16000, int_values=False, augmentor=None
        )

        if config.get('is_speaker_emb', False):
            # directly load stored external embedding

            dataset = feature_to_label_dataset.get_feature_seq_speakerlabel_dataset(
                feature_loader=feature_loader, config=config
            )

            batch_size = config['batch_size']
            collate_func = dataset.collate_fn

        # [TODO] Instantiate tarred dataset loader or normal dataset loader
        # [TODO] load from other type of input
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of a Neural Clustering Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.feature_to_label.FeatureToSeqSpeakerLabelDataset `
            -   :class:`~nemo.collections.asr.data.feature_to_label.FeatureToSeqSpeakerLabelDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # [TODO] add tarred support

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of a Neural Clustering Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.feature_to_label.FeatureToSeqSpeakerLabelDataset `
            -   :class:`~nemo.collections.asr.data.feature_to_label.FeatureToSeqSpeakerLabelDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.

        Args:
            test_data_config: A config that contains the information regarding construction
                of a Neural Clustering Training dataset.

                Supported Datasets:
            -   :class:`~nemo.collections.asr.data.feature_to_label.FeatureToSeqSpeakerLabelDataset `
            -   :class:`~nemo.collections.asr.data.feature_to_label.FeatureToSeqSpeakerLabelDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        # [TODO] add more input data type
        # Support extracted embedding now

        # if config.get('is_speaker_emb', False):
        #     # no processed_signal
        #     input_signal_eltype = AcousticEncodedRepresentation()
        return {
            # [TODO] check optional and input emb shape!!
            "input_signal": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation(), optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "label_seq": NeuralType(('B', 'T'), LabelsType(), optional=True),
        }

    # TODO check if greedy prediction related to beam search
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "outputs_label_pad": NeuralType(('B', 'T'), LabelsType()),
            # "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            # "greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
        }

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        label_seq=None,
    ):
        # [TODO] only take 'preprocessed' emb now

        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None

        # TODO add processed later
        processed_signal, processed_signal_length = input_signal, input_signal_length

        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        # if not has_processed_signal:
        #     processed_signal, processed_signal_length = self.preprocessor(
        #         input_signal=input_signal, length=input_signal_length,
        #     )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal)

        ## Comformer!!!
        # encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        # let try conformert encoder
        # src_hiddens, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        # src_hiddens = torch.transpose(src_hiddens, 1, 2)
        # print("conformer encoder src_hiddens.shape", src_hiddens.shape)

        # ## Transformer
        # src_mask = (~make_pad_mask(processed_signal_length.tolist())).to(processed_signal.device).unsqueeze(-2)
        # src_mask = torch.squeeze(src_mask)

        # Might need to clean up the logic here
        if self._cfg.decoder._target_ == "nemo.collections.asr.modules.TransformerDecoderNM":

            # BDT -> transformer -> BTD -> (BTD) transformer -> BTD
            src_hiddens, length, src_mask = self.encoder(input_ids=processed_signal)

            # BDT -> conformer -> BDT -> (BTD) transformer -> BTD
            if self._cfg.encoder._target_ != "nemo.collections.asr.modules.TransformerEncoderNM":
                src_hiddens = torch.transpose(src_hiddens, 1, 2)

            logits, ys_out = self.decoder(
                input_ids=label_seq,
                encoder_embeddings=src_hiddens,  # output of the encoder (B x L_enc x H)
                encoder_mask=src_mask,  # encoder inputs mask (B x L_enc)
            )

        else:
            # BDT -> conformer -> BDT -> (BDT) lstm/conv -> BTD
            encoded, length = self.encoder(input_ids=processed_signal)

            # BDT -> transformer -> BTD -> (BDT) lstm -> BTD
            if self._cfg.encoder._target_ == "nemo.collections.asr.modules.TransformerEncoderNM":
                encoded = torch.transpose(encoded, 1, 2)

            logits = self.decoder(encoder_output=encoded, return_logits=True)
            ys_out = label_seq

        # return softmax log resutls  TODO might select top k results

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        # return ys_out_pad as well for loss and accuracy calculation
        return log_probs, ys_out

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # signal is feature. speaker embedding now
        signal, signal_len, label_seq, label_seq_len = batch
        signal = signal.transpose(1, 2)  # convert (B, T, D) to (B, D, T)

        log_probs, ys_out = self.forward(input_signal=signal, input_signal_length=signal_len, label_seq=label_seq)

        # something wrong with ysmask
        # log_probs = log_probs[:,:50,:]
        loss_value = self.loss(log_probs=log_probs, labels=ys_out)
        tensorboard_logs = {'train_loss': loss_value, 'learning_rate': self._optimizer.param_groups[0]['lr']}

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        log_probs_flatten = torch.flatten(log_probs, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(ys_out, start_dim=0, end_dim=-1)

        # whether logits or log_probas is fine for accuracy
        # TODO update TopKClassificationAccuracy
        self._accuracy(logits=log_probs_flatten, labels=labels_flatten)
        topk_scores = self._accuracy.compute()

        tensorboard_logs = {'train_loss': loss_value, 'learning_rate': self._optimizer.param_groups[0]['lr']}

        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            tensorboard_logs.update({'training_batch_accuracy_top@{}'.format(top_k): score})

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, label_seq, label_seq_len = batch
        signal = signal.transpose(1, 2)  # convert (B, T, D) to (B, D, T)

        log_probs, ys_out = self.forward(input_signal=signal, input_signal_length=signal_len, label_seq=label_seq)
        loss_value = self.loss(log_probs=log_probs, labels=ys_out)

        log_probs_flatten = torch.flatten(log_probs, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(ys_out, start_dim=0, end_dim=-1)

        acc = self._accuracy(logits=log_probs_flatten, labels=labels_flatten)
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k
        acc = [correct_counts.float() / total_counts]

        return {
            'val_loss': loss_value,
            'val_correct_counts': correct_counts,
            'val_total_counts': total_counts,
            'val_acc': acc,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {
            'test_loss': logs['val_loss'],
            'test_correct_counts': logs['val_correct_counts'],
            'test_total_counts': logs['val_total_counts'],
            'test_acc': logs['val_acc'],
        }
        return test_logs

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_correct_counts = torch.stack([x['val_correct_counts'] for x in outputs]).sum()
        val_total_counts = torch.stack([x['val_total_counts'] for x in outputs]).sum()
        tensorboard_logs = {'val_loss': val_loss_mean, 'val_acc': val_correct_counts / val_total_counts}
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_correct_counts = torch.stack([x['test_correct_counts'] for x in outputs]).sum()
        val_total_counts = torch.stack([x['val_total_counts'] for x in outputs]).sum()
        tensorboard_logs = {'test_loss': test_loss_mean, 'test_wer': test_correct_counts / val_total_counts}
        return {'test_loss': val_loss_mean, 'log': tensorboard_logs}

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    # [TODO] transcribe dataloder
