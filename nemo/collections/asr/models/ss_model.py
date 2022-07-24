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
from random import sample
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import torchaudio
import tqdm
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.data import audio_to_audio_dataset
from nemo.collections.asr.losses.ss_losses.si_snr import PermuationInvarianceWrapper
from nemo.collections.asr.models.separation_model import SeparationModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType, SpectrogramType, VoidType
from nemo.utils import logging

EPS = 1e-8

__all__ = ['EncDecSpeechSeparationModel']


class EncDecSpeechSeparationModel(SeparationModel):
    """Base class for encoder-decoder models used for Speech Separation"""

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        return []

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor = EncDecSpeechSeparationModel.from_config_dict(self._cfg.preprocessor)
        self.encoder = EncDecSpeechSeparationModel.from_config_dict(self._cfg.encoder)
        self.decoder = EncDecSpeechSeparationModel.from_config_dict(self._cfg.decoder)

        base_loss = EncDecSpeechSeparationModel.from_config_dict(self._cfg.loss.base_loss)
        if self._cfg.loss.loss_wrapper is not None:
            if self._cfg.loss.loss_wrapper == 'permutation_invariance':
                self.loss = PermuationInvarianceWrapper(base_loss)
        else:
            self.loss = base_loss

        self.num_sources = self._cfg.train_ds.num_sources

        # for future purposes
        # self.spec_augmentation = EncDecSpeechSeparationModel.from_config_dict(self._cfg.spec_augment)

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=augmentor,
        )

        shuffle = config['shuffle']

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            logging.warning("Tarred dataset not supported!")
            return None

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning("Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        dataset = audio_to_audio_dataset.get_audio_to_source_dataset(config=config, featurizer=featurizer,)

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = dataset.datasets[0].collate_fn

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

        return loader

    def setup_training_data(self, train_data_config):

        """
        Sets up the training data loader via a Dict-like object.
        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.
        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_config):

        """
        Sets up the training data loader via a Dict-like object.
        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.
        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_audio.AudioToSourceDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in val_data_config and val_data_config['is_tarred']:
            # We also need to check if limit_val_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_val_batches, float):
                self._trainer.limit_val_batches = int(
                    self._trainer.limit_val_batches
                    * ceil((len(self._validation_dl.dataset) / self.world_size) / val_data_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "validation batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_test_data(self, test_data_config):

        """
        Sets up the training data loader via a Dict-like object.
        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.
        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_audio.AudioToSourceDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in test_data_config and test_data_config['is_tarred']:
            # We also need to check if limit_val_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_val_batches, float):
                self._trainer.limit_val_batches = int(
                    self._trainer.limit_val_batches
                    * ceil((len(self._validation_dl.dataset) / self.world_size) / test_data_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "test batches will be used. Please set the trainer and rebuild the dataset."
                )

    def forward(self, mix_audio):
        """
        Forward pass of the model.

        Args:

        Returns:
    
        """
        mix_feat = self.preprocessor(mix_audio)
        mask_estimate = self.encoder(mix_feat)

        mix_feat = torch.stack([mix_feat] * self.num_sources)
        sep_feat = mix_feat * mask_estimate

        # decode
        target_estimate = torch.cat(
            [self.decoder(sep_feat[i]).unsqueeze(-1) for i in range(self.num_sources)], dim=-1,
        )

        T_original = mix_audio.size(1)
        T_estimated = target_estimate.size(1)

        if T_original > T_estimated:
            target_estimate = F.pad(target_estimate, (0, 0, 0, T_original - T_estimated))
        else:
            target_estimate = target_estimate[:, :T_original, :]

        return target_estimate

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        if self.num_sources == 2:
            input, input_len, target1, target2, sample_ids = batch
        else:
            logging.info(f"current support is only for 2 sources")

        target_estimate = self.forward(input)
        target = [target1, target2]
        target = torch.cat([target[i].unsqueeze(-1) for i in range(self.num_sources)], dim=-1,)

        loss, _ = self.loss(preds=target_estimate, targets=target)
        loss = loss.mean()

        tensorboard_logs = {'train_loss': loss, 'learning_rate': self._optimizer.param_groups[0]['lr']}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.num_sources == 2:
            input, input_len, target1, target2, sample_ids = batch
        else:
            logging.info(f"current support is only for 2 sources")
        target_estimate = self.forward(input)
        target = [target1, target2]
        target = torch.cat([target[i].unsqueeze(-1) for i in range(self.num_sources)], dim=-1,)

        loss, _ = self.loss(preds=target_estimate, targets=target)
        loss = loss.mean()

        return {
            'val_loss': loss,
        }

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': val_loss_mean}
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {'test_loss': logs['val_loss']}

        return test_logs

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': test_loss_mean}
        return {'test_loss': test_loss_mean, 'log': tensorboard_logs}

    @torch.no_grad()
    def extract_sources(
        self,
        paths2audio_files: List[str],
        save_dir: str = None,
        orig_sr: int = 16000,
        num_sources: int = 2,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        """
        Takes paths to audio files and saves separated sources
        Args:
            paths2audio_files: paths to audio fragment to be separated
            save_dir: where to save the sources
        """

        if paths2audio_files is None or len(paths2audio_files) == 0:
            raise ValueError(f"zero files received in extract_sources fn")

        if not self.num_sources == num_sources:
            raise ValueError(f"model trained for {self.num_sources} sources, but got {num_sources} sources")

        if save_dir is None:
            raise ValueError(f"save_dir is not specified")

        # Models mode and device
        mode = self.training
        device = next(self.parameters()).device

        # create save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        try:
            # switch to eval mode
            self.eval()

            # freeze modules
            self.preprocessor.freeze()
            self.encoder.freeze()
            self.decoder.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)

            # work in temp directory for manifest creation
            with tempfile.TemporaryDirectory() as tmp_dir:
                with open(os.path.join(tmp_dir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                    for audio_file in paths2audio_files:
                        entry = {
                            'audio_filepath': [audio_file, audio_file],
                            'duration': [10000, 10000],
                        }
                        fp.write(json.dumps(entry) + '\n')

                config = {
                    'manifest_filepath': os.path.join(tmp_dir, 'manifest.json'),
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': num_workers,
                    'num_sources': self.num_sources,
                    'orig_sr': orig_sr,
                    'sample_rate': self._cfg.sample_rate,
                }

                extract_dataloader = self._setup_dataloader_from_config(config)
                for batch in tqdm.tqdm(extract_dataloader, desc="Extracting sources"):
                    target_estimate = self.forward(batch[0].to(device))

                    self._save_audio(
                        id=batch[-1].cpu().item(),
                        mixture=batch[0].cpu(),
                        target_estimate=target_estimate.cpu(),
                        save_dir=save_dir,
                        sample_rate=self._cfg.sample_rate,
                    )

        finally:
            # set modes
            self.train(mode=mode)

            logging.set_verbosity(logging_level)
            if mode is True:
                self.preprocessor.unfreeze()
                self.encoder.unfreeze()
                self.decoder.unfreeze()

        return

    def _save_audio(self, id, mixture, target_estimate, save_dir, sample_rate):
        """
        Saves the test audio (mixture and estimates)
        """

        # save mixture
        samples = mixture[0, :]
        samples = samples / samples.abs().max()
        save_path = os.path.join(save_dir, f"item{id}_mix.wav")
        torchaudio.save(
            save_path, samples.unsqueeze(0), sample_rate,
        )

        # save estimated sources
        for n_src in range(self.num_sources):
            samples = target_estimate[0, :, n_src]
            samples = samples / samples.abs().max()
            save_path = os.path.join(save_dir, f"item{id}_source{n_src+1}hat.wav")
            torchaudio.save(
                save_path, samples.unsqueeze(0), sample_rate,
            )
