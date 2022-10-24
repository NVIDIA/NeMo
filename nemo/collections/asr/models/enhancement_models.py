# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import json
import os
import tempfile
from typing import Dict, List, Optional, Union

import librosa
import soundfile as sf
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from tqdm import tqdm

from nemo.collections.asr.data import audio_to_audio_dataset, audio_to_text_dataset
from nemo.collections.asr.models.audio_processing_model import AudioProcessingModel
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = ['EncMaskDecAudioProcessingModel']


@experimental
class EncMaskDecAudioProcessingModel(AudioProcessingModel):
    """Class for encoder-mask-decoder audio processing models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        super().__init__(cfg=cfg, trainer=trainer)
        self.sample_rate = self._cfg.sample_rate
        self.num_outputs = self._cfg.num_outputs
        self.encoder = EncMaskDecAudioProcessingModel.from_config_dict(self._cfg.encoder)
        self.mask = EncMaskDecAudioProcessingModel.from_config_dict(self._cfg.mask)
        self.processor = EncMaskDecAudioProcessingModel.from_config_dict(self._cfg.processor)
        self.decoder = EncMaskDecAudioProcessingModel.from_config_dict(self._cfg.decoder)
        self.loss = EncMaskDecAudioProcessingModel.from_config_dict(self._cfg.loss)

        # Setup optional Optimization flags
        self.setup_optimization_flags()

    @torch.no_grad()
    def process(
        self,
        paths2audio_files: List[str],
        output_dir: str,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
        input_channel_selector: Optional[ChannelSelectorType] = None,
    ) -> List[str]:
        """
        # TODO

        Args:
            paths2audio_files: (a list) of paths to audio files. \
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            output_dir: 
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`.

        Returns:
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        # Output
        paths2processed_files = []

        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device

        try:
            # Switch model to evaluation mode
            self.eval()
            # Freeze modules
            self.encoder.freeze()
            self.mask.freeze()
            self.processor.freeze()
            self.decoder.freeze()

            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)

            # Processing
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save temporary manifest
                temporary_manifest_filepath = os.path.join(tmpdir, 'manifest.json')
                with open(temporary_manifest_filepath, 'w', encoding='utf-8') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'input_filepath': audio_file, 'duration': librosa.get_duration(filename=audio_file)}
                        fp.write(json.dumps(entry) + '\n')

                assert (
                    batch_size == 1
                ), f'batch_size is constrained to 1, since input_length is currently not supported by all modules'
                config = {
                    'manifest_filepath': temporary_manifest_filepath,
                    'input_key': 'input_filepath',
                    'input_channel_selector': input_channel_selector,
                    'batch_size': min(batch_size, len(paths2audio_files)),
                    'num_workers': num_workers,
                    'channel_selector': input_channel_selector,
                }

                # Create output dir if necessary
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

                # DataLoader for the input files
                temporary_dataloader = self._setup_process_dataloader(config)

                # Indexing of the original files, used to form output file name
                file_idx = 0

                # Process batches
                for test_batch in tqdm(temporary_dataloader, desc="Processing"):
                    input_signal = test_batch[0]
                    input_length = test_batch[1]

                    # Expand channel dimension, if necessary
                    # For consistency, the model uses multi-channel format, even if the channel dimension is 1
                    if input_signal.ndim == 2:
                        input_signal = input_signal.unsqueeze(2)

                    processed_batch = self.forward(
                        input_signal=input_signal.to(device), input_length=input_length.to(device)
                    )

                    for b in range(processed_batch.shape[0]):
                        # This assumes the data loader is not shuffling files
                        file_name = os.path.basename(paths2audio_files[file_idx])
                        # Prepare output file
                        output_file = os.path.join(output_dir, f'processed_{file_name}')
                        # Crop the output signal to the actual length
                        output_signal = processed_batch[b, : input_length[b], :].cpu().numpy()
                        # Write audio
                        sf.write(output_file, output_signal, self.sample_rate, 'float')
                        # Update the file counter
                        file_idx += 1
                        # Save processed file
                        paths2processed_files.append(output_file)

                    del test_batch
                    del processed_batch

        finally:
            # set mode back to its original value
            self.train(mode=mode)
            if mode is True:
                self.encoder.unfreeze()
                self.mask.unfreeze()
                self.processor.unfreeze()
                self.decoder.unfreeze()
            logging.set_verbosity(logging_level)

        return paths2processed_files

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            raise NotImplementedError('Augmentations not implemented')

        is_concat = config.get('is_concat', False)
        if is_concat:
            raise NotImplementedError('Concat not implemented')

        # TODO: below should be moved from `audio_to_text_dataset` to some utility module?
        # Automatically inject args from model config to dataloader config
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')

        shuffle = config['shuffle']

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            raise NotImplementedError('Tarred datasets not supported')

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        dataset = audio_to_audio_dataset.get_audio_to_target_dataset(config=config)

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = dataset.datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
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
                of a training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_audio.AudioToTargetDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            raise NotImplementedError('Tarred datasets not supported')

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of a validation dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_audio.AudioToTargetDataset`
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
                of a test dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_audio.AudioToTargetDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    def _setup_process_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """Prepare a dataloader for processing files.

        Args:
            config: A python dictionary which contains the following keys:
                manifest_filepath: path to a manifest file
                input_key: key with audio filepaths in the manifest
                input_channel_selector: Optional, used to select a subset of channels from input audio files
                batch_size: batch size for the dataloader
                num_workers: number of workers for the dataloader

        Returns:
            A pytorch DataLoader for the given manifest filepath.
        """
        dl_config = {
            'manifest_filepath': config['manifest_filepath'],
            'sample_rate': self.sample_rate,
            'input_key': config['input_key'],
            'input_channel_selector': config.get('input_channel_selector', None),
            'target_key': None,
            'target_channel_selector': None,
            'batch_size': config['batch_size'],
            'shuffle': False,
            'num_workers': config.get('num_workers', min(config['batch_size'], os.cpu_count() - 1)),
            'pin_memory': True,
        }

        temporary_dataloader = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_dataloader

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        return {
            "input_signal": NeuralType(
                ('B', 'T', 'C'), AudioSignal(freq=self.sample_rate)
            ),  # multi-channel format, channel dimension can be 1 for single-channel audio
            "input_length": NeuralType(
                tuple('B'), LengthsType()
            ),  # multi-channel format, channel dimension can be 1 for single-channel audio
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        return {
            "output_signal": NeuralType(
                ('B', 'T', 'C'), AudioSignal(freq=self.sample_rate)
            ),  # multi-channel format, channel dimension can be 1 for single-channel audio
        }

    def match_batch_length(self, signal: torch.Tensor, batch_length: int):
        """Trim or pad the output to match the batch length.

        Args:
            signal: tensor with shape (B, T, C)
            batch_length: int

        Returns:
            Tensor with shape (B, T, C), where T matches the
            batch length.
        """
        signal_length = signal.shape[1]
        pad_length = batch_length - signal_length
        pad = (0, 0, 0, pad_length)
        # pad with zeros or crop
        return torch.nn.functional.pad(signal, pad, 'constant', 0)

    @typecheck()
    def forward(self, input_signal, input_length):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T] or [B, T, C]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.

        Returns:
        """
        # TODO: use input_length to mask estimation on padded parts
        # This is not critical for training, since we can load fixed-length segments for training.
        # However, it's necessary for some cases, e.g., batched inference.
        batch_length = input_signal.shape[1]

        # Encoder
        encoded = self.encoder(input=input_signal)

        # Mask estimator
        mask = self.mask(input=encoded)

        # Mask-based processor in the encoded domain
        enhanced = self.processor(input=encoded, mask=mask)

        # Decoder
        enhanced = self.decoder(input=enhanced)

        # Trim or pad the estimated signal to match input length
        enhanced = self.match_batch_length(signal=enhanced, batch_length=batch_length)
        return enhanced

    # PTL-specific methods
    def training_step(self, batch, batch_idx):
        input_signal, input_length, target_signal, target_length = batch

        # Expand channel dimension, if necessary
        # For consistency, the model uses multi-channel format, even if the channel dimension is 1
        if input_signal.ndim == 2:
            input_signal = input_signal.unsqueeze(2)
        if target_signal.ndim == 2:
            target_signal = target_signal.unsqueeze(2)

        processed_signal = self.forward(input_signal=input_signal, input_length=input_length)

        loss_value = self.loss(estimate=processed_signal, target=target_signal, input_length=input_length)

        tensorboard_logs = {
            'train_loss': loss_value,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input_signal, input_length, target_signal, target_length = batch

        # Expand channel dimension, if necessary
        # For consistency, the model uses multi-channel format, even if the channel dimension is 1
        if input_signal.ndim == 2:
            input_signal = input_signal.unsqueeze(2)
        if target_signal.ndim == 2:
            target_signal = target_signal.unsqueeze(2)

        processed_signal = self.forward(input_signal=input_signal, input_length=input_length)

        loss_value = self.loss(estimate=processed_signal, target=target_signal, input_length=input_length)

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return {
            'val_loss': loss_value,
        }

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
        results = []

        return results
