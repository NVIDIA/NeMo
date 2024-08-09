# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from math import ceil
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.asr.data import audio_to_text_dataset, ssl_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.data.dataclasses import AudioNoiseBatch
from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel
from nemo.collections.asr.modules.ssl_modules.masking import ConvFeatureMaksingWrapper
from nemo.collections.asr.parts.mixins import ASRModuleMixin
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.utils import move_data_to_device
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType
from nemo.utils import logging


class EncDecSpeechSSLModel(SpeechEncDecSelfSupervisedModel):
    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []
        return results

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)

        if self.cfg.get("mask_position", "pre_conv") == "post_conv":
            # adjust config for post-convolution masking
            self.cfg.quantizer.feat_in = self.cfg.encoder.d_model
            self.cfg.masking.feat_in = self.cfg.encoder.d_model
            self.cfg.masking.block_size = self.cfg.masking.block_size // self.cfg.encoder.subsampling_factor
            self.cfg.loss.combine_time_steps = 1

        self.quantizer = self.from_config_dict(self.cfg.quantizer)
        self.mask_processor = self.from_config_dict(self.cfg.masking)
        self.encoder = self.from_config_dict(self.cfg.encoder)
        self.decoder = self.from_config_dict(self.cfg.decoder)
        self.loss = self.from_config_dict(self.cfg.loss)

        self.pre_encoder = None
        if self.cfg.get("mask_position", "pre_conv") == "post_conv":
            # hacked to mask features after convolutional sub-sampling
            self.pre_encoder = ConvFeatureMaksingWrapper(self.encoder.pre_encode, self.mask_processor)
            self.encoder.pre_encode = self.pre_encoder

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "targets": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "target_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "apply_mask": NeuralType(optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        if self.cfg.num_books == 1 and self.cfg.squeeze_single:
            logprobs = NeuralType(('B', 'T', 'C'), LogprobsType())
            tokens = NeuralType(('B', 'T'), LabelsType())
        else:
            logprobs = NeuralType(('B', 'T', 'C', 'H'), LogprobsType())
            tokens = NeuralType(('B', 'T', 'H'), LabelsType())
        return {
            "logprobs": logprobs,
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
            "masks": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "tokens": tokens,
        }

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
        When using pipeline parallelism, we need the global batch to remain on the CPU,
        since the memory overhead will be too high when using a large number of microbatches.
        Microbatches are transferred from CPU to GPU inside the pipeline.
        """
        batch = move_data_to_device(batch, device)
        return batch

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        apply_mask=False,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        if self.pre_encoder is not None:
            # mask after convolutional sub-sampling
            self.pre_encoder.set_masking_enabled(apply_mask=apply_mask)
            encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
            masks = self.pre_encoder.get_current_mask()
            feats = self.pre_encoder.get_current_feat()
            _, tokens = self.quantizer(input_signal=feats.transpose(1, 2))
        else:
            _, tokens = self.quantizer(input_signal=processed_signal)
            if apply_mask:
                masked_signal, masks = self.mask_processor(
                    input_feats=processed_signal, input_lengths=processed_signal_length
                )
            else:
                masked_signal = processed_signal
                masks = torch.zeros_like(processed_signal)
            encoded, encoded_len = self.encoder(audio_signal=masked_signal, length=processed_signal_length)

        log_probs = self.decoder(encoder_output=encoded)

        return log_probs, encoded_len, masks, tokens

    def training_step(self, batch, batch_idx):
        input_signal, input_signal_length, _, _ = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, masks, tokens = self.forward(
                processed_signal=input_signal, processed_signal_length=input_signal_length, apply_mask=True
            )
        else:
            log_probs, encoded_len, masks, tokens = self.forward(
                input_signal=input_signal, input_signal_length=input_signal_length, apply_mask=True
            )

        loss_value = self.loss(masks=masks, decoder_outputs=log_probs, targets=tokens, decoder_lengths=encoded_len)

        tensorboard_logs = {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': self.trainer.global_step,
            'train_loss': loss_value,
        }

        return {'loss': loss_value, 'log': tensorboard_logs}

    def inference_pass(self, batch, batch_idx, dataloader_idx=0, mode='val'):
        input_signal, input_signal_length, _, _ = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, masks, tokens = self.forward(
                processed_signal=input_signal, processed_signal_length=input_signal_length, apply_mask=True
            )
        else:
            log_probs, encoded_len, masks, tokens = self.forward(
                input_signal=input_signal, input_signal_length=input_signal_length, apply_mask=True
            )

        loss_value = self.loss(masks=masks, decoder_outputs=log_probs, targets=tokens, decoder_lengths=encoded_len)

        return {f'{mode}_loss': loss_value}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.inference_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.inference_pass(batch, batch_idx, dataloader_idx, mode="test")
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def multi_validation_epoch_end(self, outputs: list, dataloader_idx: int = 0):
        # loss_list = [x['val_loss'] for x in outputs]
        loss_list = []
        for i, x in enumerate(outputs):
            if not isinstance(x, dict):
                logging.warning(f'Batch {i} output in validation dataloader {dataloader_idx} is not a dictionary: {x}')
            if 'val_loss' in x:
                loss_list.append(x['val_loss'])
            else:
                logging.warning(
                    f'Batch {i} output in validation dataloader {dataloader_idx} does not have key `val_loss`: {x}'
                )

        if len(loss_list) == 0:
            logging.warning(
                f'Epoch {self.current_epoch} received no batches for validation dataloader {dataloader_idx}.'
            )
            return {}

        val_loss_mean = torch.stack(loss_list).mean()
        tensorboard_logs = {'val_loss': val_loss_mean}
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': test_loss_mean}
        return {'test_loss': test_loss_mean, 'log': tensorboard_logs}


class EncDecSpeechDenoiseMLMModel(EncDecSpeechSSLModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')

        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=ssl_dataset.LhotseAudioNoiseDataset(
                    noise_manifest=config.get('noise_manifest', None),
                    batch_augmentor_cfg=config.get('batch_augmentor', None),
                ),
            )

        dataset = ssl_dataset.get_audio_noise_dataset_from_config(
            config,
            global_rank=self.global_rank,
            world_size=self.world_size,
        )

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "noise_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "noise_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_noise_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_noise_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "noisy_input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "noisy_input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_noisy_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_noisy_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "apply_mask": NeuralType(optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        if self.cfg.num_books == 1 and self.cfg.squeeze_single:
            logprobs = NeuralType(('B', 'T', 'C'), LogprobsType())
            tokens = NeuralType(('B', 'T'), LabelsType())
        else:
            logprobs = NeuralType(('B', 'T', 'C', 'H'), LogprobsType())
            tokens = NeuralType(('B', 'T', 'H'), LabelsType())
        return {
            "logprobs": logprobs,
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
            "masks": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "tokens": tokens,
        }

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """
        PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
        """
        batch = move_data_to_device(batch, device)
        return batch

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        noise_signal=None,
        noise_signal_length=None,
        processed_noise_signal=None,
        processed_noise_signal_length=None,
        noisy_input_signal=None,
        noisy_input_signal_length=None,
        processed_noisy_input_signal=None,
        processed_noisy_input_signal_length=None,
        apply_mask=False,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )
        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        has_noise_signal = noise_signal is not None and noise_signal_length is not None
        has_processed_noise_signal = processed_noise_signal is not None and processed_noise_signal_length is not None
        if (has_noise_signal ^ has_processed_noise_signal) == False:
            raise ValueError(
                f"{self} Arguments ``noise_signal`` and ``noise_signal_length`` are mutually exclusive "
                " with ``processed_noise_signal`` and ``processed_noise_signal_len`` arguments."
            )
        if not has_processed_noise_signal:
            processed_noise_signal, processed_noise_signal_length = self.preprocessor(
                input_signal=noise_signal,
                length=noise_signal_length,
            )

        has_noisy_input_signal = noisy_input_signal is not None and noisy_input_signal_length is not None
        has_processed_noisy_input_signal = (
            processed_noisy_input_signal is not None and processed_noisy_input_signal_length is not None
        )
        if (has_noisy_input_signal ^ has_processed_noisy_input_signal) == False:
            raise ValueError(
                f"{self} Arguments ``noisy_input_signal`` and ``noisy_input_signal_length`` are mutually exclusive "
                " with ``processed_noisy_input_signal`` and ``processed_noisy_input_signal_len`` arguments."
            )
        if not has_processed_noisy_input_signal:
            processed_noisy_input_signal, processed_noisy_input_signal_length = self.preprocessor(
                input_signal=noisy_input_signal,
                length=noisy_input_signal_length,
            )

        if self.pre_encoder is not None:
            # mask after convolutional sub-sampling
            feats, _ = self.pre_encoder.pre_encode(x=processed_signal, lengths=processed_signal_length)
            _, tokens = self.quantizer(input_signal=feats.transpose(1, 2))

            self.pre_encoder.set_masking_enabled(apply_mask=apply_mask)
            encoded, encoded_len = self.encoder(
                audio_signal=processed_noisy_input_signal, length=processed_noisy_input_signal_length
            )
            masks = self.pre_encoder.get_current_mask()
        else:
            _, tokens = self.quantizer(input_signal=processed_signal)
            if apply_mask:
                masked_signal, masks = self.mask_processor(
                    input_feats=processed_noisy_input_signal, input_lengths=processed_noisy_input_signal_length
                )
            else:
                masked_signal = processed_noisy_input_signal
                masks = torch.zeros_like(processed_noisy_input_signal)
            encoded, encoded_len = self.encoder(audio_signal=masked_signal, length=processed_noisy_input_signal_length)

        log_probs = self.decoder(encoder_output=encoded)

        return log_probs, encoded_len, masks, tokens

    def training_step(self, batch: AudioNoiseBatch, batch_idx: int):
        log_probs, encoded_len, masks, tokens = self.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_len,
            noise_signal=batch.noise,
            noise_signal_length=batch.noise_len,
            noisy_input_signal=batch.noisy_audio,
            noisy_input_signal_length=batch.noisy_audio_len,
            apply_mask=True,
        )

        loss_value = self.loss(masks=masks, decoder_outputs=log_probs, targets=tokens, decoder_lengths=encoded_len)

        if loss_value.isnan():
            logging.error(f"NaN detected in loss at step {self.global_step}.")
        if log_probs.isnan().any():
            logging.error(f"NaN detected in log_probs at step {self.global_step}.")
        if masks.isnan().any():
            logging.error(f"NaN detected in masks at step {self.global_step}.")

        tensorboard_logs = {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': self.trainer.global_step,
            'train_loss': loss_value,
        }

        return {'loss': loss_value, 'log': tensorboard_logs}

    def inference_pass(self, batch: AudioNoiseBatch, batch_idx: int, dataloader_idx: int = 0, mode: str = 'val'):
        log_probs, encoded_len, masks, tokens = self.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_len,
            noise_signal=batch.noise,
            noise_signal_length=batch.noise_len,
            noisy_input_signal=batch.noisy_audio,
            noisy_input_signal_length=batch.noisy_audio_len,
            apply_mask=True,
        )

        loss_value = self.loss(masks=masks, decoder_outputs=log_probs, targets=tokens, decoder_lengths=encoded_len)
        if loss_value.isnan():
            logging.error(
                f"NaN detected in val loss at dataloader {self._validation_names[dataloader_idx]} batch {batch_idx}."
            )
        if log_probs.isnan().any():
            logging.error(
                f"NaN detected in val log_probs at dataloader {self._validation_names[dataloader_idx]} batch {batch_idx}."
            )
        if masks.isnan().any():
            logging.error(
                f"NaN detected in val masks at dataloader {self._validation_names[dataloader_idx]} batch {batch_idx}."
            )
        if tokens.isnan().any():
            logging.error(
                f"NaN detected in val tokens at dataloader {self._validation_names[dataloader_idx]} batch {batch_idx}."
            )
        return {f'{mode}_loss': loss_value}


class SemiSupervisedSpeechMAEModel(ModelPT, ASRModuleMixin, AccessMixin):
    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []
        return results

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)

        self.preprocessor = self.from_config_dict(self.cfg.preprocessor)
        self.mask_processor = self.from_config_dict(self.cfg.masking)
        self.encoder = self.from_config_dict(self.cfg.encoder)
        self.decoder = self.from_config_dict(self.cfg.decoder)
        self.loss = self.from_config_dict(self.cfg.loss)

        self.pre_encoder = None
        if self.cfg.get("mask_position", "pre_conv") == "post_conv":
            # hacked to mask features after convolutional sub-sampling
            self.pre_encoder = ConvFeatureMaksingWrapper(self.encoder.pre_encode, self.mask_processor)
            self.encoder.pre_encode = self.pre_encoder

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """
        PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
        """
        batch = move_data_to_device(batch, device)
        return batch

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')

        dataset = ssl_dataset.get_audio_noise_dataset_from_config(
            config,
            global_rank=self.global_rank,
            world_size=self.world_size,
        )

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        apply_mask=False,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )
        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        if apply_mask:
            masked_signal, masks = self.mask_processor(
                input_feats=processed_signal, input_lengths=processed_signal_length
            )
        else:
            masked_signal = processed_signal
            masks = torch.zeros_like(processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=masked_signal, length=processed_signal_length)

    def forward_speaker_and_content(
        self,
        processed_signal=None,
        processed_signal_length=None,
    ):
        content_feats = None
        speaker_feats = None
        feat_length = None
        return content_feats, speaker_feats, feat_length
