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

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text import _AudioTextDataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset
from nemo.collections.asr.data.audio_to_eou_label_lhotse import LhotseSpeechToTextBpeEOUDataset
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding, RNNTBPEDecodingConfig
from nemo.collections.asr.parts.utils.asr_batching import get_semi_sorted_batch_sampler
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils


class EncDecRNNTBPEEOUModel(EncDecRNNTBPEModel):
    """Base class for encoder decoder RNNT-based models with subword tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                # During transcription, the model is initially loaded on the CPU.
                # To ensure the correct global_rank and world_size are set,
                # these values must be passed from the configuration.
                global_rank=self.global_rank if not config.get("do_transcribe", False) else config.get("global_rank"),
                world_size=self.world_size if not config.get("do_transcribe", False) else config.get("world_size"),
                dataset=LhotseSpeechToTextBpeEOUDataset(
                    cfg=config,
                    tokenizer=self.tokenizer,
                    return_cuts=config.get("do_transcribe", False),
                ),
                tokenizer=self.tokenizer,
            )

        dataset = audio_to_text_dataset.get_audio_to_text_bpe_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            tokenizer=self.tokenizer,
            preprocessor_cfg=self.cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        if isinstance(dataset, AudioToBPEDALIDataset):
            # DALI Dataset implements dataloader interface
            return dataset

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

        batch_sampler = None
        if config.get('use_semi_sorted_batching', False):
            if not isinstance(dataset, _AudioTextDataset):
                raise RuntimeError(
                    "Semi Sorted Batch sampler can be used with AudioToCharDataset or AudioToBPEDataset "
                    f"but found dataset of type {type(dataset)}"
                )
            # set batch_size and batch_sampler to None to disable automatic batching
            batch_sampler = get_semi_sorted_batch_sampler(self, dataset, config)
            config['batch_size'] = None
            config['drop_last'] = False
            shuffle = False

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            sampler=batch_sampler,
            batch_sampler=None,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def training_step(self, batch, batch_nb):
        signal, signal_len, _, _, tokens, token_len = batch

        batch = (signal, signal_len, tokens, token_len)

        return super().training_step(batch, batch_nb)