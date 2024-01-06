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

import os
import json
import tempfile
import subprocess
from tqdm import tqdm

import types
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from collections import namedtuple, Iterable


import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from omegaconf import DictConfig, OmegaConf, open_dict
from dataclasses import dataclass, field

import nemo.collections.asr.models as asr_models
from nemo.collections.asr.parts.mixins.asr_adapter_mixins import ASRAdapterModelMixin
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.utils import asr_module_utils
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.common import tokenizers
from nemo.core.classes import typecheck
from nemo.utils import logging

try:
    import requests
    HAVE_REQUESTS = True
except (ImportError, ModuleNotFoundError):
    HAVE_REQUESTS = False


@dataclass
class InternalTranscribeConfig:
    # Internal values
    dtype: Optional[torch.dtype] = None
    training_mode: bool = False
    logging_level: Optional[Any] = None

    # Preprocessor values
    dither_value: float = 0.0
    pad_to_value: int = 0

    # Scratch space
    temp_dir: Optional[str] = None


@dataclass
class TranscribeConfig:
    batch_size: int = 4
    return_hypotheses: bool = False
    num_workers: Optional[int] = None
    channel_selector: ChannelSelectorType = None
    augmentor: Optional[DictConfig] = None
    verbose: bool = True

    return_generator: bool = False

    # DEPRECATED?
    logprobs: bool = False

    _internal: Optional[InternalTranscribeConfig] = None

class Transcribable(ABC):
    """
    An abstract class for transcribable models.
    """

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[str, List[str], np.ndarray],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        override_config: Optional[TranscribeConfig] = None,
    ) -> List[str]:

        if override_config is None:
            transcribe_cfg = TranscribeConfig(
                batch_size=batch_size,
                return_hypotheses=return_hypotheses,
                num_workers=num_workers,
                channel_selector=channel_selector,
                augmentor=augmentor,
                verbose=verbose,
            )
        else:
            transcribe_cfg = override_config

        # Add new internal config
        transcribe_cfg._internal = InternalTranscribeConfig()

        # Hold the results here
        results = None

        try:
            # Initialize and assert the transcription environment
            self._transcribe_on_begin(audio, transcribe_cfg)

            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                transcribe_cfg._internal.temp_dir = tmpdir

                dataloader = self._transcribe_input_processing(audio, transcribe_cfg)

                for test_batch in tqdm(dataloader, desc="Transcribing", disable=not transcribe_cfg.verbose):
                    model_outputs = self._transcribe_forward(test_batch, transcribe_cfg)
                    processed_outputs = self._transcribe_output_processing(model_outputs, transcribe_cfg)

                    # Yield results if generator
                    if transcribe_cfg.return_generator:
                        yield processed_outputs

                    else:
                        # Store results
                        if isinstance(processed_outputs, list):
                            # Create a results of the same type as each element in processed_outputs
                            if results is None:
                                results = []

                                # if list of inner list of results, copy structure
                                if isinstance(processed_outputs[0], list):
                                    for _ in processed_outputs:
                                        results.append([])

                            # If nested list structure
                            if isinstance(processed_outputs[0], list):
                                for i, processed_output in enumerate(processed_outputs):
                                    results[i].extend(processed_output)
                            else:
                                # If flat list structure
                                results.extend(processed_outputs)

                        elif isinstance(processed_outputs, dict):
                            # Create a results of the same type as each element in processed_outputs
                            if results is None:
                                results = processed_outputs
                            else:
                                for k, v in processed_outputs.items():
                                    results[k].extend(v)

                        elif isinstance(processed_outputs, tuple):
                            # Create a results of the same type as each element in processed_outputs
                            if results is None:
                                results = tuple([[] for _ in processed_outputs])

                            # If nested list structure
                            if isinstance(processed_outputs[0], list):
                                for i, processed_output in enumerate(processed_outputs):
                                    results[i].extend(processed_output)
                            else:
                                # If flat list structure
                                results.extend(processed_outputs)

                        else:
                            raise NotImplemented("Given output result for transcription is not supported. "
                                                 "Please return a list of results, list of list of results, "
                                                 "a dict of list of results, or "
                                                 "a tuple of list of results.")


        finally:
            # set mode back to its original value
            self._transcribe_on_end(transcribe_cfg)

        return results

    """
    Transcribe Execution Flow
    """

    def _transcribe_on_begin(self, audio, trcfg: TranscribeConfig):
        if audio is None:
            return {}

        if isinstance(audio, str):
            audio = [audio]

        if isinstance(audio, list) and len(audio) == 0:
            return {}

        if trcfg.num_workers is None:
            trcfg.num_workers = min(trcfg.batch_size, os.cpu_count() - 1)

        if trcfg.dtype is None:
            trcfg.dtype = next(self.parameters()).dtype

        # Model's mode and device
        trcfg._internal.training_mode = self.training
        trcfg._internal.dither_value = self.preprocessor.featurizer.dither
        trcfg._internal.pad_to_value = self.preprocessor.featurizer.pad_to

        # Switch model to evaluation mode
        self.preprocessor.featurizer.dither = 0.0
        self.preprocessor.featurizer.pad_to = 0
        # Switch model to evaluation mode
        self.eval()
        # Freeze the encoder and decoure_exder modules
        self.encoder.freeze()
        self.decoder.freeze()

        # Disable logging
        trcfg._internal.logging_level = logging.get_verbosity()
        logging.set_verbosity(logging.WARNING)

    def _transcribe_input_processing(self, audio, trcfg: TranscribeConfig):
        if isinstance(audio, (list, tuple, Iterable)):
            audio_files = list(audio)

            with open(os.path.join(trcfg._internal.temp_dir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                for audio_file in audio_files:
                    entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                    fp.write(json.dumps(entry) + '\n')

            ds_config = {
                'paths2audio_files': audio_files,
                'batch_size': trcfg.batch_size,
                'temp_dir': trcfg._internal.temp_dir,
                'num_workers': trcfg.num_workers,
                'channel_selector': trcfg.channel_selector,
            }

            if trcfg.augmentor:
                ds_config['augmentor'] = trcfg.augmentor

            temp_dataloader = self._setup_transcribe_dataloader(ds_config)
            return temp_dataloader

        elif isinstance(audio, np.ndarray):
            raise NotImplemented()

    @abstractmethod
    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        pass

    @abstractmethod
    def _transcribe_output_processing(self, outputs, trcfg: TranscribeConfig):
        pass

    def _transcribe_on_end(self, trcfg: TranscribeConfig):
        # set mode back to its original value
        self.train(mode=trcfg._internal.training_mode)
        self.preprocessor.featurizer.dither = trcfg._internal.dither_value
        self.preprocessor.featurizer.pad_to = trcfg._internal.pad_to_value
        if trcfg._internal.training_mode is True:
            self.encoder.unfreeze()
            self.decoder.unfreeze()

        logging.set_verbosity(trcfg._internal.logging_level)


    @abstractmethod
    def _setup_transcribe_dataloader(self, config: Dict) -> DataLoader:
        pass

    """
    Utility Methods
    """
    def _transcribe_preprocess_array(self, inputs, transcribe_cfg: TranscribeConfig):
        if inputs.ndim > 1 and isinstance(transcribe_cfg.channel_selector, int):
            inputs = inputs[transcribe_cfg.channel_selector, :]



        return inputs
