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

import json
import os
import subprocess
import tempfile
import types
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.utils import logging

try:
    import requests

    HAVE_REQUESTS = True
except (ImportError, ModuleNotFoundError):
    HAVE_REQUESTS = False


TranscriptionType = Union[List[Any], List[List[Any]], Tuple[Any], Tuple[List[Any]], Dict[str, List[Any]]]


@dataclass
class InternalTranscribeConfig:
    # Internal values
    device: Optional[torch.device] = None
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

    # Utility
    return_generator: bool = False
    partial_hypothesis: Optional[List[Any]] = False

    # DEPRECATED?
    logprobs: bool = False

    _internal: Optional[InternalTranscribeConfig] = None


def move_to_device(batch, device):
    """
    Recursively move all tensors in `batch` to `device`.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return [move_to_device(x, device) for x in batch]
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    else:
        raise TypeError(f"Unsupported type: {type(batch)}")


class TranscriptionMixin(ABC):
    """
    An abstract class for transcribe-able models.
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
        **config_kwargs,
    ) -> List[str]:
        """
        Template function that defines the execution strategy for transcribing audio.

        Args:
            audio: (a single or list) of paths to audio files or a np.ndarray audio array.
                Recommended length per file is between 5 and 25 seconds.
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from
                multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set
                to `None`. Defaults to `None`. Uses zero-based indexing.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
            override_config: (Optional[TranscribeConfig]) override transcription config pre-defined by the user.
                **Note**: All other arguments in the function will be ignored if override_config is passed.
                You should call this argument as `model.transcribe(audio, override_config=TranscribeConfig(...))`.

        Returns:
            Output is defined by the subclass implementation of `TranscriptionMixin._transcribe_output_processing()`.
            It can be:
            - List[str/Hypothesis]
            - List[List[str/Hypothesis]]
            - Tuple[str/Hypothesis]
            - Tuple[List[str/Hypothesis]]
            - Dict[str, List[str/Hypothesis]]
        """

        if override_config is None:
            transcribe_cfg = TranscribeConfig(
                batch_size=batch_size,
                return_hypotheses=return_hypotheses,
                num_workers=num_workers,
                channel_selector=channel_selector,
                augmentor=augmentor,
                verbose=verbose,
                **config_kwargs,
            )
        else:
            if not isinstance(override_config, TranscribeConfig):
                raise ValueError("`override_config` must be of an object of type TranscribeConfig or its subclass")

            transcribe_cfg = override_config

        # Add new internal config
        if transcribe_cfg._internal is None:
            transcribe_cfg._internal = InternalTranscribeConfig()
        else:
            # Check if internal config is valid
            if not isinstance(transcribe_cfg._internal, InternalTranscribeConfig):
                raise ValueError(
                    "`transcribe_cfg._internal` must be of an object of type InternalTranscribeConfig or "
                    "its subclass"
                )

        # Hold the results here
        results = None  # type: TranscriptionType

        try:
            generator = self.transcribe_generator(audio, override_config=transcribe_cfg)

            for processed_outputs in generator:
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
                        if len(processed_outputs) != len(results):
                            raise RuntimeError(
                                f"The number of elements in the result ({len(results)}) does not "
                                f"match the results of the current batch ({len(processed_outputs)})."
                            )

                        for i, processed_output in enumerate(processed_outputs):
                            results[i].append(processed_output)

                else:
                    raise NotImplemented(
                        "Given output result for transcription is not supported. "
                        "Please return a list of results, list of list of results, "
                        "a dict of list of results, or "
                        "a tuple of list of results."
                    )
        except StopIteration:
            pass

        return results

    def transcribe_generator(self, audio, override_config: Optional[TranscribeConfig]):
        """
        A generator version of `transcribe` function.
        """

        transcribe_cfg = override_config
        transcribe_cfg.return_generator = True

        try:
            # Initialize and assert the transcription environment
            self._transcribe_on_begin(audio, transcribe_cfg)

            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                transcribe_cfg._internal.temp_dir = tmpdir

                dataloader = self._transcribe_input_processing(audio, transcribe_cfg)

                for test_batch in tqdm(dataloader, desc="Transcribing", disable=not transcribe_cfg.verbose):
                    # Move batch to device
                    test_batch = move_to_device(test_batch, transcribe_cfg._internal.device)

                    # Run forward pass
                    model_outputs = self._transcribe_forward(test_batch, transcribe_cfg)
                    processed_outputs = self._transcribe_output_processing(model_outputs, transcribe_cfg)

                    # clear up memory
                    del test_batch

                    # Yield results if generator
                    yield processed_outputs

                    del model_outputs, processed_outputs

        finally:
            # set mode back to its original value
            self._transcribe_on_end(transcribe_cfg)

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

        if trcfg._internal.device is None:
            trcfg._internal.device = next(self.parameters()).device

        if trcfg._internal.dtype is None:
            trcfg._internal.dtype = next(self.parameters()).dtype

        if trcfg.num_workers is None:
            trcfg.num_workers = min(trcfg.batch_size, os.cpu_count() - 1)

        # Model's mode and device
        trcfg._internal.training_mode = self.training

        # Switch model to evaluation mode
        if hasattr(self, 'preprocessor'):
            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'dither'):
                trcfg._internal.dither_value = self.preprocessor.featurizer.dither
                self.preprocessor.featurizer.dither = 0.0

            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'pad_to'):
                trcfg._internal.pad_to_value = self.preprocessor.featurizer.pad_to
                self.preprocessor.featurizer.pad_to = 0

        # Switch model to evaluation mode
        self.eval()

        # Disable logging
        trcfg._internal.logging_level = logging.get_verbosity()
        logging.set_verbosity(logging.WARNING)

    def _transcribe_input_processing(self, audio, trcfg: TranscribeConfig):
        if isinstance(audio, (list, tuple, Iterable)):
            audio_files = list(audio)

            tmp_dir = trcfg._internal.temp_dir
            ds_config = self._transcribe_input_manifest_processing(audio_files, tmp_dir, trcfg)

            temp_dataloader = self._setup_transcribe_dataloader(ds_config)
            return temp_dataloader

        elif isinstance(audio, np.ndarray):
            raise NotImplemented()

    @abstractmethod
    def _transcribe_input_manifest_processing(
        self, audio_files: List[str], temp_dir: str, trcfg: TranscribeConfig
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _setup_transcribe_dataloader(self, config: Dict) -> DataLoader:
        pass

    @abstractmethod
    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        pass

    @abstractmethod
    def _transcribe_output_processing(self, outputs, trcfg: TranscribeConfig) -> TranscriptionType:
        pass

    def _transcribe_on_end(self, trcfg: TranscribeConfig):
        # set mode back to its original value
        self.train(mode=trcfg._internal.training_mode)

        if hasattr(self, 'preprocessor'):
            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'dither'):
                self.preprocessor.featurizer.dither = trcfg._internal.dither_value

            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'pad_to'):
                self.preprocessor.featurizer.pad_to = trcfg._internal.pad_to_value

        if trcfg._internal.logging_level is not None:
            logging.set_verbosity(trcfg._internal.logging_level)

    """
    Utility Methods
    """

    def _transcribe_preprocess_array(self, inputs, transcribe_cfg: TranscribeConfig):
        if inputs.ndim > 1 and isinstance(transcribe_cfg.channel_selector, int):
            inputs = inputs[transcribe_cfg.channel_selector, :]

        return inputs


class ASRTranscriptionMixin(TranscriptionMixin):
    def _transcribe_input_manifest_processing(
        self, audio_files: List[str], temp_dir: str, trcfg: TranscribeConfig
    ) -> Dict[str, Any]:
        with open(os.path.join(temp_dir, 'manifest.json'), 'w', encoding='utf-8') as fp:
            for audio_file in audio_files:
                entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                fp.write(json.dumps(entry) + '\n')

        ds_config = {
            'paths2audio_files': audio_files,
            'batch_size': trcfg.batch_size,
            'temp_dir': temp_dir,
            'num_workers': trcfg.num_workers,
            'channel_selector': trcfg.channel_selector,
        }

        if trcfg.augmentor:
            ds_config['augmentor'] = trcfg.augmentor

        return ds_config

    def _transcribe_on_begin(self, audio, trcfg: TranscribeConfig):
        super()._transcribe_on_begin(audio, trcfg)

        # Freeze the encoder and decoder modules
        if hasattr(self, 'encoder'):
            self.encoder.freeze()

        if hasattr(self, 'decoder'):
            self.decoder.freeze()

        if hasattr(self, 'joint'):
            self.joint.freeze()

    def _transcribe_on_end(self, trcfg: TranscribeConfig):
        super()._transcribe_on_end(trcfg)

        # Unfreeze the encoder and decoder modules
        if hasattr(self, 'encoder'):
            self.encoder.unfreeze()

        if hasattr(self, 'decoder'):
            self.decoder.unfreeze()

        if hasattr(self, 'joint'):
            self.joint.unfreeze()
