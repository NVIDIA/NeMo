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

import abc
import math
import random
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.common.parts.preprocessing.collections import AudioCollection
from nemo.collections.common.parts.utils import flatten
from nemo.core.classes import Dataset
from nemo.core.neural_types import AudioSignal, EncodedRepresentation, LengthsType, NeuralType
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = [
    'AudioToTargetDataset',
    'AudioToTargetWithReferenceDataset',
    'AudioToTargetWithEmbeddingDataset',
]


def load_samples_synchronized(
    audio_files: List[str],
    sample_rate: int,
    duration: Optional[float] = None,
    channel_selectors: Optional[List[ChannelSelectorType]] = None,
    fixed_offset: float = 0,
    random_offset: bool = False,
) -> List[np.ndarray]:
    """Load samples from multiple files with the same start and end point.

    Args:
        audio_files: list of paths to audio files
        sample_rate: desired sample rate for output samples
        duration: Optional desired duration of output samples.
                  If `None`, the complete files will be loaded.
                  If set, a segment of `duration` seconds will be loaded from
                  all files. Segment is synchronized across files, so that
                  start and end points are the same.
        channel_selectors: Optional channel selector for each signal, for selecting
                           a subset of channels.
        fixed_offset: Optional fixed offset when loading samples.
        random_offset: If `True`, offset will be randomized when loading a short segment
                       from a file. The value is randomized between fixed_offset and
                       max_offset (set depending on the duration and fixed_offset).

    Returns:
        List with the same size as `audio_files` but containing numpy arrays
        with samples from each audio file.
        Each array has shape (num_samples, ) or (num_samples, num_channels), for single-
        or multi-channel signal, respectively.
        For example, if `audio_files = [path/to/file_1.wav, path/to/file_2.wav]`,
        the output will be a list `output = [samples_1, samples_2]`.
    """
    if channel_selectors is None:
        channel_selectors = [None] * len(audio_files)

    output = []

    if duration is None:
        # Load complete files starting from a fixed offset
        output = []

        for audio_file, channel_selector in zip(audio_files, channel_selectors):
            if isinstance(audio_file, str):
                segment = AudioSegment.from_file(
                    audio_file=audio_file,
                    target_sr=sample_rate,
                    offset=fixed_offset,
                    channel_selector=channel_selector,
                )
                output.append(segment.samples)
            elif isinstance(audio_file, list):
                samples_list = []
                for f in audio_file:
                    segment = AudioSegment.from_file(
                        audio_file=f, target_sr=sample_rate, offset=fixed_offset, channel_selector=channel_selector
                    )
                    samples_list.append(segment.samples)
                output.append(samples_list)
            elif audio_file is None:
                # Support for inference, when the target signal is `None`
                output.append([])
            else:
                raise RuntimeError(f'Unexpected audio_file type {type(audio_file)}')

    else:
        audio_durations = [librosa.get_duration(filename=f) for f in flatten(audio_files)]
        min_duration = min(audio_durations)
        available_duration = min_duration - fixed_offset

        if available_duration <= 0:
            raise ValueError(f'Fixed offset {fixed_offset}s is larger than shortest file {min_duration}s.')
        elif min_duration < duration + fixed_offset:
            logging.warning(
                f'Shortest file ({min_duration}s) is less than desired duration {duration}s + fixed offset {fixed_offset}s. Returned signals will be shortened to {available_duration}s.'
            )
            offset = fixed_offset
            num_samples = math.floor(available_duration * sample_rate)
        elif random_offset:
            # Randomize offset based on the shortest file
            max_offset = min_duration - duration
            offset = random.uniform(fixed_offset, max_offset)
            # Fixed number of samples
            num_samples = math.floor(duration * sample_rate)
        else:
            # Fixed offset
            offset = fixed_offset
            # Fixed number of samples
            num_samples = math.floor(duration * sample_rate)

        # Prepare segments
        for audio_file, channel_selector in zip(audio_files, channel_selectors):
            # Load segments starting from the same offset
            if isinstance(audio_file, str):
                segment = AudioSegment.segment_from_file(
                    audio_file=audio_file,
                    target_sr=sample_rate,
                    n_segments=num_samples,
                    offset=offset,
                    channel_selector=channel_selector,
                )
                output.append(segment.samples)
            elif isinstance(audio_file, list):
                samples_list = []
                for f in audio_file:
                    segment = AudioSegment.segment_from_file(
                        audio_file=f,
                        target_sr=sample_rate,
                        n_segments=num_samples,
                        offset=offset,
                        channel_selector=channel_selector,
                    )
                    samples_list.append(segment.samples)
                output.append(samples_list)
            else:
                raise RuntimeError(f'Unexpected audio_file type {type(audio_file)}')

    return output


def load_samples(
    audio_file: str,
    sample_rate: int,
    duration: Optional[float] = None,
    channel_selector: ChannelSelectorType = None,
    fixed_offset: float = 0,
    random_offset: bool = False,
) -> np.ndarray:
    """Load samples from an audio file.
    For a single-channel signal, the output is shape (num_samples,).
    For a multi-channel signal, the output is shape (num_samples, num_channels).

    Args:
        audio_file: path to an audio file
        sample_rate: desired sample rate for output samples
        duration: Optional desired duration of output samples.
                  If `None`, the complete file will be loaded.
                  If set, a segment of `duration` seconds will be loaded.
        channel_selector: Optional channel selector, for selecting a subset of channels.
        fixed_offset: Optional fixed offset when loading samples.
        random_offset: If `True`, offset will be randomized when loading a short segment
                       from a file. The value is randomized between fixed_offset and
                       max_offset (set depending on the duration and fixed_offset).

    Returns:
        Numpy array with samples from audio file.
        The array has shape (num_samples,) for a single-channel signal
        or (num_samples, num_channels) for a multi-channel signal.
    """
    output = load_samples_synchronized(
        audio_files=[audio_file],
        sample_rate=sample_rate,
        duration=duration,
        channel_selectors=[channel_selector],
        fixed_offset=fixed_offset,
        random_offset=random_offset,
    )

    return output[0]


def load_embedding(filepath: str) -> np.ndarray:
    """Load an embedding vector from a file.

    Args:
        filepath: path to a file storing a vector.
                  Currently, it is assumed the file is a npy file.
    
    Returns:
        Array loaded from filepath.
    """
    if filepath.endswith('.npy'):
        with open(filepath, 'rb') as f:
            embedding = np.load(f)
    else:
        raise RuntimeError(f'Unknown embedding file format in file: {filepath}')

    return embedding


def list_to_multichannel(signal: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    """Convert a list of signals into a multi-channel signal by concatenating
    the elements of the list along the channel dimension.

    If input is not a list, it is returned unmodified.

    Args:
        signal: list of arrays

    Returns:
        Numpy array obrained by concatenating the elements of the list
        along the channel dimension (axis=1).
    """
    if not isinstance(signal, list):
        # Nothing to do there
        return signal
    elif len(signal) == 0:
        # Nothing to do, return as is
        return signal
    elif len(signal) == 1:
        # Nothing to concatenate, return the original format
        return signal[0]

    # If multiple signals are provided in a list, we concatenate them along the channel dimension
    if signal[0].ndim == 1:
        # Single-channel individual files
        mc_signal = np.stack(signal, axis=1)
    elif signal[0].ndim == 2:
        # Multi-channel individual files
        mc_signal = np.concatenate(signal, axis=1)
    else:
        raise RuntimeError(f'Unexpected target with {signal[0].ndim} dimensions.')

    return mc_signal


def _audio_collate_fn(batch: List[dict]) -> Tuple[torch.Tensor]:
    """Collate a batch of items returned by __getitem__.
    Examples for each signal are zero padded to the same length
    (batch_length), which is determined by the longest example.
    Lengths of the original signals are returned in the output.

    Args:
        batch: List of dictionaries. Each element of the list
            has the following format
            ```
            {
                'signal_0': 1D or 2D tensor,
                'signal_1': 1D or 2D tensor,
                ...
                'signal_N': 1D or 2D tensor,
            }
            ```
            1D tensors have shape (num_samples,) and 2D tensors
            have shape (num_samples, num_channels)

    Returns:
        A tuple containing signal tensor and signal length tensor (in samples)
        for each signal.
        The output has the following format:
        ```
        (signal_0, signal_0_length, signal_1, signal_1_length, ..., signal_N, signal_N_length)
        ```
    """
    signals = batch[0].keys()

    batched = tuple()

    for signal in signals:
        signal_length = [b[signal].shape[0] for b in batch]
        # Batch length is determined by the longest signal in the batch
        batch_length = max(signal_length)
        b_signal = []
        for s_len, b in zip(signal_length, batch):
            # check if padding is necessary
            if s_len < batch_length:
                if b[signal].ndim == 1:
                    # single-channel signal
                    pad = (0, batch_length - s_len)
                elif b[signal].ndim == 2:
                    # multi-channel signal
                    pad = (0, 0, 0, batch_length - s_len)
                else:
                    raise RuntimeError(
                        f'Signal {signal} has unsuported dimensions {signal.shape}. Currently, only 1D and 2D arrays are supported.'
                    )
                b[signal] = torch.nn.functional.pad(b[signal], pad)
            # append the current padded signal
            b_signal.append(b[signal])
        # (signal_batched, signal_length)
        batched += (torch.stack(b_signal), torch.tensor(signal_length, dtype=torch.int32))

    # Currently, outputs are expected to be in a tuple, where each element must correspond
    # to the output type in the OrderedDict returned by output_types.
    #
    # Therefore, we return batched signals by interleaving signals and their length:
    #   (signal_0, signal_0_length, signal_1, signal_1_length, ...)
    return batched


@experimental
class BaseAudioDataset(Dataset):
    """Base class of audio datasets, providing common functionality
    for other audio datasets.

    Each line of the manifest file is expected to have the following format
        ```
        {
            audio_key[0]: 'path/to/audio_file_0',
            audio_key[1]: 'path/to/audio_file_1',
            ...
            'duration': duration_of_input,
        }
        ```

    Args:
        manifest_filepath: Path to manifest file in a format described above.
        sample_rate: Sample rate for loaded audio signals.
        audio_to_manifest_key: Dictionary mapping audio signal labels to manifest keys.
        audio_duration: Optional duration of each item returned by __getitem__.
                        If `None`, complete audio will be loaded.
                        If set, a subsegment will be loaded synchronously from
                        target and audio, i.e., with the same start and end point.
        random_offset: If `True`, offset will be randomized when loading a subsegment
                       from a file.
        max_duration: If audio exceeds this length, do not include in dataset.
        min_duration: If audio is less than this length, do not include in dataset.
        max_utts: Limit number of utterances.
    """

    @property
    @abc.abstractmethod
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
        """

    def __init__(
        self,
        manifest_filepath: str,
        sample_rate: int,
        audio_to_manifest_key: Dict[str, Union[str, List[str]]],
        audio_duration: Optional[float] = None,
        random_offset: bool = False,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
    ):
        """Instantiates an audio_dataset.
        """
        super().__init__()

        if type(manifest_filepath) == str:
            manifest_filepath = manifest_filepath.split(',')

        for audio_key, manifest_key in audio_to_manifest_key.items():
            if type(manifest_key) == str and ',' in manifest_key:
                audio_to_manifest_key[audio_key] = manifest_key.split(',')

        self.collection = AudioCollection(
            manifest_files=manifest_filepath,
            audio_to_manifest_key=audio_to_manifest_key,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
        )

        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.random_offset = random_offset

    def num_channels(self, signal_key) -> int:
        """Returns the number of channels for a particular signal in
        items prepared by this dictionary.

        More specifically, this will get the tensor from the first
        item in the dataset, check if it's a one- or two-dimensional
        tensor, and return the number of channels based on the size
        of the second axis (shape[1]).

        NOTE:
        This assumes that all examples returned by `__getitem__`
        have the same number of channels.
        
        Args:
            signal_key: string, used to select a signal from the dictionary
                        output by __getitem__

        Returns:
            Number of channels for the selected signal.
        """
        # Assumption: whole dataset has the same number of channels
        item = self.__getitem__(0)
        if item[signal_key].ndim == 1:
            return 1
        elif item[signal_key].ndim == 2:
            return item[signal_key].shape[1]
        else:
            raise RuntimeError(
                f'Unexpected number of dimension for signal {signal_key} with shape {item[signal_key].shape}'
            )

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return a single example from the dataset.

        Args:
            index: integer index of an example in the collection

        Returns:
            Dictionary providing mapping from signal to its tensor.
            For example:
            ```
            {
                'input': input_tensor,
                'target': target_tensor,
            }
            ```
        """

    def __len__(self) -> int:
        """Return the number of examples in the dataset.
        """
        return len(self.collection)

    def _collate_fn(self, batch) -> Tuple[torch.Tensor]:
        """Collate items in a batch.
        """
        return _audio_collate_fn(batch)


@experimental
class AudioToTargetDataset(BaseAudioDataset):
    """A dataset for audio-to-audio tasks where the goal is to use
    an input signal to recover the corresponding target signal.

    Each line of the manifest file is expected to have the following format
        ```
        {
            'input_key': 'path/to/input.wav',
            'target_key': 'path/to/path_to_target.wav',
            'duration': duration_of_input,
        }
        ```

    Additionally, multiple audio files may be provided for each key in the manifest, for example,
        ```
        {
            'input_key': 'path/to/input.wav',
            'target_key': ['path/to/path_to_target_ch0.wav', 'path/to/path_to_target_ch1.wav'],
            'duration': duration_of_input,
        }
        ```

    Keys for input and target signals can be configured in the constructor (`input_key` and `target_key`).

    Args:
        manifest_filepath: Path to manifest file in a format described above.
        sample_rate: Sample rate for loaded audio signals.
        input_key: Key pointing to input audio files in the manifest
        target_key: Key pointing to target audio files in manifest
        audio_duration: Optional duration of each item returned by __getitem__.
                        If `None`, complete audio will be loaded.
                        If set, a random subsegment will be loaded synchronously from
                        target and audio, i.e., with the same start and end point.
        random_offset: If `True`, offset will be randomized when loading a subsegment
                       from a file.
        max_duration: If audio exceeds this length, do not include in dataset.
        min_duration: If audio is less than this length, do not include in dataset.
        max_utts: Limit number of utterances.
        input_channel_selector: Optional, select subset of channels from each input audio file.
                                If `None`, all channels will be loaded.
        target_channel_selector: Optional, select subset of channels from each input audio file.
                                 If `None`, all channels will be loaded.
    """

    def __init__(
        self,
        manifest_filepath: str,
        sample_rate: int,
        input_key: str,
        target_key: str,
        audio_duration: Optional[float] = None,
        random_offset: bool = False,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: Optional[int] = None,
        input_channel_selector: Optional[int] = None,
        target_channel_selector: Optional[int] = None,
    ):
        self.audio_to_manifest_key = {
            'input': input_key,
            'target': target_key,
        }
        self.input_channel_selector = input_channel_selector
        self.target_channel_selector = target_channel_selector

        super().__init__(
            manifest_filepath=manifest_filepath,
            audio_to_manifest_key=self.audio_to_manifest_key,
            sample_rate=sample_rate,
            audio_duration=audio_duration,
            random_offset=random_offset,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
        )

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.

        Returns:
            Ordered dictionary in the following form:
            ```
            {
                'input_signal': batched single- or multi-channel format,
                'input_length': batched original length of each input signal
                'target_signal': batched single- or multi-channel format,
                'target_length': batched original length of each target signal                
            }
            ```
        """
        sc_audio_type = NeuralType(('B', 'T'), AudioSignal())
        mc_audio_type = NeuralType(('B', 'T', 'C'), AudioSignal())

        return OrderedDict(
            input_signal=sc_audio_type if self.num_channels('input_signal') == 1 else mc_audio_type,
            input_length=NeuralType(('B',), LengthsType()),
            target_signal=sc_audio_type if self.num_channels('target_signal') == 1 else mc_audio_type,
            target_length=NeuralType(('B',), LengthsType()),
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return a single example from the dataset.

        Args:
            index: integer index of an example in the collection

        Returns:
            Dictionary providing mapping from signal to its tensor.
            ```
            {
                'input_signal': input_tensor,
                'target_signal': target_tensor,
            }
            ```
        """
        example = self.collection[index]

        input_file = example.audio_files['input']
        target_file = example.audio_files['target']

        # Load the same segment for different signals
        input_signal, target_signal = load_samples_synchronized(
            audio_files=[input_file, target_file],
            channel_selectors=[self.input_channel_selector, self.target_channel_selector],
            sample_rate=self.sample_rate,
            duration=self.audio_duration,
            fixed_offset=example.offset,
            random_offset=self.random_offset,
        )

        # If necessary, convert a list of arrays into a multi-channel array
        input_signal = list_to_multichannel(input_signal)
        target_signal = list_to_multichannel(target_signal)

        # Output dictionary
        output = OrderedDict(input_signal=torch.tensor(input_signal), target_signal=torch.tensor(target_signal),)

        return output


@experimental
class AudioToTargetWithReferenceDataset(BaseAudioDataset):
    """A dataset for audio-to-audio tasks where the goal is to use
    an input signal to recover the corresponding target signal and an
    additional reference signal is available.

    This can be used, for example, when a reference signal is
    available from
    - enrollment utterance for the target signal
    - echo reference from playback
    - reference from another sensor that correlates with the target signal

    Each line of the manifest file is expected to have the following format
        ```
        {
            'input_key': 'path/to/input.wav',
            'target_key': 'path/to/path_to_target.wav',
            'reference_key': 'path/to/path_to_reference.wav',
            'duration': duration_of_input,
        }
        ```

    Keys for input, target and reference signals can be configured in the constructor.

    Args:
        manifest_filepath: Path to manifest file in a format described above.
        sample_rate: Sample rate for loaded audio signals.
        input_key: Key pointing to input audio files in the manifest
        target_key: Key pointing to target audio files in manifest
        reference_key: Key pointing to reference audio files in manifest
        audio_duration: Optional duration of each item returned by __getitem__.
                        If `None`, complete audio will be loaded.
                        If set, a random subsegment will be loaded synchronously from
                        target and audio, i.e., with the same start and end point.
        random_offset: If `True`, offset will be randomized when loading a subsegment
                       from a file.
        max_duration: If audio exceeds this length, do not include in dataset.
        min_duration: If audio is less than this length, do not include in dataset.
        max_utts: Limit number of utterances.
        input_channel_selector: Optional, select subset of channels from each input audio file.
                                If `None`, all channels will be loaded.
        target_channel_selector: Optional, select subset of channels from each input audio file.
                                 If `None`, all channels will be loaded.
        reference_channel_selector: Optional, select subset of channels from each input audio file.
                                    If `None`, all channels will be loaded.
        reference_is_synchronized: If True, it is assumed that the reference signal is synchronized
                                   with the input signal, so the same subsegment will be loaded as for
                                   input and target. If False, reference signal will be loaded independently
                                   from input and target.
    """

    def __init__(
        self,
        manifest_filepath: str,
        sample_rate: int,
        input_key: str,
        target_key: str,
        reference_key: str,
        audio_duration: Optional[float] = None,
        random_offset: bool = False,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: Optional[int] = None,
        input_channel_selector: Optional[int] = None,
        target_channel_selector: Optional[int] = None,
        reference_channel_selector: Optional[int] = None,
        reference_is_synchronized: bool = True,  # can be disable when reference is an enrollment utterance
    ):
        self.audio_to_manifest_key = {
            'input': input_key,
            'target': target_key,
            'reference': reference_key,
        }
        self.input_channel_selector = input_channel_selector
        self.target_channel_selector = target_channel_selector
        self.reference_channel_selector = reference_channel_selector
        self.reference_is_synchronized = reference_is_synchronized

        super().__init__(
            manifest_filepath=manifest_filepath,
            audio_to_manifest_key=self.audio_to_manifest_key,
            sample_rate=sample_rate,
            audio_duration=audio_duration,
            random_offset=random_offset,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
        )

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.

        Returns:
            Ordered dictionary in the following form:
            ```
            {
                'input_signal': batched single- or multi-channel format,
                'input_length': batched original length of each input signal
                'target_signal': batched single- or multi-channel format,
                'target_length': batched original length of each target signal
                'reference_signal': single- or multi-channel format,
                'reference_length': original length of each reference signal
            }
            ```
        """
        sc_audio_type = NeuralType(('B', 'T'), AudioSignal())
        mc_audio_type = NeuralType(('B', 'T', 'C'), AudioSignal())

        return OrderedDict(
            input_signal=sc_audio_type if self.num_channels('input_signal') == 1 else mc_audio_type,
            input_length=NeuralType(('B',), LengthsType()),
            target_signal=sc_audio_type if self.num_channels('target_signal') == 1 else mc_audio_type,
            target_length=NeuralType(('B',), LengthsType()),
            reference_signal=sc_audio_type if self.num_channels('reference_signal') == 1 else mc_audio_type,
            reference_length=NeuralType(('B',), LengthsType()),
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return a single example from the dataset.

        Args:
            index: integer index of an example in the collection

        Returns:
            Dictionary providing mapping from signal to its tensor.
            ```
            {
                'input_signal': input_tensor,
                'target_signal': target_tensor,
                'reference_signal': reference_tensor,
            }
            ```
        """

        example = self.collection[index]

        input_file = example.audio_files['input']
        target_file = example.audio_files['target']
        reference_file = example.audio_files['reference']

        if self.reference_is_synchronized:
            # Load synchronized segments from input, target and reference
            input_signal, target_signal, reference_signal = load_samples_synchronized(
                audio_files=[input_file, target_file, reference_file],
                channel_selectors=[
                    self.input_channel_selector,
                    self.target_channel_selector,
                    self.reference_channel_selector,
                ],
                sample_rate=self.sample_rate,
                duration=self.audio_duration,
                fixed_offset=example.offset,
                random_offset=self.random_offset,
            )
        else:
            # Load the synchronized segments from input and target
            input_signal, target_signal = load_samples_synchronized(
                audio_files=[input_file, target_file],
                channel_selectors=[self.input_channel_selector, self.target_channel_selector],
                sample_rate=self.sample_rate,
                duration=self.audio_duration,
                fixed_offset=example.offset,
                random_offset=self.random_offset,
            )

            # Reference is not synchronized with input/target, get samples independently
            reference_signal = load_samples(
                audio_file=reference_file,
                sample_rate=self.sample_rate,
                duration=None,  # TODO: add reference_duration to __init__
                channel_selector=self.reference_channel_selector,
            )

        # If necessary, convert a list of arrays into a multi-channel array
        input_signal = list_to_multichannel(input_signal)
        target_signal = list_to_multichannel(target_signal)
        reference_signal = list_to_multichannel(reference_signal)

        # Output dictionary
        output = OrderedDict(
            input_signal=torch.tensor(input_signal),
            target_signal=torch.tensor(target_signal),
            reference_signal=torch.tensor(reference_signal),
        )

        return output


@experimental
class AudioToTargetWithEmbeddingDataset(BaseAudioDataset):
    """A dataset for audio-to-audio tasks where the goal is to use
    an input signal to recover the corresponding target signal and an
    additional embedding signal. It is assumed that the embedding
    is in a form of a vector.

    Each line of the manifest file is expected to have the following format
        ```
        {
            input_key: 'path/to/input.wav',
            target_key: 'path/to/path_to_target.wav',
            embedding_key: 'path/to/path_to_reference.npy',
            'duration': duration_of_input,
        }
        ```

    Keys for input, target and embedding signals can be configured in the constructor.

    Args:
        manifest_filepath: Path to manifest file in a format described above.
        sample_rate: Sample rate for loaded audio signals.
        input_key: Key pointing to input audio files in the manifest
        target_key: Key pointing to target audio files in manifest
        embedding_key: Key pointing to embedding files in manifest
        audio_duration: Optional duration of each item returned by __getitem__.
                        If `None`, complete audio will be loaded.
                        If set, a random subsegment will be loaded synchronously from
                        target and audio, i.e., with the same start and end point.
        random_offset: If `True`, offset will be randomized when loading a subsegment
                       from a file.
        max_duration: If audio exceeds this length, do not include in dataset.
        min_duration: If audio is less than this length, do not include in dataset.
        max_utts: Limit number of utterances.
        input_channel_selector: Optional, select subset of channels from each input audio file.
                                If `None`, all channels will be loaded.
        target_channel_selector: Optional, select subset of channels from each input audio file.
                                 If `None`, all channels will be loaded.
    """

    def __init__(
        self,
        manifest_filepath: str,
        sample_rate: int,
        input_key: str,
        target_key: str,
        embedding_key: str,
        audio_duration: Optional[float] = None,
        random_offset: bool = False,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: Optional[int] = None,
        input_channel_selector: Optional[int] = None,
        target_channel_selector: Optional[int] = None,
    ):
        self.audio_to_manifest_key = {
            'input': input_key,
            'target': target_key,
            'embedding': embedding_key,
        }
        self.input_channel_selector = input_channel_selector
        self.target_channel_selector = target_channel_selector

        super().__init__(
            manifest_filepath=manifest_filepath,
            audio_to_manifest_key=self.audio_to_manifest_key,
            sample_rate=sample_rate,
            audio_duration=audio_duration,
            random_offset=random_offset,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
        )

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.

        Returns:
            Ordered dictionary in the following form:
            ```
            {
                'input_signal': batched single- or multi-channel format,
                'input_length': batched original length of each input signal
                'target_signal': batched single- or multi-channel format,
                'target_length': batched original length of each target signal
                'embedding_vector': batched embedded vector format,
                'embedding_length': batched original length of each embedding vector
            }
            ```
        """
        sc_audio_type = NeuralType(('B', 'T'), AudioSignal())
        mc_audio_type = NeuralType(('B', 'T', 'C'), AudioSignal())

        return OrderedDict(
            input_signal=sc_audio_type if self.num_channels('input_signal') == 1 else mc_audio_type,
            input_length=NeuralType(('B',), LengthsType()),
            target_signal=sc_audio_type if self.num_channels('target_signal') == 1 else mc_audio_type,
            target_length=NeuralType(('B',), LengthsType()),
            embedding_vector=NeuralType(('B', 'D'), EncodedRepresentation()),
            embedding_length=NeuralType(('B',), LengthsType()),
        )

    def __getitem__(self, index):
        """Return a single example from the dataset.

        Args:
            index: integer index of an example in the collection

        Returns:
            Dictionary providing mapping from signal to its tensor.
            ```
            {
                'input_signal': input_tensor,
                'target_signal': target_tensor,
                'embedding_vector': embedding_tensor,
            }
            ```
        """

        example = self.collection[index]

        input_file = example.audio_files['input']
        target_file = example.audio_files['target']
        embedding_file = example.audio_files['embedding']

        # Load synchronized segments from input and target
        input_signal, target_signal = load_samples_synchronized(
            audio_files=[input_file, target_file],
            channel_selectors=[self.input_channel_selector, self.target_channel_selector,],
            sample_rate=self.sample_rate,
            duration=self.audio_duration,
            fixed_offset=example.offset,
            random_offset=self.random_offset,
        )

        # Load embedding
        embedding_vector = load_embedding(embedding_file)

        # If necessary, convert a list of arrays into a multi-channel array
        input_signal = list_to_multichannel(input_signal)
        target_signal = list_to_multichannel(target_signal)

        # Output dictionary
        output = OrderedDict(
            input_signal=torch.tensor(input_signal),
            target_signal=torch.tensor(target_signal),
            embedding_vector=torch.tensor(embedding_vector),
        )

        return output
