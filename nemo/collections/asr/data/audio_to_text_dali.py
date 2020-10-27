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

import math
from collections.abc import Iterator
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig

from nemo.collections.asr.parts import parsers
from nemo.utils.decorators import experimental

try:
    import nvidia.dali as dali
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as DALIPytorchIterator
    from nvidia.dali.plugin.pytorch import LastBatchPolicy as LastBatchPolicy

    HAVE_DALI = True
except (ImportError, ModuleNotFoundError):
    HAVE_DALI = False

__all__ = [
    'AudioToCharDALIDataset',
]


class DALIOutputs(object):
    def __init__(self, out_dict):
        self._has_processed_signal = 'processed_signal' in out_dict and 'processed_signal_len' in out_dict
        if not self._has_processed_signal:
            assert 'audio' in out_dict and 'audio_len' in out_dict
        assert 'transcript' in out_dict and 'transcript_len' in out_dict
        if self._has_processed_signal:
            self._outs = (
                out_dict['processed_signal'],
                out_dict['processed_signal_len'].reshape(-1),
                out_dict['transcript'],
                out_dict['transcript_len'].reshape(-1),
            )
        else:
            self._outs = (
                out_dict['audio'],
                out_dict['audio_len'].reshape(-1),
                out_dict['transcript'],
                out_dict['transcript_len'].reshape(-1),
            )

    @property
    def has_processed_signal(self):
        return self._has_processed_signal

    def __getitem__(self, key):
        return self._outs[key]

    def __len__(self):
        return len(self._outs)


@experimental
class AudioToCharDALIDataset(Iterator):
    """
    NVIDIA DALI pipeline that loads tensors via one or more manifest files where each line containing a sample descriptor in JSON,
    including audio files, transcripts, and durations (in seconds).
    Here's an example:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest file with the format described above. Can be comma-separated paths.
        labels: String containing all the possible characters to map to.
        sample_rate (int): Sample rate to resample loaded audio to.
        batch_size (int): Number of samples in a batch.
        num_threads (int): Number of CPU processing threads to be created by the DALI pipeline.
        max_duration (float): Determines the maximum allowed duration, in seconds, of the loaded audio files.
        min_duration (float): Determines the minimum allowed duration, in seconds, of the loaded audio files.
        blank_index (int): blank character index, default = -1
        unk_index (int): unk_character index, default = -1
        normalize (bool): whether to normalize transcript text (default): True
        bos_id (int): Id of beginning of sequence symbol to append if not None
        eos_id (int): Id of end of sequence symbol to append if not None
        trim (bool): If True, it will extract the nonsilent region of the loaded audio signal.
        shuffle (bool): If set to True, the dataset will shuffled after loading.
        drop_last (bool): If set to True, the last batch will be dropped if incomplete. This will be the case when the shard size is not divisible by the batch size.
                          If set to False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
        device (str): Determines the device type to be used for preprocessing. Allowed values are: 'cpu', 'gpu'.
        device_id (int): Index of the GPU to be used (local_rank). Only applicable when device == 'gpu'. Defaults to 0.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 1.
        preprocessor_cfg (DictConfig): Preprocessor configuration. Supports AudioToMelSpectrogramPreprocessor and AudioToMFCCPreprocessor.
    """

    def __init__(
        self,
        manifest_filepath: str,
        device: str,
        batch_size: int,
        labels: Union[str, List[str]],
        sample_rate: int = 16000,
        num_threads: int = 4,
        max_duration: float = 0.0,
        min_duration: float = 0.0,
        blank_index: int = -1,
        unk_index: int = -1,
        normalize: bool = True,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        trim: bool = False,
        shuffle: bool = True,
        drop_last: bool = False,
        parser: Union[str, Callable] = 'en',
        device_id: int = 0,
        global_rank: int = 0,
        world_size: int = 1,
        preprocessor_cfg: DictConfig = None,
    ):
        if not HAVE_DALI:
            raise ModuleNotFoundError(
                f"{self} requires NVIDIA DALI to be installed. "
                f"See: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html#id1"
            )

        if device not in ('cpu', 'gpu'):
            raise ValueError(
                f"{self} received an unexpected device argument {device}. Supported values are: 'cpu', 'gpu'"
            )

        self.batch_size = batch_size  # Used by NeMo

        self.device = device
        self.device_id = device_id

        if world_size > 1:
            self.shard_id = global_rank
            self.num_shards = world_size
        else:
            self.shard_id = None
            self.num_shards = None

        self.labels = labels
        if self.labels is None or len(self.labels) == 0:
            raise ValueError(f"{self} expects non empty labels list")

        self.parser = parsers.make_parser(
            labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize,
        )

        self.eos_id = eos_id
        self.bos_id = bos_id
        self.sample_rate = sample_rate

        self.pipe = Pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=self.device_id,
            exec_async=True,
            exec_pipelined=True,
        )

        has_preprocessor = preprocessor_cfg is not None
        if has_preprocessor:
            if preprocessor_cfg.cls == "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor":
                feature_type = "mel_spectrogram"
            elif preprocessor_cfg.cls == "nemo.collections.asr.modules.AudioToMFCCPreprocessor":
                feature_type = "mfcc"
            else:
                raise ValueError(
                    f"{self} received an unexpected preprocessor configuration: {preprocessor_cfg.cls}."
                    f" Supported preprocessors are: AudioToMelSpectrogramPreprocessor, AudioToMFCCPreprocessor"
                )

            # Default values taken from AudioToMelSpectrogramPreprocessor
            params = preprocessor_cfg.params
            self.dither = params['dither'] if 'dither' in params else 0.0
            self.preemph = params['preemph'] if 'preemph' in params else 0.97
            self.window_size_sec = params['window_size'] if 'window_size' in params else 0.02
            self.window_stride_sec = params['window_stride'] if 'window_stride' in params else 0.01
            self.sample_rate = params['sample_rate'] if 'sample_rate' in params else sample_rate
            self.window_size = int(self.window_size_sec * self.sample_rate)
            self.window_stride = int(self.window_size_sec * self.sample_rate)

            normalize = params['normalize'] if 'normalize' in params else 'per_feature'
            if normalize == 'per_feature':  # Each freq channel independently
                self.normalization_axes = (1,)
            elif normalize == 'all_features':
                self.normalization_axes = (0, 1)
            else:
                raise ValueError(
                    f"{self} received {normalize} for the normalize parameter."
                    f" It must be either 'per_feature' or 'all_features'."
                )

            self.window = None
            window_name = params['window'] if 'window' in params else None
            torch_windows = {
                'hamming': torch.hamming_window,
                'blackman': torch.blackman_window,
                'bartlett': torch.bartlett_window,
            }
            if window_name is None or window_name == 'hann':
                self.window = None  # Hann is DALI's default
            elif window_name == 'ones':
                self.window = torch.ones(self.window_size)
            else:
                try:
                    window_fn = torch_windows.get(window_name, None)
                    self.window = window_fn(self.window_size, periodic=False)
                except:
                    raise ValueError(
                        f"{self} received {window_name} for the window parameter."
                        f" It must be one of: ('hann', 'ones', 'hamming', 'blackman', 'bartlett', None)."
                        f" None is equivalent to 'hann'."
                    )

            self.n_fft = params['n_fft'] if 'n_fft' in params else None  # None means default
            self.n_mels = params['n_mels'] if 'n_mels' in params else 64
            self.n_mfcc = params['n_mfcc'] if 'n_mfcc' in params else 64

            features = params['features'] if 'features' in params else 0
            if features > 0:
                if feature_type == 'mel_spectrogram':
                    self.n_mels = features
                elif feature_type == 'mfcc':
                    self.n_mfcc = features

            # TODO Implement frame splicing
            if 'frame_splicing' in params:
                assert params['frame_splicing'] == 1, "Frame splicing is not implemented"

            self.freq_low = params['lowfreq'] if 'lowfreq' in params else 0.0
            self.freq_high = params['highfreq'] if 'highfreq' in params else self.sample_rate / 2.0
            self.log_features = params['log'] if 'log' in params else True

            # We want to avoid taking the log of zero
            # There are two options: either adding or clamping to a small value

            self.log_zero_guard_type = params['log_zero_guard_type'] if 'log_zero_guard_type' in params else 'add'
            if self.log_zero_guard_type not in ["add", "clamp"]:
                raise ValueError(
                    f"{self} received {self.log_zero_guard_type} for the "
                    f"log_zero_guard_type parameter. It must be either 'add' or "
                    f"'clamp'."
                )

            self.log_zero_guard_value = params['log_zero_guard_value'] if 'log_zero_guard_value' in params else 1e-05
            if isinstance(self.log_zero_guard_value, str):
                if self.log_zero_guard_value == "tiny":
                    self.log_zero_guard_value = torch.finfo(torch.float32).tiny
                elif self.log_zero_guard_value == "eps":
                    self.log_zero_guard_value = torch.finfo(torch.float32).eps
                else:
                    raise ValueError(
                        f"{self} received {self.log_zero_guard_value} for the log_zero_guard_type parameter."
                        f"It must be either a number, 'tiny', or 'eps'"
                    )

            self.mag_power = params['mag_power'] if 'mag_power' in params else 2
            if self.mag_power != 1.0 and self.mag_power != 2.0:
                raise ValueError(
                    f"{self} received {self.mag_power} for the mag_power parameter." f" It must be either 1.0 or 2.0."
                )

            self.pad_to = params['pad_to'] if 'pad_to' in params else 16
            self.pad_value = params['pad_value'] if 'pad_value' in params else 0.0

        with self.pipe:
            audio, transcript = dali.fn.nemo_asr_reader(
                name="Reader",
                manifest_filepaths=manifest_filepath.split(','),
                dtype=dali.types.FLOAT,
                downmix=True,
                sample_rate=float(self.sample_rate),
                min_duration=min_duration,
                max_duration=max_duration,
                read_sample_rate=False,
                read_text=True,
                random_shuffle=shuffle,
                shard_id=self.shard_id,
                num_shards=self.num_shards,
                pad_last_batch=True,
            )

            transcript_len = dali.fn.shapes(dali.fn.reshape(transcript, shape=[-1]))
            transcript = dali.fn.pad(transcript)

            # Extract nonsilent region, if necessary
            if trim:
                # Need to extract non-silent region before moving to the GPU
                roi_start, roi_len = dali.fn.nonsilent_region(audio, cutoff_db=-60)
                audio = audio.gpu() if self.device == 'gpu' else audio
                audio = dali.fn.slice(
                    audio, roi_start, roi_len, normalized_anchor=False, normalized_shape=False, axes=[0]
                )
            else:
                audio = audio.gpu() if self.device == 'gpu' else audio

            if not has_preprocessor:
                # No preprocessing, the output is the audio signal
                audio = dali.fn.pad(audio)
                audio_len = dali.fn.shapes(dali.fn.reshape(audio, shape=[-1]))
                self.pipe.set_outputs(audio, audio_len, transcript, transcript_len)
            else:
                # Additive gaussian noise (dither)
                if self.dither > 0.0:
                    gaussian_noise = dali.fn.normal_distribution(device=self.device)
                    audio = audio + self.dither * gaussian_noise

                # Preemphasis filter
                if self.preemph > 0.0:
                    audio = dali.fn.preemphasis_filter(audio, preemph_coeff=self.preemph)

                # Power spectrogram
                spec = dali.fn.spectrogram(
                    audio, nfft=self.n_fft, window_length=self.window_size, window_step=self.window_stride
                )

                if feature_type == 'mel_spectrogram' or feature_type == 'mfcc':
                    # Spectrogram to Mel Spectrogram
                    spec = dali.fn.mel_filter_bank(
                        spec,
                        sample_rate=self.sample_rate,
                        nfilter=self.n_mels,
                        normalize=True,
                        freq_low=self.freq_low,
                        freq_high=self.freq_high,
                    )
                    # Mel Spectrogram to MFCC
                    if feature_type == 'mfcc':
                        spec = dali.fn.mfcc(spec, n_mfcc=self.n_mfcc)

                # Logarithm
                if self.log_zero_guard_type == 'add':
                    spec = spec + self.log_zero_guard_value

                spec = dali.fn.to_decibels(
                    spec, multiplier=math.log(10), reference=1.0, cutoff_db=math.log(self.log_zero_guard_value)
                )

                # Normalization
                spec = dali.fn.normalize(spec, axes=self.normalization_axes)

                # Extracting the length of the spectrogram
                shape_start = dali.types.Constant(np.array([1], dtype=np.float32), device='cpu')
                shape_len = dali.types.Constant(np.array([1], dtype=np.float32), device='cpu')
                spec_len = dali.fn.slice(
                    dali.fn.shapes(spec),
                    shape_start,
                    shape_len,
                    normalized_anchor=False,
                    normalized_shape=False,
                    axes=(0,),
                )

                # Pads feature dimension to be a multiple of `pad_to` and the temporal dimension to be as big as the largest sample (shape -1)
                spec = dali.fn.pad(spec, fill_value=self.pad_value, axes=(0, 1), align=(self.pad_to, 1), shape=(1, -1))
            self.pipe.set_outputs(spec, spec_len, transcript, transcript_len)
        # Building DALI pipeline
        self.pipe.build()

        if has_preprocessor:
            output_names = ['processed_signal', 'processed_signal_len', 'transcript_raw', 'transcript_raw_len']
        else:
            output_names = ['audio', 'audio_len', 'transcript_raw', 'transcript_raw_len']

        last_batch_policy = LastBatchPolicy.DROP if drop_last else LastBatchPolicy.PARTIAL
        self._iter = DALIPytorchIterator(
            [self.pipe],
            output_map=output_names,
            reader_name="Reader",
            last_batch_policy=last_batch_policy,
            dynamic_shape=True,
            auto_reset=True,
        )

        # TODO come up with a better solution
        class DummyDataset:
            def __init__(self, parent):
                self.parent = parent

            def __len__(self):
                return self.parent.size

        self.dataset = DummyDataset(self)  # Used by NeMo

    def reset(self):
        self._iter.reset()

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    @property
    def size(self):
        return self._iter.size

    def __len__(self):
        return len(self._iter)

    def __next__(self):
        outputs = self._iter.next()
        assert len(outputs) == 1
        out = outputs[0]
        text_raw_len = out['transcript_raw_len'].numpy()
        text_raw = out['transcript_raw'].numpy()

        text_tokens = []
        text_tokens_len = []
        max_len = 0
        batch_size = text_raw.shape[0]
        for i, text in enumerate(text_raw):
            n = text_raw_len[i][0]
            tbytes = str(text[:n].tobytes(), encoding='utf8')
            ttokens = self.parser(tbytes)
            if self.bos_id is not None:
                ttokens = [self.bos_id] + ttokens
            if self.eos_id is not None:
                ttokens = ttokens + [self.eos_id]
            ttokens_len = len(ttokens)
            text_tokens_len.append(ttokens_len)
            text_tokens.append(ttokens)
            if ttokens_len > max_len:
                max_len = ttokens_len

        transcript_out = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, n in enumerate(text_tokens_len):
            transcript_out[i, :n] = torch.tensor(text_tokens[i], dtype=torch.long)
        transcript_len_out = torch.tensor(text_tokens_len, dtype=torch.long)

        out['transcript'] = transcript_out
        out['transcript_len'] = transcript_len_out
        return DALIOutputs(out)
