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
#
# Copyright (c) 2018 Ryan Leary
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# This file contains code artifacts adapted from https://github.com/ryanleary/patter
import copy
import io
import os
import random

import librosa
import numpy as np
import webdataset as wd
from omegaconf import DictConfig, OmegaConf
from scipy import signal
from torch.utils.data import IterableDataset
from typing import List, Union

from nemo.collections.asr.parts import collections, parsers
from nemo.collections.asr.parts.segment import AudioSegment
from nemo.utils import logging

try:
    from nemo.collections.asr.parts import numba_utils

    HAVE_NUMBA = True
except (ImportError, ModuleNotFoundError):
    HAVE_NUMBA = False


def read_one_audiosegment(manifest, target_sr, rng, tarred_audio=False, audio_dataset=None):
    if tarred_audio:
        audio_file, file_id = next(audio_dataset)
        manifest_idx = manifest.mapping[file_id]
        manifest_entry = manifest[manifest_idx]

        offset = manifest_entry.offset
        if offset is None:
            offset = 0
    else:
        audio_record = rng.sample(manifest.data, 1)[0]
        audio_file = audio_record.audio_file
        offset = audio_record.offset

    return AudioSegment.from_file(audio_file, target_sr=target_sr, offset=offset)


class Perturbation(object):
    def max_augmentation_length(self, length):
        return length

    def perturb(self, data):
        raise NotImplementedError


class SpeedPerturbation(Perturbation):
    def __init__(self, sr, resample_type, min_speed_rate=0.9, max_speed_rate=1.1, num_rates=5, rng=None):
        """
        Performs Speed Augmentation by re-sampling the data to a different sampling rate,
        which does not preserve pitch.

        Note: This is a very slow operation for online augmentation. If space allows,
        it is preferable to pre-compute and save the files to augment the dataset.

        Args:
            sr: Original sampling rate.
            resample_type: Type of resampling operation that will be performed.
                For better speed using `resampy`'s fast resampling method, use `resample_type='kaiser_fast'`.
                For high-quality resampling, set `resample_type='kaiser_best'`.
                To use `scipy.signal.resample`, set `resample_type='fft'` or `resample_type='scipy'`
            min_speed_rate: Minimum sampling rate modifier.
            max_speed_rate: Maximum sampling rate modifier.
            num_rates: Number of discrete rates to allow. Can be a positive or negative
                integer.
                If a positive integer greater than 0 is provided, the range of
                speed rates will be discretized into `num_rates` values.
                If a negative integer or 0 is provided, the full range of speed rates
                will be sampled uniformly.
                Note: If a positive integer is provided and the resultant discretized
                range of rates contains the value '1.0', then those samples with rate=1.0,
                will not be augmented at all and simply skipped. This is to unnecessary
                augmentation and increase computation time. Effective augmentation chance
                in such a case is = `prob * (num_rates - 1 / num_rates) * 100`% chance
                where `prob` is the global probability of a sample being augmented.
            rng: Random seed number.
        """
        min_rate = min(min_speed_rate, max_speed_rate)
        if min_rate < 0.0:
            raise ValueError("Minimum sampling rate modifier must be > 0.")

        if resample_type not in ('kaiser_best', 'kaiser_fast', 'fft', 'scipy'):
            raise ValueError("Supported `resample_type` values are ('kaiser_best', 'kaiser_fast', 'fft', 'scipy')")

        self._sr = sr
        self._min_rate = min_speed_rate
        self._max_rate = max_speed_rate
        self._num_rates = num_rates
        if num_rates > 0:
            self._rates = np.linspace(self._min_rate, self._max_rate, self._num_rates, endpoint=True)
        self._res_type = resample_type
        self._rng = random.Random() if rng is None else rng

    def max_augmentation_length(self, length):
        return length * self._max_rate

    def perturb(self, data):
        # Select speed rate either from choice or random sample
        if self._num_rates < 0:
            speed_rate = self._rng.uniform(self._min_rate, self._max_rate)
        else:
            speed_rate = self._rng.choice(self._rates)

        # Skip perturbation in case of identity speed rate
        if speed_rate == 1.0:
            return

        new_sr = int(self._sr * speed_rate)
        data._samples = librosa.core.resample(data._samples, self._sr, new_sr, res_type=self._res_type)


class TimeStretchPerturbation(Perturbation):
    def __init__(self, min_speed_rate=0.9, max_speed_rate=1.1, num_rates=5, n_fft=512, rng=None):
        """
        Time-stretch an audio series by a fixed rate while preserving pitch, based on [1, 2].

        Note:
        This is a simplified implementation, intended primarily for reference and pedagogical purposes.
        It makes no attempt to handle transients, and is likely to produce audible artifacts.

        Reference
        [1] [Ellis, D. P. W. “A phase vocoder in Matlab.” Columbia University, 2002.]
        (http://www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/)
        [2] [librosa.effects.time_stretch]
        (https://librosa.github.io/librosa/generated/librosa.effects.time_stretch.html)

        Args:
            min_speed_rate: Minimum sampling rate modifier.
            max_speed_rate: Maximum sampling rate modifier.
            num_rates: Number of discrete rates to allow. Can be a positive or negative
                integer.
                If a positive integer greater than 0 is provided, the range of
                speed rates will be discretized into `num_rates` values.
                If a negative integer or 0 is provided, the full range of speed rates
                will be sampled uniformly.
                Note: If a positive integer is provided and the resultant discretized
                range of rates contains the value '1.0', then those samples with rate=1.0,
                will not be augmented at all and simply skipped. This is to avoid unnecessary
                augmentation and increase computation time. Effective augmentation chance
                in such a case is = `prob * (num_rates - 1 / num_rates) * 100`% chance
                where `prob` is the global probability of a sample being augmented.
            n_fft: Number of fft filters to be computed.
            rng: Random seed number.
        """
        min_rate = min(min_speed_rate, max_speed_rate)
        if min_rate < 0.0:
            raise ValueError("Minimum sampling rate modifier must be > 0.")

        self._min_rate = min_speed_rate
        self._max_rate = max_speed_rate
        self._num_rates = num_rates
        if num_rates > 0:
            self._rates = np.linspace(self._min_rate, self._max_rate, self._num_rates, endpoint=True)
        self._rng = random.Random() if rng is None else rng

        # Pre-compute constants
        self._n_fft = int(n_fft)
        self._hop_length = int(n_fft // 2)

        # Pre-allocate buffers
        self._phi_advance_fast = np.linspace(0, np.pi * self._hop_length, self._hop_length + 1)
        self._scale_buffer_fast = np.empty(self._hop_length + 1, dtype=np.float32)

        self._phi_advance_slow = np.linspace(0, np.pi * self._n_fft, self._n_fft + 1)
        self._scale_buffer_slow = np.empty(self._n_fft + 1, dtype=np.float32)

    def max_augmentation_length(self, length):
        return length * self._max_rate

    def perturb(self, data):
        # Select speed rate either from choice or random sample
        if self._num_rates < 0:
            speed_rate = self._rng.uniform(self._min_rate, self._max_rate)
        else:
            speed_rate = self._rng.choice(self._rates)

        # Skip perturbation in case of identity speed rate
        if speed_rate == 1.0:
            return

        # Increase `n_fft` based on task (speed up or slow down audio)
        # This greatly reduces upper bound of maximum time taken
        # to compute slowed down audio segments.
        if speed_rate >= 1.0:  # Speed up audio
            fft_multiplier = 1
            phi_advance = self._phi_advance_fast
            scale_buffer = self._scale_buffer_fast

        else:  # Slow down audio
            fft_multiplier = 2
            phi_advance = self._phi_advance_slow
            scale_buffer = self._scale_buffer_slow

        n_fft = int(self._n_fft * fft_multiplier)
        hop_length = int(self._hop_length * fft_multiplier)

        # Perform short-term Fourier transform (STFT)
        stft = librosa.core.stft(data._samples, n_fft=n_fft, hop_length=hop_length)

        # Stretch by phase vocoding
        if HAVE_NUMBA:
            stft_stretch = numba_utils.phase_vocoder(stft, speed_rate, phi_advance, scale_buffer)

        else:
            stft_stretch = librosa.core.phase_vocoder(stft, speed_rate, hop_length)

        # Predict the length of y_stretch
        len_stretch = int(round(len(data._samples) / speed_rate))

        # Invert the STFT
        y_stretch = librosa.core.istft(
            stft_stretch, dtype=data._samples.dtype, hop_length=hop_length, length=len_stretch
        )

        data._samples = y_stretch


class GainPerturbation(Perturbation):
    def __init__(self, min_gain_dbfs=-10, max_gain_dbfs=10, rng=None):
        self._min_gain_dbfs = min_gain_dbfs
        self._max_gain_dbfs = max_gain_dbfs
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        gain = self._rng.uniform(self._min_gain_dbfs, self._max_gain_dbfs)
        # logging.debug("gain: %d", gain)
        data._samples = data._samples * (10.0 ** (gain / 20.0))


class ImpulsePerturbation(Perturbation):
    def __init__(self, manifest_path=None, rng=None, audio_tar_filepaths=None, shuffle_n=128, shift_impulse=False):
        self._manifest = collections.ASRAudioText(manifest_path, parser=parsers.make_parser([]), index_by_file_id=True)
        self._audiodataset = None
        self._tarred_audio = False
        self._shift_impulse = shift_impulse

        if audio_tar_filepaths:
            self._tarred_audio = True
            self._audiodataset = AugmentationDataset(manifest_path, audio_tar_filepaths, shuffle_n)
            self._data_iterator = iter(self._audiodataset)

        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        impulse = read_one_audiosegment(
            self._manifest,
            data.sample_rate,
            self._rng,
            tarred_audio=self._tarred_audio,
            audiodata=self._data_iterator,
        )
        if not self._shift_impulse:
            impulse_norm = (impulse.samples - min(impulse.samples)) / (max(impulse.samples) - min(impulse.samples))
            data._samples = signal.fftconvolve(data._samples, impulse_norm, "same")
        else:
            # Find peak and shift peak to left
            impulse_norm = (impulse.samples - min(impulse.samples)) / (max(impulse.samples) - min(impulse.samples))
            max_ind = np.argmax(np.abs(impulse_norm))

            impulse_resp = impulse_norm[max_ind:]
            delay_after = len(impulse_resp)
            data._samples = signal.fftconvolve(data._samples, impulse_resp, "full")[:-delay_after]


class ShiftPerturbation(Perturbation):
    def __init__(self, min_shift_ms=-5.0, max_shift_ms=5.0, rng=None):
        self._min_shift_ms = min_shift_ms
        self._max_shift_ms = max_shift_ms
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        shift_ms = self._rng.uniform(self._min_shift_ms, self._max_shift_ms)
        if abs(shift_ms) / 1000 > data.duration:
            # TODO: do something smarter than just ignore this condition
            return
        shift_samples = int(shift_ms * data.sample_rate // 1000)
        # logging.debug("shift: %s", shift_samples)
        if shift_samples < 0:
            data._samples[-shift_samples:] = data._samples[:shift_samples]
            data._samples[:-shift_samples] = 0
        elif shift_samples > 0:
            data._samples[:-shift_samples] = data._samples[shift_samples:]
            data._samples[-shift_samples:] = 0


class NoisePerturbation(Perturbation):
    def __init__(
        self,
        manifest_path=None,
        min_snr_db=10,
        max_snr_db=50,
        max_gain_db=300.0,
        rng=None,
        audio_tar_filepaths=None,
        shuffle_n=100,
        max_freq=16000,
    ):
        self._manifest = collections.ASRAudioText(manifest_path, parser=parsers.make_parser([]), index_by_file_id=True)
        self._audiodataset = None
        self._tarred_audio = False
        self._max_freq = max_freq
        self._data_iterator = None

        if audio_tar_filepaths:
            self._tarred_audio = True
            self._audiodataset = AugmentationDataset(manifest_path, audio_tar_filepaths, shuffle_n)
            self._data_iterator = iter(self._audiodataset)

        self._rng = random.Random() if rng is None else rng
        self._min_snr_db = min_snr_db
        self._max_snr_db = max_snr_db
        self._max_gain_db = max_gain_db

    @property
    def max_freq(self):
        return self._max_freq

    def get_one_noise_sample(self, target_sr):
        return read_one_audiosegment(
            self._manifest, target_sr, self._rng, tarred_audio=self._tarred_audio, audiodata=self._data_iterator
        )

    def perturb(self, data):
        noise = read_one_audiosegment(
            self._manifest, data.sample_rate, self._rng, tarred_audio=self._tarred_audio, audiodata=self._data_iterator
        )
        self.perturb_with_input_noise(data, noise)

    def perturb_with_input_noise(self, data, noise, data_rms=None):
        snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db)
        if data_rms is None:
            data_rms = data.rms_db
        noise_gain_db = min(data_rms - noise.rms_db - snr_db, self._max_gain_db)
        # logging.debug("noise: %s %s %s", snr_db, noise_gain_db, noise_record.audio_file)

        # calculate noise segment to use
        start_time = self._rng.uniform(0.0, noise.duration - data.duration)
        if noise.duration > (start_time + data.duration):
            noise.subsegment(start_time=start_time, end_time=start_time + data.duration)

        # adjust gain for snr purposes and superimpose
        noise.gain_db(noise_gain_db)

        if noise._samples.shape[0] < data._samples.shape[0]:
            noise_idx = self._rng.randint(0, data._samples.shape[0] - noise._samples.shape[0])
            data._samples[noise_idx : noise_idx + noise._samples.shape[0]] += noise._samples

        else:
            data._samples += noise._samples

    def perturb_with_point_noise(
        self, data, noise, data_rms=None, max_noise_dur=2, max_additions=1,
    ):
        snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db)
        if not data_rms:
            data_rms = data.rms_db

        noise_gain_db = min(data_rms - noise.rms_db - snr_db, self._max_gain_db)
        n_additions = self._rng.randint(1, max_additions)

        for i in range(n_additions):
            noise_dur = self._rng.uniform(0.0, max_noise_dur)
            start_time = self._rng.uniform(0.0, noise.duration)
            start_sample = int(round(start_time * noise.sample_rate))
            end_sample = int(round(min(noise.duration, (start_time + noise_dur)) * noise.sample_rate))
            noise_samples = np.copy(noise._samples[start_sample:end_sample])
            # adjust gain for snr purposes and superimpose
            noise_samples *= 10.0 ** (noise_gain_db / 20.0)

            if noise_samples.shape[0] > data._samples.shape[0]:
                noise_samples = noise_samples[0 : data._samples.shape[0]]

            noise_idx = self._rng.randint(0, data._samples.shape[0] - noise_samples.shape[0])
            data._samples[noise_idx : noise_idx + noise_samples.shape[0]] += noise_samples


class WhiteNoisePerturbation(Perturbation):
    def __init__(self, min_level=-90, max_level=-46, rng=None):
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        self._rng = np.random.RandomState() if rng is None else rng

    def perturb(self, data):
        noise_level_db = self._rng.randint(self.min_level, self.max_level, dtype='int32')
        noise_signal = self._rng.randn(data._samples.shape[0]) * (10.0 ** (noise_level_db / 20.0))
        data._samples += noise_signal


perturbation_types = {
    "speed": SpeedPerturbation,
    "time_stretch": TimeStretchPerturbation,
    "gain": GainPerturbation,
    "impulse": ImpulsePerturbation,
    "shift": ShiftPerturbation,
    "noise": NoisePerturbation,
    "white_noise": WhiteNoisePerturbation,
}


def register_perturbation(name: str, perturbation: Perturbation):
    if name in perturbation_types.keys():
        raise KeyError(
            f"Perturbation with the name {name} exists. " f"Type of perturbation : {perturbation_types[name]}."
        )

    perturbation_types[name] = perturbation


class AudioAugmentor(object):
    def __init__(self, perturbations=None, rng=None):
        self._rng = random.Random() if rng is None else rng
        self._pipeline = perturbations if perturbations is not None else []

    def perturb(self, segment):
        for (prob, p) in self._pipeline:
            if self._rng.random() < prob:
                p.perturb(segment)
        return

    def max_augmentation_length(self, length):
        newlen = length
        for (prob, p) in self._pipeline:
            newlen = p.max_augmentation_length(newlen)
        return newlen

    @classmethod
    def from_config(cls, config):
        ptbs = []
        for p in config:
            if p['aug_type'] not in perturbation_types:
                logging.warning("%s perturbation not known. Skipping.", p['aug_type'])
                continue
            perturbation = perturbation_types[p['aug_type']]
            ptbs.append((p['prob'], perturbation(**p['cfg'])))
        return cls(perturbations=ptbs)


def process_augmentations(augmenter) -> AudioAugmentor:
    """Process list of online data augmentations.
    Accepts either an AudioAugmentor object with pre-defined augmentations,
    or a dictionary that points to augmentations that have been defined.
    If a dictionary is passed, must follow the below structure:
    Dict[str, Dict[str, Any]]: Which refers to a dictionary of string
    names for augmentations, defined in `asr/parts/perturb.py`.
    The inner dictionary may contain key-value arguments of the specific
    augmentation, along with an essential key `prob`. `prob` declares the
    probability of the augmentation being applied, and must be a float
    value in the range [0, 1].
    # Example in YAML config file
    Augmentations are generally applied only during training, so we can add
    these augmentations to our yaml config file, and modify the behaviour
    for training and evaluation.
    ```yaml
    AudioToSpeechLabelDataLayer:
        ...  # Parameters shared between train and evaluation time
        train:
            augmentor:
                shift:
                    prob: 0.5
                    min_shift_ms: -5.0
                    max_shift_ms: 5.0
                white_noise:
                    prob: 1.0
                    min_level: -90
                    max_level: -46
                ...
        eval:
            ...
    ```
    Then in the training script,
    ```python
    import copy
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    with open(model_config) as f:
        params = yaml.load(f)
    # Train Config for Data Loader
    train_dl_params = copy.deepcopy(params["AudioToTextDataLayer"])
    train_dl_params.update(params["AudioToTextDataLayer"]["train"])
    del train_dl_params["train"]
    del train_dl_params["eval"]
    data_layer_train = nemo_asr.AudioToTextDataLayer(
        ...,
        **train_dl_params,
    )
    # Evaluation Config for Data Loader
    eval_dl_params = copy.deepcopy(params["AudioToTextDataLayer"])
    eval_dl_params.update(params["AudioToTextDataLayer"]["eval"])
    del eval_dl_params["train"]
    del eval_dl_params["eval"]
    data_layer_eval = nemo_asr.AudioToTextDataLayer(
        ...,
        **eval_dl_params,
    )
    ```
    # Registering your own Augmentations
    To register custom augmentations to obtain the above convenience of
    the declaring the augmentations in YAML, you can put additional keys in
    `perturbation_types` dictionary as follows.
    ```python
    from nemo.collections.asr.parts import perturb
    # Define your own perturbation here
    class CustomPerturbation(perturb.Perturbation):
        ...
    perturb.register_perturbation(name_of_perturbation, CustomPerturbation)
    ```
    Args:
        augmenter: AudioAugmentor object or
            dictionary of str -> kwargs (dict) which is parsed and used
            to initialize an AudioAugmentor.
            Note: It is crucial that each individual augmentation has
            a keyword `prob`, that defines a float probability in the
            the range [0, 1] of this augmentation being applied.
            If this keyword is not present, then the augmentation is
            disabled and a warning is logged.
    Returns: AudioAugmentor object
    """
    if isinstance(augmenter, AudioAugmentor):
        return augmenter

    if not type(augmenter) in {dict, DictConfig}:
        raise ValueError("Cannot parse augmenter. Must be a dict or an AudioAugmentor object ")

    if isinstance(augmenter, DictConfig):
        augmenter = OmegaConf.to_container(augmenter, resolve=True)

    augmenter = copy.deepcopy(augmenter)

    augmentations = []
    for augment_name, augment_kwargs in augmenter.items():
        prob = augment_kwargs.get('prob', None)

        if prob is None:
            raise KeyError(
                f'Augmentation "{augment_name}" will not be applied as '
                f'keyword argument "prob" was not defined for this augmentation.'
            )

        else:
            _ = augment_kwargs.pop('prob')

            if prob < 0.0 or prob > 1.0:
                raise ValueError("`prob` must be a float value between 0 and 1.")

            try:
                augmentation = perturbation_types[augment_name](**augment_kwargs)
                augmentations.append([prob, augmentation])
            except KeyError:
                raise KeyError(f"Invalid perturbation name. Allowed values : {perturbation_types.keys()}")

    augmenter = AudioAugmentor(perturbations=augmentations)
    return augmenter


class AugmentationDataset(IterableDataset):
    """
        A class that loads tarred audio files and cycles over the files in the dataset.

        Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToCharDataset/AudioToBPEDataset),
        as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
        contain the information for one audio file, including at least the transcript and name of the audio
        file within the tarball.

        Valid formats for the audio_tar_filepaths argument include:
        (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
        (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

        See the WebDataset documentation for more information about accepted data and input formats.
    """

    def __init__(self, manifest_path: str, tar_filepaths: Union[str, List[str]], shuffle_n: int = 128):
        self._manifest = collections.ASRAudioText(manifest_path, parser=parsers.make_parser([]), index_by_file_id=True)
        self.audio_dataset = (
            wd.Dataset(tar_filepaths).shuffle(shuffle_n).rename(audio='wav', key='__key__').to_tuple('audio', 'key')
        )
        self.audio_iter = iter(self.audio_dataset)

    def __len__(self):
        return len(self._manifest)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                audio_bytes, audio_filename = next(self.audio_iter)

            except StopIteration:
                self.audio_iter = iter(self.audio_dataset)
                audio_bytes, audio_filename = next(self.audio_iter)
            file_id, _ = os.path.splitext(os.path.basename(audio_filename))

            # Convert audio bytes to IO stream for processing (for SoundFile to read)
            audio_file = io.BytesIO(audio_bytes)
            return audio_file, file_id
