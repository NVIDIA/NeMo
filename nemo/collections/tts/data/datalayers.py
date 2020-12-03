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

# MIT License

# Copyright (c) 2019 Jeongmin Liu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# TODO: Need to fix line endings on this file

import os
import shutil
import sys
import collections as py_collections
import json
import pickle
from pathlib import Path
from os.path import expanduser
from typing import Any, Dict, Optional, Union, Callable

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from nemo.collections.asr.parts import collections, parsers
from nemo.collections.asr.parts.segment import AudioSegment
from nemo.collections.asr.parts.features import WaveformFeaturizer
from nemo.core.classes import Dataset
from nemo.core.neural_types.elements import *
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging

DataDict = Dict[str, Any]


class AudioDataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            "audio_signal": NeuralType(("B", "T"), AudioSignal()),
            "a_sig_length": NeuralType(tuple("B"), LengthsType()),
        }

    def __init__(
        self,
        manifest_filepath: Union[str, "pathlib.Path"],
        n_segments: int,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        trim: Optional[bool] = False,
        truncate_to: Optional[int] = 1,
    ):
        """
        Mostly compliant with nemo.collections.asr.data.datalayers.AudioToTextDataset except it only returns Audio
        without text. Dataset that loads tensors via a json file containing paths to audio files, transcripts, and
        durations (in seconds). Each new line is a different sample. Note that text is required, but is ignored for
        AudioDataset. Example below:
        {"audio_filepath": "/path/to/audio.wav", "text_filepath":
        "/path/to/audio.txt", "duration": 23.147}
        ...
        {"audio_filepath": "/path/to/audio.wav", "text": "the
        transcription", "offset": 301.75, "duration": 0.82, "utt":
        "utterance_id", "ctm_utt": "en_4156", "side": "A"}
        Args:
            manifest_filepath (str, Path): Path to manifest json as described above. Can be comma-separated paths
                such as "train_1.json,train_2.json" which is treated as two separate json files.
            n_segments (int): The length of audio in samples to load. For example, given a sample rate of 16kHz, and
                n_segments=16000, a random 1 second section of audio from the clip will be loaded. The section will
                be randomly sampled everytime the audio is batched. Can be set to -1 to load the entire audio.
            max_duration (float): If audio exceeds this length in seconds, it is filtered from the dataset.
                Defaults to None, which does not filter any audio.
            min_duration(float): If audio is less than this length in seconds, it is filtered from the dataset.
                Defaults to None, which does not filter any audio.
            trim (bool): Whether to use librosa.effects.trim on the audio clip
            truncate_to (int): Ensures that the audio segment returned is a multiple of truncate_to.
                Defaults to 1, which does no truncating.
        """

        self.collection = collections.ASRAudioText(
            manifests_files=manifest_filepath.split(","),
            parser=parsers.make_parser(),
            min_duration=min_duration,
            max_duration=max_duration,
        )
        self.trim = trim
        self.n_segments = n_segments
        self.truncate_to = truncate_to

    def _collate_fn(self, batch):
        """
        Takes a batch: a lists of length batch_size, defined in the dataloader. Returns 2 padded and batched
        tensors corresponding to the audio and audio_length.
        """

        def find_max_len(seq, index):
            max_len = -1
            for item in seq:
                if item[index].size(0) > max_len:
                    max_len = item[index].size(0)
            return max_len

        batch_size = len(batch)

        audio_signal, audio_lengths = None, None
        if batch[0][0] is not None:
            if self.n_segments > 0:
                max_audio_len = self.n_segments
            else:
                max_audio_len = find_max_len(batch, 0)

            audio_signal = torch.zeros(batch_size, max_audio_len, dtype=torch.float)
            audio_lengths = []
            for i, sample in enumerate(batch):
                audio_signal[i].narrow(0, 0, sample[0].size(0)).copy_(sample[0])
                audio_lengths.append(sample[1])
            audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

        return audio_signal, audio_lengths

    def __getitem__(self, index):
        """
        Given a index, returns audio and audio_length of the corresponding element. Audio clips of n_segments are
        randomly chosen if the audio is longer than n_segments.
        """
        example = self.collection[index]
        features = AudioSegment.segment_from_file(example.audio_file, n_segments=self.n_segments, trim=self.trim,)
        features = torch.tensor(features.samples)
        audio, audio_length = features, torch.tensor(features.shape[0]).long()

        truncate = audio_length % self.truncate_to
        if truncate != 0:
            audio_length -= truncate.long()
            audio = audio[:audio_length]

        return audio, audio_length

    def __len__(self):
        return len(self.collection)


class SplicedAudioDataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        manifest_filepath: Union[str, 'pathlib.Path'],
        n_segments: int,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        trim: Optional[bool] = False,
        truncate_to: Optional[int] = 1,
    ):
        """
        See above AudioDataset for details on dataset and manifest formats.

        Unlike the regular AudioDataset, which samples random segments from each audio array as an example,
        SplicedAudioDataset concatenates all audio arrays together and indexes segments as examples. This way,
        the model sees more data (about 9x for LJSpeech) per epoch.

        Note: this class is not recommended to be used in validation.

        Args:
            manifest_filepath (str, Path): Path to manifest json as described above. Can be comma-separated paths
                such as "train_1.json,train_2.json" which is treated as two separate json files.
            n_segments (int): The length of audio in samples to load. For example, given a sample rate of 16kHz, and
                n_segments=16000, a random 1 second section of audio from the clip will be loaded. The section will
                be randomly sampled everytime the audio is batched. Can be set to -1 to load the entire audio.
            max_duration (float): If audio exceeds this length in seconds, it is filtered from the dataset.
                Defaults to None, which does not filter any audio.
            min_duration(float): If audio is less than this length in seconds, it is filtered from the dataset.
                Defaults to None, which does not filter any audio.
            trim (bool): Whether to use librosa.effects.trim on the audio clip
            truncate_to (int): Ensures that the audio segment returned is a multiple of truncate_to.
                Defaults to 1, which does no truncating.
        """
        assert n_segments > 0

        collection = collections.ASRAudioText(
            manifests_files=manifest_filepath.split(','),
            parser=parsers.make_parser(),
            min_duration=min_duration,
            max_duration=max_duration,
        )
        self.trim = trim
        self.n_segments = n_segments
        self.truncate_to = truncate_to

        self.samples = []
        for index in range(len(collection)):
            example = collection[index]
            with sf.SoundFile(example.audio_file, 'r') as f:
                samples = f.read(dtype='float32').transpose()
                self.samples.append(samples)
        self.samples = np.concatenate(self.samples, axis=0)
        self.samples = self.samples[: self.samples.shape[0] - (self.samples.shape[0] % self.n_segments), ...]

    def __getitem__(self, index):
        """
        Given a index, returns audio and audio_length of the corresponding element. Audio clips of n_segments are
        randomly chosen if the audio is longer than n_segments.
        """
        audio_index = index * self.n_segments
        audio = self.samples[audio_index : audio_index + self.n_segments]

        return audio, self.n_segments

    def __len__(self):
        return self.samples.shape[0] // self.n_segments


class NoisySpecsDataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'x': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            'mag': NeuralType(('B', 'any', 'D', 'T'), SpectrogramType()),
            'max_length': NeuralType(None, LengthsType()),
            'y': NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            'T_ys': NeuralType(tuple('B'), LengthsType()),
            'length': NeuralType(tuple('B'), LengthsType()),
            'path_speech': NeuralType(tuple('B'), StringType()),
        }

    def __init__(
        self, destination: Union[str, 'pathlib.Path'], subdir: str, n_fft: int, hop_length: int, num_snr: int,
    ):
        self.tar_dir = Path("%s/degli_data_%d_%dx%d/%s/" % (destination, n_fft, hop_length, num_snr, subdir))
        """
        A modified dataset for training deep-griffin-lim iteration. Contains MSTFT (mag), STFT (y) , and noisy STFT which is
        used for initial phase. By using different levels of noise, the Degli model can learn to improve any phase, and thus
        it can be used iteratively.

        Args:
            destination (str, Path): Path to a directory containing the main data set folder, Similar to the directory
            provided to the preprocessor script, which generates this dataset.
            subdir (str): Either 'train', or 'valid', when using the standard script for generation.
            n_fft (int): STFT parameter. Also detrmines the STFT filter length.
            hop_length (int): STFT parameter.
            num_snr (int): number of noisy samples per clean audio in the original dataset.
        """

        self._all_files = [f for f in os.listdir(self.tar_dir) if 'npz' in f]

    def __getitem__(self, index):

        file = Path(self.tar_dir / self._all_files[index])
        sample = dict()

        with np.load(file, mmap_mode='r') as npz_data:
            for k, v in npz_data.items():
                if k in ['x', 'y', 'y_mag']:
                    sample[k] = torch.from_numpy(v)
                elif k == "path_speech":
                    sample[k] = str(v)
                elif k in ['T_x', 'T_y', 'length']:
                    sample[k] = int(v)
                else:
                    sample[k] = v
        return sample

    def __len__(self):
        return len(self._all_files)

    @torch.no_grad()
    def _collate_fn(self, batch):
        """ return data with zero-padding

        Important data like x, y are all converted to Tensor(cpu).
        :param batch:
        :return: DataDict
            Values can be an Tensor(cpu), list of str, ndarray of int.
        """

        result = dict()
        T_xs = np.array([item.pop('T_x') for item in batch])
        idxs_sorted = np.argsort(T_xs)
        T_xs = T_xs[idxs_sorted].tolist()
        T_ys = [batch[idx].pop('T_y') for idx in idxs_sorted]
        length = [batch[idx].pop('length') for idx in idxs_sorted]

        result['T_xs'], result['T_ys'], result['length'] = T_xs, T_ys, length

        for key, value in batch[0].items():
            if type(value) == str:
                list_data = [batch[idx][key] for idx in idxs_sorted]
                set_data = set(list_data)
                if len(set_data) == 1:
                    result[key] = set_data.pop()
                else:
                    result[key] = list_data
            else:
                if len(batch) > 1:
                    # B, T, F, C
                    data = [batch[idx][key].permute(1, 0, 2) for idx in idxs_sorted]
                    data = pad_sequence(data, batch_first=True)
                    # B, C, F, T
                    data = data.permute(0, 3, 2, 1)
                else:  # B, C, F, T
                    data = batch[0][key].unsqueeze(0).permute(0, 3, 1, 2)

                result[key] = data.contiguous()

        x = result['x']
        mag = result['y_mag']
        max_length = max(result['length'])
        y = result['y']
        T_ys = result['T_ys']
        length = result['length']
        path_speech = result['path_speech']
        return x, mag, max_length, y, T_ys, length, path_speech

    @staticmethod
    @torch.no_grad()
    def decollate_padded(batch: DataDict, idx: int) -> DataDict:
        """ select the `idx`-th data, get rid of padded zeros and return it.

        Important data like x, y are all converted to ndarray.
        :param batch:
        :param idx:
        :return: DataDict
            Values can be an str or ndarray.
        """
        result = dict()
        for key, value in batch.items():
            if type(value) == str:
                result[key] = value
            elif type(value) == list:
                result[key] = value[idx]
            elif not key.startswith('T_'):
                T_xy = 'T_xs' if 'x' in key else 'T_ys'
                value = value[idx, :, :, : batch[T_xy][idx]]  # C, F, T
                value = value.permute(1, 2, 0).contiguous()  # F, T, C
                value = value.numpy()
                if value.shape[-1] == 2:
                    value = value.view(dtype=np.complex64)  # F, T, 1
                result[key] = value

        return result


def setup_noise_augmented_dataset(files_list, num_snr, kwargs_stft, dest, desc):

    os.makedirs(dest)
    with open(files_list, 'r') as list_file:
        all_lines = [line for line in list_file]
        list_file_pbar = tqdm(all_lines, desc=desc, dynamic_ncols=True)

        i_speech = 0
        for line in list_file_pbar:
            audio_file = line.split('|')[0]
            speech = sf.read(audio_file)[0].astype(np.float32)
            spec_clean = np.ascontiguousarray(librosa.stft(speech, **kwargs_stft))
            mag_clean = np.ascontiguousarray(np.abs(spec_clean)[..., np.newaxis])
            signal_power = np.mean(np.abs(speech) ** 2)

            y = spec_clean.view(dtype=np.float32).reshape((*spec_clean.shape, 2))
            ##y = torch.from_numpy(y)
            T_y = spec_clean.shape[1]
            ##mag_clean = torch.from_numpy(mag_clean)
            for k in range(num_snr):
                snr_db = -6 * np.random.rand()
                snr = librosa.db_to_power(snr_db)
                noise_power = signal_power / snr
                noisy = speech + np.sqrt(noise_power) * np.random.randn(len(speech))
                spec_noisy = librosa.stft(noisy, **kwargs_stft)
                spec_noisy = np.ascontiguousarray(spec_noisy)
                T_x = spec_noisy.shape[1]
                x = spec_noisy.view(dtype=np.float32).reshape((*spec_noisy.shape, 2))
                ##x = torch.from_numpy(x)
                mdict = dict(x=x, y=y, y_mag=mag_clean, path_speech=audio_file, length=len(speech), T_x=T_x, T_y=T_y)
                np.savez(
                    f"{dest}/audio_{i_speech}_{k}.npz", **mdict,
                )
                i_speech = i_speech + 1

    return i_speech


def preprocess_linear_specs_dataset(valid_filelist, train_filelist, n_fft, hop_length, num_snr, destination):
    kwargs_stft = dict(hop_length=hop_length, window='hann', center=True, n_fft=n_fft, dtype=np.complex64)

    tar_dir = "%s/degli_data_%d_%dx%d/" % (destination, n_fft, hop_length, num_snr)
    if not os.path.isdir(tar_dir):

        if valid_filelist == "none" or train_filelist == "none":
            logging.error(f"Director {tar_dir} does not exist. Filelists for validation and train must be provided.")
            raise NameError("Missing Argument")
        else:
            logging.info(
                f"Director {tar_dir} does not exist. Preprocessing audio files listed in {valid_filelist}, {train_filelist} to create new dataset."
            )
        os.makedirs(tar_dir)
        n_train = 0
        n_valid = 0
        try:
            n_train = setup_noise_augmented_dataset(
                train_filelist, num_snr, kwargs_stft, tar_dir + "train/", desc="Initializing Train Dataset"
            )
            n_valid = setup_noise_augmented_dataset(
                valid_filelist, num_snr, kwargs_stft, tar_dir + "valid/", desc="Initializing Validation Dataset"
            )
        except FileNotFoundError as err:
            shutil.rmtree(tar_dir)
            raise err
        except:
            e = sys.exc_info()[0]
            shutil.rmtree(tar_dir)
            raise e

        if n_train == 0:
            shutil.rmtree(tar_dir)
            raise EOFError("Dataset initialization failed. No files to preprocess train dataset")

        if n_valid == 0:
            shutil.rmtree(tar_dir)
            raise EOFError("Dataset initialization failed. No files to preprocess validation dataset")

    return tar_dir


class FastSpeechWithDurs(Dataset):
    # @property
    # def output_types(self) -> Optional[Dict[str, NeuralType]]:
    #     """Returns definitions of module output ports."""
    #     return {
    #         'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
    #         'a_sig_length': NeuralType(('B'), LengthsType()),
    #         'transcripts': NeuralType(('B', 'T'), LabelsType()),
    #         'transcript_length': NeuralType(('B'), LengthsType()),
    #     }

    def __init__(
        self,
        manifest_filepath: str,
        sample_rate: int,
        supplementary_dir: str,
        # int_values: bool = False,
        # augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        ignore_file: Optional[str] = None,
        max_utts: int = 0,
        trim: bool = False,
        # bos_id: Optional[int] = None,
        # eos_id: Optional[int] = None,
        # pad_id: int = 0,
        # load_audio: bool = True,
        # add_misc: bool = False,
    ):
        super().__init__()

        self.mapping = {
            0: {'count': 39, 'symbol': '!'},
            1: {'count': 141, 'symbol': "'"},
            2: {'count': 706, 'symbol': '"'},
            3: {'count': 11043, 'symbol': ','},
            4: {'count': 7565, 'symbol': '.'},
            5: {'count': 141, 'symbol': ':'},
            6: {'count': 520, 'symbol': ';'},
            7: {'count': 70, 'symbol': '?'},
            8: {'count': 170543, 'symbol': ' '},
            14: {'count': 1316, 'symbol': '-'},
            16: {'count': 3, 'symbol': '['},
            17: {'count': 3, 'symbol': ']'},
            18: {'count': 50, 'symbol': '('},
            19: {'count': 50, 'symbol': ')'},
            102: {'count': 225, 'symbol': '@AA0'},
            103: {'count': 11880, 'symbol': '@AA1'},
            104: {'count': 658, 'symbol': '@AA2'},
            106: {'count': 435, 'symbol': '@AE0'},
            107: {'count': 15084, 'symbol': '@AE1'},
            108: {'count': 827, 'symbol': '@AE2'},
            110: {'count': 59710, 'symbol': '@AH0'},
            111: {'count': 14996, 'symbol': '@AH1'},
            112: {'count': 339, 'symbol': '@AH2'},
            114: {'count': 1145, 'symbol': '@AO0'},
            115: {'count': 10623, 'symbol': '@AO1'},
            116: {'count': 706, 'symbol': '@AO2'},
            118: {'count': 16, 'symbol': '@AW0'},
            119: {'count': 2962, 'symbol': '@AW1'},
            120: {'count': 255, 'symbol': '@AW2'},
            122: {'count': 273, 'symbol': '@AY0'},
            123: {'count': 8205, 'symbol': '@AY1'},
            124: {'count': 854, 'symbol': '@AY2'},
            125: {'count': 11763, 'symbol': '@B'},
            126: {'count': 3567, 'symbol': '@CH'},
            127: {'count': 33358, 'symbol': '@D'},
            128: {'count': 22814, 'symbol': '@DH'},
            130: {'count': 735, 'symbol': '@EH0'},
            131: {'count': 18100, 'symbol': '@EH1'},
            132: {'count': 1549, 'symbol': '@EH2'},
            134: {'count': 12579, 'symbol': '@ER0'},
            135: {'count': 5202, 'symbol': '@ER1'},
            136: {'count': 117, 'symbol': '@ER2'},
            138: {'count': 535, 'symbol': '@EY0'},
            139: {'count': 12971, 'symbol': '@EY1'},
            140: {'count': 1157, 'symbol': '@EY2'},
            141: {'count': 12972, 'symbol': '@F'},
            142: {'count': 4416, 'symbol': '@G'},
            143: {'count': 10564, 'symbol': '@HH'},
            145: {'count': 19161, 'symbol': '@IH0'},
            146: {'count': 19879, 'symbol': '@IH1'},
            147: {'count': 2485, 'symbol': '@IH2'},
            149: {'count': 10160, 'symbol': '@IY0'},
            150: {'count': 11012, 'symbol': '@IY1'},
            151: {'count': 754, 'symbol': '@IY2'},
            152: {'count': 3707, 'symbol': '@JH'},
            153: {'count': 21486, 'symbol': '@K'},
            154: {'count': 24768, 'symbol': '@L'},
            155: {'count': 18003, 'symbol': '@M'},
            156: {'count': 52522, 'symbol': '@N'},
            157: {'count': 5529, 'symbol': '@NG'},
            159: {'count': 1090, 'symbol': '@OW0'},
            160: {'count': 6331, 'symbol': '@OW1'},
            161: {'count': 428, 'symbol': '@OW2'},
            163: {'count': 3, 'symbol': '@OY0'},
            164: {'count': 597, 'symbol': '@OY1'},
            165: {'count': 40, 'symbol': '@OY2'},
            166: {'count': 15440, 'symbol': '@P'},
            167: {'count': 30768, 'symbol': '@R'},
            168: {'count': 33268, 'symbol': '@S'},
            169: {'count': 6225, 'symbol': '@SH'},
            170: {'count': 50815, 'symbol': '@T'},
            171: {'count': 2727, 'symbol': '@TH'},
            173: {'count': 22, 'symbol': '@UH0'},
            174: {'count': 2086, 'symbol': '@UH1'},
            175: {'count': 94, 'symbol': '@UH2'},
            177: {'count': 894, 'symbol': '@UW0'},
            178: {'count': 10345, 'symbol': '@UW1'},
            179: {'count': 477, 'symbol': '@UW2'},
            180: {'count': 15056, 'symbol': '@V'},
            181: {'count': 15645, 'symbol': '@W'},
            182: {'count': 3400, 'symbol': '@Y'},
            183: {'count': 21323, 'symbol': '@Z'},
            184: {'count': 483, 'symbol': '@ZH'},
        }

        for i, key in enumerate(self.mapping):
            self.mapping[key]["symbol"] = i

        # Load data from manifests
        audio_files = []
        total_duration = 0
        if isinstance(manifest_filepath, str):
            manifest_filepath = [manifest_filepath]
        for manifest_file in manifest_filepath:
            with open(expanduser(manifest_file), 'r') as f:
                logging.info(f"Loading dataset from {manifest_file}.")
                for line in f:
                    item = json.loads(line)
                    audio_files.append({"audio_filepath": item["audio_filepath"], "duration": item["duration"]})
                    total_duration += item["duration"]

        # Prune data according to max/min_duration and ignore_file
        total_dataset_len = len(audio_files)
        logging.info(f"Loaded dataset with {total_dataset_len} files totalling {total_duration} seconds.")
        self.data = []
        dataitem = py_collections.namedtuple(
            typename='AudioTextEntity', field_names='audio_file duration text_tokens pitches'
        )

        if ignore_file:
            logging.info(f"using {ignore_file} to prune dataset.")
            with open(ignore_file, "rb") as f:
                wavs_to_ignore = pickle.load(f)

        pruned_duration = 0
        pruned_items = 0
        # error = []
        for item in audio_files:
            LJ_id = item["audio_filepath"].split("/")[-1].split(".")[0]

            # Prune according to duration & the ignore file
            if (min_duration and item["duration"] < min_duration) or (
                max_duration and item["duration"] > max_duration
            ):
                pruned_duration += item["duration"]
                pruned_items += 1
                continue
            if ignore_file:
                found_id = False
                for i, wav_id in enumerate(wavs_to_ignore):
                    if LJ_id == wav_id:
                        pruned_duration += item["duration"]
                        wavs_to_ignore.pop(i)
                        pruned_items += 1
                        found_id = True
                        break
                if found_id:
                    continue
            # Else not pruned, load durations file
            durations = torch.load(Path(supplementary_dir) / f"{LJ_id}_mfa_adjusted_enctxt_tkndur.pt")

            # Load pitch file (F0s)
            pitches = torch.load(Path(supplementary_dir) / f"{LJ_id}_melodia_f0min80_f0max800_harm1.0_mps0.0.pt")

            # Load energy file (L2-norm of the amplitude of each STFT frame of an utterance)
            energies = torch.from_numpy(np.load(Path(supplementary_dir) / f"{LJ_id}_stft_energy.npy"))

            # Get text tokens from lookup to match with durations
            text_tokens = [self.mapping[int(i)]["symbol"] for i in durations["text_encoded"]]

            self.data.append(
                dataitem(
                    audio_file=item["audio_filepath"],
                    duration=durations["token_duration"],
                    pitches=torch.clamp(pitches['f0'], min=1e-5),
                    energies=energies,
                    text_tokens=text_tokens,
                )
            )

        # print(error)
        # torch.save(error, "error.pt")
        # exit()

        logging.info(f"Pruned {pruned_items} files and {pruned_duration} seconds.")
        logging.info(f"Final dataset contains {len(self.data)} files and {total_duration-pruned_duration} seconds.")

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate)
        self.trim = trim

    def __getitem__(self, index):
        sample = self.data[index]

        features = self.featurizer.process(sample.audio_file, trim=self.trim)
        f, fl = features, torch.tensor(features.shape[0]).long()
        t, tl = torch.tensor(sample.text_tokens).long(), torch.tensor(len(sample.text_tokens)).long()

        return f, fl, t, tl, sample.duration, sample.pitches, sample.energies

    def __len__(self):
        return len(self.data)

    def _collate_fn(self, batch):
        pad_id = len(self.mapping)
        _, audio_lengths, _, tokens_lengths, duration, pitches = zip(*batch)

        max_audio_len = 0
        max_audio_len = max(audio_lengths).item()
        max_tokens_len = max(tokens_lengths).item()
        max_durations_len = max([len(i) for i in duration])
        max_pitches_len = max([len(i) for i in pitches])
        max_energies_len = max([len(i) for i in energies])

        # Add padding where necessary
        audio_signal, tokens, duration_batched, pitches_batched, energies_batched = [], [], [], [], []
        for sig, sig_len, tokens_i, tokens_i_len, duration, pitch, energy in batch:
            # TODO: Refactoring -- write general padding utility function for cleanliness
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)

            tokens_i_len = tokens_i_len.item()
            if tokens_i_len < max_tokens_len:
                pad = (0, max_tokens_len - tokens_i_len)
                tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
            tokens.append(tokens_i)

            if len(duration) < max_durations_len:
                pad = (0, max_durations_len - len(duration))
                duration = torch.nn.functional.pad(duration, pad)
            duration_batched.append(duration)

            if len(pitch) < max_pitches_len:
                pad = (0, max_pitches_len - len(pitch))
                pitch = torch.nn.functional.pad(pitch, pad)
            pitches_batched.append(pitch)

            if len(energy) < max_energies_len:
                pad = (0, max_energies_len - len(energy))
                energy = torch.nn.functional.pad(energy, pad)
            energies_batched.append(energy)

        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
        tokens = torch.stack(tokens)
        tokens_lengths = torch.stack(tokens_lengths)
        duration_batched = torch.stack(duration_batched)
        pitches_batched = torch.stack(pitches_batched)
        energies_batched = torch.stack(energies_batched)

        return (
            audio_signal,
            audio_lengths,
            tokens,
            tokens_lengths,
            duration_batched,
            pitches_batched,
            energies_batched,
        )
