# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any, Dict, Optional, Union
from pathlib import Path
import pickle
import math

import torch
import numpy as np
import librosa


from nemo.core.classes import Dataset
from nemo.core.neural_types.elements import *
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging
from nemo.collections.common.parts.preprocessing.parsers import make_parser
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.tts.torch.helpers import beta_binomial_prior_distribution

CONSTANT = 1e-5


class TextMelAudioDataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'transcripts': NeuralType(('B', 'T'), TokenIndex()),
            'transcript_length': NeuralType(('B'), LengthsType()),
            'mels': NeuralType(('B', 'D', 'T'), TokenIndex()),
            'mel_length': NeuralType(('B'), LengthsType()),
            'audio': NeuralType(('B', 'T'), AudioSignal()),
            'audio_length': NeuralType(('B'), LengthsType()),
            'duration_prior': NeuralType(('B', 'T'), TokenDurationType()),
            'pitches': NeuralType(('B', 'T'), RegressionValuesType()),
            'energies': NeuralType(('B', 'T'), RegressionValuesType()),
        }

    def __init__(
        self,
        manifest_filepath: str,
        sample_rate: int,
        supplementary_folder: Path,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        ignore_file: Optional[str] = None,
        trim: bool = False,
        n_fft=1024,
        win_length=None,
        hop_length=None,
        window="hann",
        n_mels=64,
        lowfreq=0,
        highfreq=None,
        pitch_fmin=None,
        pitch_fmax=None,
        pitch_avg=0,
        pitch_std=1,
    ):
        """Dataset that loads audio, log mel specs, text tokens, duration / attention priors, pitches, and energies.
        Log mels, priords, pitches, and energies will be computed on the fly and saved in the supplementary_folder if
        they did not exist before.

        Note: This dataset currently only support characters. Phone-support is next to be added.


            destination (str, Path): Path to a directory containing the main data set folder, Similar to the directory
            provided to the preprocessor script, which generates this dataset.
            subdir (str): Either 'train', or 'valid', when using the standard script for generation.
            n_fft (int): STFT parameter. Also detrmines the STFT filter length.
            hop_length (int): STFT parameter.
            num_snr (int): number of noisy samples per clean audio in the original dataset.


        Args:
            manifest_filepath (str, Path, List[str, Path]): Path(s) to the .json manifests containing information on the
                dataset. Each line in the .json file should be valid json. Note: the .json file itself is not valid
                json. Each line should contain the following:
                    "audio_filepath": <PATH_TO_WAV>
                    "mel_filepath": <PATH_TO_LOG_MEL_PT> (Optional)
                    "duration": <Duration of audio clip in seconds> (Optional)
                    "text": <THE_TRANSCRIPT> (Optional)
            sample_rate (int): The sample rate of the audio. Or the sample rate that we will resample all files to.
            supplementary_folder (Path): A folder that contains or will contain extra information such as log_mel if not
                specified in the manifest .json file. It will also contain priors, pitches, and energies
            max_duration (Optional[float]): Max duration of audio clips in seconds. All samples exceeding this will be
                pruned prior to training. Note: Requires "duration" to be set in the manifest file. It does not load
                audio to compute duration. Defaults to None which does not prune.
            min_duration (Optional[float]): Min duration of audio clips in seconds. All samples lower than this will be
                pruned prior to training. Note: Requires "duration" to be set in the manifest file. It does not load
                audio to compute duration. Defaults to None which does not prune.
            ignore_file (Optional[str, Path]): The location of a pickle-saved list of audio_ids (the stem of the audio
                files) that will be pruned prior to training. Defaults to None which does not prune.
            trim (Optional[bool]): Whether to apply librosa.effects.trim to the audio file. Defaults to False.
            n_fft (Optional[int]): The number of fft samples. Defaults to 1024
            win_length (Optional[int]): The length of the stft windows. Defaults to None which uses n_fft.
            hop_length (Optional[int]): The hope length between fft computations. Defaults to None which uses n_fft//4.
            window (Optional[str]): One of 'hann', 'hamming', 'blackman','bartlett', 'none'. Which corresponds to the
                equivalent torch window function.
            n_mels (Optional[int]): The number of mel filters. Defaults to 64.
            lowfreq (Optional[int]): The lowfreq input to the mel filter calculation. Defaults to 0.
            highfreq (Optional[int]): The highfreq input to the mel filter calculation. Defaults to None.
            pitch_fmin (Optional[int]): The fmin input to librosa.pyin. Defaults to None.
            pitch_fmax (Optional[int]): The fmax input to librosa.pyin. Defaults to None.
            pitch_avg (Optional[float]): The mean that we use to normalize the pitch. Defaults to 0.
            pitch_std (Optional[float]): The std that we use to normalize the pitch. Defaults to 1.
        """
        super().__init__()

        self.pitch_fmin = pitch_fmin
        self.pitch_fmax = pitch_fmax
        self.pitch_avg = pitch_avg
        self.pitch_std = pitch_std
        self.win_length = win_length or n_fft
        self.sample_rate = sample_rate
        self.hop_len = hop_length or n_fft // 4

        audio_files = []
        total_duration = 0
        self.parser = make_parser(name="en")
        Path(supplementary_folder).mkdir(parents=True, exist_ok=True)
        self.supplementary_folder = supplementary_folder

        # Load data from manifests
        # Note: audio is always required, even for text -> mel_spectrogram models, due to the fact that most models
        # extract pitch from the audio
        # Note: mel_filepath is not required and if not present, we then check the supplementary folder. If we fail, we
        # compute the mel on the fly and save it to the supplementary folder
        # Note: text is not required. Any models that require on text (spectrogram generators, end-to-end models) will
        # fail if not set. However vocoders (mel -> audio) will be able to work without text
        if isinstance(manifest_filepath, str):
            manifest_filepath = [manifest_filepath]
        for manifest_file in manifest_filepath:
            with open(Path(manifest_file).expanduser(), 'r') as f:
                logging.info(f"Loading dataset from {manifest_file}.")
                for line in f:
                    item = json.loads(line)
                    # Grab audio, text, mel if they exist
                    file_info = {}
                    file_info["audio_filepath"] = item["audio_filepath"]
                    file_info["mel_filepath"] = item["mel_filepath"] if "mel_filepath" in item else None
                    file_info["duration"] = item["duration"] if "duration" in item else None
                    # Parse text
                    file_info["text_tokens"] = None
                    if "text" in item:
                        text = item["text"]
                        text_tokens = self.parser(text)
                        file_info["text_tokens"] = text_tokens
                    audio_files.append(file_info)
                    if file_info["duration"] is None:
                        logging.info(
                            "Not all audio files have duration information. Duration logging will be disabled."
                        )
                        total_duration = None
                    if total_duration is not None:
                        total_duration += item["duration"]

        logging.info(f"Loaded dataset with {len(audio_files)} files.")
        if total_duration is not None:
            logging.info(f"Dataset contains {total_duration/3600:.2f} hours.")

        self.data = []

        if ignore_file:
            logging.info(f"using {ignore_file} to prune dataset.")
            with open(Path(ignore_file).expanduser(), "rb") as f:
                wavs_to_ignore = set(pickle.load(f))

        pruned_duration = 0 if total_duration is not None else None
        pruned_items = 0
        for item in audio_files:
            audio_path = item['audio_filepath']
            audio_id = Path(audio_path).stem

            # Prune data according to min/max_duration & the ignore file
            if total_duration is not None:
                if (min_duration and item["duration"] < min_duration) or (
                    max_duration and item["duration"] > max_duration
                ):
                    pruned_duration += item["duration"]
                    pruned_items += 1
                    continue

            if ignore_file and (audio_id in wavs_to_ignore):
                pruned_items += 1
                pruned_duration += item["duration"]
                wavs_to_ignore.remove(audio_id)
                continue

            self.data.append(item)

        logging.info(f"Pruned {pruned_items} files. Final dataset contains {len(self.data)} files")
        if pruned_duration is not None:
            logging.info(
                f"Pruned {pruned_duration/3600:.2f} hours. Final dataset contains "
                f"{(total_duration-pruned_duration)/3600:.2f} hours."
            )

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate)
        self.trim = trim

        filterbanks = torch.tensor(
            librosa.filters.mel(sample_rate, n_fft, n_mels=n_mels, fmin=lowfreq, fmax=highfreq), dtype=torch.float
        ).unsqueeze(0)
        self.fb = filterbanks

        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(win_length, periodic=False) if window_fn else None

        self.stft = lambda x: torch.stft(
            input=x,
            n_fft=n_fft,
            hop_length=self.hop_len,
            win_length=self.win_length,
            window=window_tensor.to(torch.float),
        )

    def __getitem__(self, index):
        spec = None
        sample = self.data[index]

        features = self.featurizer.process(sample["audio_filepath"], trim=self.trim)
        audio, audio_length = features, torch.tensor(features.shape[0]).long()
        text, text_length = torch.tensor(sample["text_tokens"]).long(), torch.tensor(len(sample["text_tokens"])).long()
        audio_stem = Path(sample["audio_filepath"]).stem

        # Load mel if it exists
        mel_path = sample["mel_filepath"]
        if mel_path and Path(mel_path).exists():
            log_mel = torch.load(mel_path)
        else:
            mel_path = Path(self.supplementary_folder) / f"mel_{audio_stem}.pt"
            if mel_path.exists():
                log_mel = torch.load(mel_path)
            else:
                # disable autocast to get full range of stft values
                with torch.cuda.amp.autocast(enabled=False):
                    spec = self.stft(audio)

                    # guard is needed for sqrt if grads are passed through
                    guard = CONSTANT  # TODO: Enable 0 if not self.use_grads else CONSTANT
                    if spec.dtype in [torch.cfloat, torch.cdouble]:
                        spec = torch.view_as_real(spec)
                    spec = torch.sqrt(spec.pow(2).sum(-1) + guard)

                    mel = torch.matmul(self.fb.to(spec.dtype), spec)

                    log_mel = torch.log(torch.clamp(mel, min=torch.finfo(mel.dtype).tiny))
                    torch.save(log_mel, mel_path)

        log_mel = log_mel.squeeze(0)
        log_mel_length = torch.tensor(log_mel.shape[1]).long()

        ### Make duration attention prior if not exist in the supplementary folder
        text_tokens = text
        prior_path = Path(self.supplementary_folder) / f"pr_tl{len(text_tokens)}_al_{log_mel_length}.pt"
        if prior_path.exists():
            duration_prior = torch.load(prior_path)
        else:
            duration_prior = beta_binomial_prior_distribution(len(text), log_mel_length)
            duration_prior = torch.from_numpy(duration_prior)
            torch.save(duration_prior, prior_path)

        # Load pitch file (F0s)
        pitch_path = (
            Path(self.supplementary_folder)
            / f"{audio_stem}_pitch_pyin_fmin{self.pitch_fmin}_fmax{self.pitch_fmax}_fl{self.win_length}_hs{self.hop_len}.pt"
        )
        if pitch_path.exists():
            pitch = torch.load(pitch_path)
        else:
            pitch, _, _ = librosa.pyin(
                audio.numpy(),
                fmin=self.pitch_fmin,
                fmax=self.pitch_fmax,
                frame_length=self.win_length,
                sr=self.sample_rate,
                fill_na=0.0,
            )
            pitch = torch.from_numpy(pitch)
            torch.save(pitch, pitch_path)
        # Standize pitch
        pitch -= self.pitch_avg
        pitch[pitch == -self.pitch_avg] = 0.0  # Zero out values that were perviously zero
        pitch /= self.pitch_std

        # Load energy file (L2-norm of the amplitude of each STFT frame of an utterance)
        energy_path = Path(self.supplementary_folder) / f"{audio_stem}_energy_wl{self.win_length}_hs{self.hop_len}.pt"
        if energy_path.exists():
            energy = torch.load(energy_path)
        else:
            if spec is None:
                spec = self.stft(audio)
            energy = torch.linalg.norm(spec.squeeze(0), axis=0)
            logging.warning(f"energy: {energy.shape}")
            # Save to new file
            torch.save(energy, energy_path)

        return text, text_length, log_mel, log_mel_length, audio, audio_length, duration_prior, pitch, energy

    def __len__(self):
        return len(self.data)

    def _collate_fn(self, batch):
        pad_id = self.parser._blank_id
        log_mel_pad = torch.finfo(batch[0][2].dtype).tiny

        _, tokens_lengths, _, log_mel_lengths, _, audio_lengths, duration_priors_list, pitches, energies = zip(*batch)

        max_tokens_len = max(tokens_lengths).item()
        max_log_mel_len = max(log_mel_lengths)
        max_audio_len = max(audio_lengths).item()
        max_pitches_len = max([len(i) for i in pitches])
        max_energies_len = max([len(i) for i in energies])
        if max_pitches_len != max_energies_len or max_pitches_len != max_log_mel_len:
            logging.warning(
                f"max_pitches_len: {max_pitches_len} != max_energies_len: {max_energies_len} != "
                f"max_mel_len:{max_log_mel_len}. Your training run will error out!"
            )

        # Define empty lists to be batched
        duration_priors = torch.zeros(
            len(duration_priors_list),
            max([prior_i.shape[0] for prior_i in duration_priors_list]),
            max([prior_i.shape[1] for prior_i in duration_priors_list]),
        )
        audios, tokens, log_mels, pitches, energies = [], [], [], [], []
        for i, sample_tuple in enumerate(batch):
            token, token_len, log_mel, log_mel_len, audio, audio_len, duration_prior, pitch, energy = sample_tuple
            # Pad text tokens
            token_len = token_len.item()
            if token_len < max_tokens_len:
                pad = (0, max_tokens_len - token_len)
                token = torch.nn.functional.pad(token, pad, value=pad_id)
            tokens.append(token)
            # Pad mel
            log_mel_len = log_mel_len
            if log_mel_len < max_log_mel_len:
                pad = (0, max_log_mel_len - log_mel_len)
                log_mel = torch.nn.functional.pad(log_mel, pad, value=log_mel_pad)
            log_mels.append(log_mel)
            # Pad audio
            audio_len = audio_len.item()
            if audio_len < max_audio_len:
                pad = (0, max_audio_len - audio_len)
                audio = torch.nn.functional.pad(audio, pad)
            audios.append(audio)
            # Pad duration_prior
            duration_priors[i, : duration_prior.shape[0], : duration_prior.shape[1]] = duration_prior
            # Pad pitch
            if len(pitch) < max_pitches_len:
                pad = (0, max_pitches_len - len(pitch))
                pitch = torch.nn.functional.pad(pitch, pad)
            pitches.append(pitch)
            # Pad energy
            if len(energy) < max_energies_len:
                pad = (0, max_energies_len - len(energy))
                energy = torch.nn.functional.pad(energy, pad)
            energies.append(energy)

        audios = torch.stack(audios)
        audio_lengths = torch.stack(audio_lengths)
        tokens = torch.stack(tokens)
        tokens_lengths = torch.stack(tokens_lengths)
        log_mels = torch.stack(log_mels)
        log_mel_lengths = torch.stack(log_mel_lengths)
        pitches = torch.stack(pitches)
        energies = torch.stack(energies)

        logging.debug(f"audios: {audios.shape}")
        logging.debug(f"audio_lengths: {audio_lengths.shape}")
        logging.debug(f"tokens: {tokens.shape}")
        logging.debug(f"tokens_lengths: {tokens_lengths.shape}")
        logging.debug(f"log_mels: {log_mels.shape}")
        logging.debug(f"log_mel_lengths: {log_mel_lengths.shape}")
        logging.debug(f"duration_priors: {duration_priors.shape}")
        logging.debug(f"pitches: {pitches.shape}")
        logging.debug(f"energies: {energies.shape}")

        return (tokens, tokens_lengths, log_mels, log_mel_lengths, duration_priors, pitches, energies)
