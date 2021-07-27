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
        n_fft=None,
        win_length=320,
        hop_length=160,
        window="hann",
        n_mels=64,
        lowfreq=0,
        highfreq=None,
        pitch_fmin=None,
        pitch_fmax=None,
        pitch_avg=0,
        pitch_std=1,
    ):
        """TODO: Add description
        """
        super().__init__()

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
        self.register_buffer("fb", filterbanks)

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
            input=x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window_tensor.to(torch.float),
        )

        self.pitch_fmin = pitch_fmin
        self.pitch_fmax = pitch_fmax
        self.pitch_avg = pitch_avg
        self.pitch_std = pitch_std
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def __getitem__(self, index):
        spec = None
        sample = self.data[index]

        features = self.featurizer.process(sample["audio_file"], trim=self.trim)
        audio, audio_length = features, torch.tensor(features.shape[0]).long()
        text, text_length = sample["text_tokens"].long(), torch.tensor(len(sample["text_tokens"])).long()

        # Load mel if it exists
        audio_stem = Path(sample["audio_file"]).stem
        mel_path = Path(self.supplementary_folder) / f"mel_{audio_stem}.npy"
        if mel_path.exists():
            log_mel = np.load(mel_path)
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
                np.save(mel_path, log_mel)

        log_mel_length = mel.shape[0]

        ### Make duration attention prior if not exist in the supplementary folder
        text_tokens = text
        prior_path = Path(self.supplementary_folder) / f"pr_tl{len(text_tokens)}_al_{log_mel_length}.npy"
        if prior_path.exists():
            duration_prior = np.load(prior_path)
        else:
            duration_prior = beta_binomial_prior_distribution(len(text), log_mel_length)
            np.save(prior_path, duration_prior)

        # Load pitch file (F0s)
        pitch_path = (
            Path(self.supplementary_folder)
            / f"{audio_stem}_pitch_pyin_fmin{self.pitch_fmin}_fmax{self.pitch_fmax}_fl{self.win_length}_hs{self.hop_len}.npy"
        )
        if pitch_path.exists():
            pitch = np.load(pitch_path)
        else:
            pitch, _, _ = librosa.pyin(
                audio.numpy(),
                fmin=self.pitch_fmin,
                fmax=self.pitch_fmax,
                frame_length=self.win_length,
                sr=self.sample_rate,
                fill_na=0.0,
            )
            np.save(pitch_path, pitch)
        # Standize pitch
        pitch -= self.pitch_avg
        pitch[pitch == -self.pitch_avg] = 0.0  # Zero out values that were perviously zero
        pitch /= self.pitch_std

        # Load energy file (L2-norm of the amplitude of each STFT frame of an utterance)
        energy_path = Path(self.supplementary_folder) / f"{audio_stem}_energy_wl{self.win_length}_hs{self.hop_len}.npy"
        if energy_path.exists():
            energy = np.load(energy_path)
        else:
            if spec is None:
                spec = self.stft(audio)
            energy = np.linalg.norm(spec, axis=0)  # axis=0 since librosa.stft -> (freq bins, frames)
            # Save to new file
            np.save(energy_path, energy)

        return text, text_length, log_mel, log_mel_length, audio, audio_length, duration_prior, pitch, energy

    def __len__(self):
        return len(self.data)

    def _collate_fn(self, batch):
        pad_id = self.parser._blank_id
        log_mel_pad = torch.finfo(batch[3][0].dtype).tiny

        _, tokens_lengths, _, log_mel_lengths, _, audio_lengths, _, pitches, energies = zip(*batch)

        max_tokens_len = max(tokens_lengths).item()
        max_log_mel_len = max([len(i) for i in log_mel_lengths])
        max_audio_len = max(audio_lengths).item()
        max_pitches_len = max([len(i) for i in pitches])
        max_energies_len = max([len(i) for i in energies])
        if max_pitches_len != max_energies_len or max_pitches_len != max_mel_len:
            logging.warning(
                f"max_pitches_len: {max_pitches_len} != max_energies_len: {max_energies_len} != "
                f"max_mel_len:{max_mel_len}. Your training run will error out!"
            )

        # Define empty lists to be batched
        duration_priors_list = batch[4]
        duration_priors = torch.zeros(
            len(duration_priors_list),
            max([prior_i.shape[0] for prior_i in duration_priors_list]),
            max([prior_i.shape[1] for prior_i in duration_priors_list]),
        )
        audios, tokens, log_mels, pitches, energies = [], [], [], [], []
        for sample_tuple in batch:
            token, token_len, log_mel, log_mel_len, audio, audio_len, duration_prior, pitch, energy = sample_tuple
            # Pad text tokens
            token_len = token_len.item()
            if token_len < max_tokens_len:
                pad = (0, max_tokens_len - token_len)
                token = torch.nn.functional.pad(token, pad, value=pad_id)
            tokens.append(token)
            # Pad mel
            log_mel_len = log_mel_len.item()
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
            pitch = pitch.squeeze(0)
            if len(pitch) < max_pitches_len:
                pad = (0, max_pitches_len - len(pitch))
                pitch = torch.nn.functional.pad(pitch.squeeze(0), pad)
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

        return (tokens, tokens_lengths, log_mels, log_mel_lengths, duration_priors, pitches, energies)
