# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import librosa
import torch
from nemo_text_processing.text_normalization.normalize import Normalizer
from tqdm import tqdm

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.common.parts.patch_utils import stft_patch
from nemo.collections.tts.data.tokenizers import BaseTokenizer
from nemo.collections.tts.torch.helpers import beta_binomial_prior_distribution, general_padding
from nemo.core.classes import Dataset
from nemo.utils import logging

VALID_SUPPLEMENTARY_DATA_TYPES = {'mel', 'durations', 'duration_prior', 'pitch', 'energy'}


class TTSDataset(Dataset):
    def __init__(
        self,
        manifest_filepath: str,
        sample_rate: int,
        text_tokenizer: Union[BaseTokenizer, Callable[[str], List[int]]],
        text_normalizer: Union[Normalizer, Callable[[str], str]] = None,
        text_normalizer_call_args: Dict = None,
        text_tokenizer_pad_id: int = None,
        sup_data_types: List[str] = None,
        sup_data_folder: Union[Path, str] = None,
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
        **kwargs,
    ):
        super().__init__()

        self.text_normalizer = text_normalizer
        self.text_normalizer_call = (
            self.text_normalizer.normalize if isinstance(self.text_normalizer, Normalizer) else self.text_normalizer
        )
        self.text_normalizer_call_args = text_normalizer_call_args

        self.text_tokenizer = text_tokenizer

        if isinstance(self.text_tokenizer, BaseTokenizer):
            self.text_tokenizer_pad_id = text_tokenizer.pad
        else:
            if text_tokenizer_pad_id is None:
                raise ValueError(f"text_tokenizer_pad_id must be specified if text_tokenizer is not BaseTokenizer")
            self.text_tokenizer_pad_id = text_tokenizer_pad_id

        if isinstance(manifest_filepath, str):
            manifest_filepath = [manifest_filepath]
        self.manifest_filepath = manifest_filepath

        if sup_data_folder is not None:
            Path(sup_data_folder).mkdir(parents=True, exist_ok=True)
            self.sup_data_folder = sup_data_folder

        self.sup_data_types = sup_data_types if sup_data_types is not None else []
        self.sup_data_types_set = set(sup_data_types)

        self.data = []
        audio_files = []
        total_duration = 0
        for manifest_file in self.manifest_filepath:
            with open(Path(manifest_file).expanduser(), 'r') as f:
                logging.info(f"Loading dataset from {manifest_file}.")
                for line in tqdm(f):
                    item = json.loads(line)

                    file_info = {
                        "audio_filepath": item["audio_filepath"],
                        "mel_filepath": item["mel_filepath"] if "mel_filepath" in item else None,
                        "duration": item["duration"] if "duration" in item else None,
                        "text_tokens": None,
                    }

                    if "text" in item:
                        text = item["text"]

                        if self.text_normalizer is not None:
                            text = self.text_normalizer_call(text, **self.text_normalizer_call_args)

                        text_tokens = self.text_tokenizer(text)
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
            logging.info(f"Dataset contains {total_duration / 3600:.2f} hours.")

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
                f"Pruned {pruned_duration / 3600:.2f} hours. Final dataset contains "
                f"{(total_duration - pruned_duration) / 3600:.2f} hours."
            )

        self.sample_rate = sample_rate
        self.featurizer = WaveformFeaturizer(sample_rate=self.sample_rate)
        self.trim = trim

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.lowfreq = lowfreq
        self.highfreq = highfreq
        self.window = window
        self.win_length = win_length or self.n_fft
        self.hop_length = hop_length
        self.hop_len = self.hop_length or self.n_fft // 4
        self.fb = torch.tensor(
            librosa.filters.mel(
                self.sample_rate, self.n_fft, n_mels=self.n_mels, fmin=self.lowfreq, fmax=self.highfreq
            ),
            dtype=torch.float,
        ).unsqueeze(0)

        window_fn = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }.get(self.window, None)

        self.stft = lambda x: stft_patch(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_length,
            window=window_fn(self.win_length, periodic=False).to(torch.float) if window_fn else None,
        )

        for data_type in self.sup_data_types:
            if data_type not in VALID_SUPPLEMENTARY_DATA_TYPES:
                raise NotImplementedError(f"Current implementation of TTSDataset doesn't support {data_type} type.")

            getattr(self, f"add_{data_type}")(**kwargs)

    def add_mel(self, **kwargs):
        pass

    def add_durations(self, **kwargs):
        durs_file = kwargs.pop('durs_file')
        durs_type = kwargs.pop('durs_type')

        audio_stem2durs = torch.load(durs_file)
        self.durs = []

        for tag in [Path(d["audio_filepath"]).stem for d in self.data]:
            durs = audio_stem2durs[tag]
            if durs_type == "aligner-based":
                self.durs.append(durs)
            else:
                raise NotImplementedError(
                    f"{durs_type} duration type is not supported. Only align-based is supported at this moment."
                )

    def add_duration_prior(self, **kwargs):
        pass

    def add_pitch(self, **kwargs):
        self.pitch_fmin = kwargs.pop("pitch_fmin")
        self.pitch_fmax = kwargs.pop("pitch_fmax")
        self.pitch_avg = kwargs.pop("pitch_avg", None)
        self.pitch_std = kwargs.pop("pitch_std", None)

        self.pitch_norm = False
        if self.pitch_avg is not None and self.pitch_std is not None:
            self.pitch_norm = True

    def add_energy(self, **kwargs):
        pass

    def get_spec(self, audio):
        with torch.cuda.amp.autocast(enabled=False):
            spec = self.stft(audio)
            if spec.dtype in [torch.cfloat, torch.cdouble]:
                spec = torch.view_as_real(spec)
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        return spec

    def get_log_mel(self, audio):
        with torch.cuda.amp.autocast(enabled=False):
            spec = self.get_spec(audio)
            mel = torch.matmul(self.fb.to(spec.dtype), spec)
            log_mel = torch.log(torch.clamp(mel, min=torch.finfo(mel.dtype).tiny))
        return log_mel

    def __getitem__(self, index):
        sample = self.data[index]
        audio_stem = Path(sample["audio_filepath"]).stem

        features = self.featurizer.process(sample["audio_filepath"], trim=self.trim)
        audio, audio_length = features, torch.tensor(features.shape[0]).long()

        text = torch.tensor(sample["text_tokens"]).long()
        text_length = torch.tensor(len(sample["text_tokens"])).long()

        log_mel, log_mel_length = None, None
        if "mel" in self.sup_data_types_set:
            mel_path = sample["mel_filepath"]

            if mel_path is not None and Path(mel_path).exists():
                log_mel = torch.load(mel_path)
            else:
                mel_path = Path(self.sup_data_folder) / f"mel_{audio_stem}.pt"

                if mel_path.exists():
                    log_mel = torch.load(mel_path)
                else:
                    log_mel = self.get_log_mel(audio)
                    torch.save(log_mel, mel_path)

            log_mel = log_mel.squeeze(0)
            log_mel_length = torch.tensor(log_mel.shape[1]).long()

        durations = None
        if "durations" in self.sup_data_types_set:
            durations = self.durs[index]

        duration_prior = None
        if "duration_prior" in self.sup_data_types_set:
            prior_path = Path(self.sup_data_folder) / f"pr_{audio_stem}.pt"

            if prior_path.exists():
                duration_prior = torch.load(prior_path)
            else:
                log_mel_length = torch.tensor(self.get_log_mel(audio).squeeze(0).shape[1]).long()
                duration_prior = beta_binomial_prior_distribution(text_length, log_mel_length)
                duration_prior = torch.from_numpy(duration_prior)
                torch.save(duration_prior, prior_path)

        pitch = None
        if "pitch" in self.sup_data_types_set:
            pitch_name = (
                f"{audio_stem}_pitch_pyin_norm_{self.pitch_norm}"
                f"_fmin{self.pitch_fmin}_fmax{self.pitch_fmax}_"
                f"fl{self.win_length}_hs{self.hop_len}.pt"
            )

            pitch_path = Path(self.sup_data_folder) / pitch_name
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

            if self.pitch_norm:
                pitch -= self.pitch_avg
                pitch[pitch == -self.pitch_avg] = 0.0  # Zero out values that were perviously zero
                pitch /= self.pitch_std

        energy = None
        if "energy" in self.sup_data_types_set:
            energy_path = Path(self.sup_data_folder) / f"{audio_stem}_energy_wl{self.win_length}_hs{self.hop_len}.pt"
            if energy_path.exists():
                energy = torch.load(energy_path)
            else:
                spec = self.get_spec(audio)
                energy = torch.linalg.norm(spec.squeeze(0), axis=0)
                torch.save(energy, energy_path)

        return (
            text,
            text_length,
            audio,
            audio_length,
            log_mel,
            log_mel_length,
            durations,
            duration_prior,
            pitch,
            energy,
        )

    def __len__(self):
        return len(self.data)

    def _collate_fn(self, batch):
        (
            _,
            tokens_lengths,
            _,
            audio_lengths,
            _,
            log_mel_lengths,
            durations_list,
            duration_priors_list,
            pitches,
            energies,
        ) = zip(*batch)

        max_tokens_len = max(tokens_lengths).item()
        max_audio_len = max(audio_lengths).item()
        max_log_mel_len = max(log_mel_lengths) if "mel" in self.sup_data_types_set else None
        max_durations_len = max([len(i) for i in durations_list]) if "durations" in self.sup_data_types_set else None
        max_pitches_len = max([len(i) for i in pitches]) if "pitch" in self.sup_data_types_set else None
        max_energies_len = max([len(i) for i in energies]) if "energy" in self.sup_data_types_set else None

        if "mel" in self.sup_data_types_set:
            log_mel_pad = torch.finfo(batch[0][2].dtype).tiny

        # Define empty lists to be batched
        duration_priors = (
            torch.zeros(
                len(duration_priors_list),
                max([prior_i.shape[0] for prior_i in duration_priors_list]),
                max([prior_i.shape[1] for prior_i in duration_priors_list]),
            )
            if "duration_prior" in self.sup_data_types_set
            else []
        )
        tokens, audios, log_mels, durations_list, pitches, energies = [], [], [], [], [], []

        for i, sample_tuple in enumerate(batch):
            (
                token,
                token_len,
                audio,
                audio_len,
                log_mel,
                log_mel_len,
                durations,
                duration_prior,
                pitch,
                energy,
            ) = sample_tuple

            token = general_padding(token, token_len.item(), max_tokens_len, pad_value=self.text_tokenizer_pad_id)
            tokens.append(token)

            audio = general_padding(audio, audio_len.item(), max_audio_len)
            audios.append(audio)

            if "mel" in self.sup_data_types_set:
                log_mels.append(general_padding(log_mel, log_mel_len, max_log_mel_len, pad_value=log_mel_pad))
            if "durations" in self.sup_data_types_set:
                durations_list.append(general_padding(durations, len(durations), max_durations_len))
            if "duration_prior" in self.sup_data_types_set:
                duration_priors[i, : duration_prior.shape[0], : duration_prior.shape[1]] = duration_prior
            if "pitch" in self.sup_data_types_set:
                pitches.append(general_padding(pitch, len(pitch), max_pitches_len))
            if "energy" in self.sup_data_types_set:
                energies.append(general_padding(energy, len(energy), max_energies_len))

        result = [torch.stack(tokens), torch.stack(tokens_lengths), torch.stack(audios), torch.stack(audio_lengths)]
        for sup_data_type in self.sup_data_types:
            if sup_data_type == "mel":
                result.extend([torch.stack(log_mels), torch.stack(log_mel_lengths)])
            elif sup_data_type == "durations":
                result.append(torch.stack(durations_list))
            elif sup_data_type == "duration_prior":
                result.append(duration_priors)
            elif sup_data_type == "pitch":
                result.append(torch.stack(pitches))
            elif sup_data_type == "energy":
                result.append(torch.stack(energies))

        return tuple(result)
