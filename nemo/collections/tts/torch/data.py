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
from nemo.collections.tts.torch.helpers import beta_binomial_prior_distribution, general_padding
from nemo.collections.tts.torch.tts_data_types import (
    DATA_STR2DATA_CLASS,
    MAIN_DATA_TYPES,
    VALID_SUPPLEMENTARY_DATA_TYPES,
    DurationPrior,
    Durations,
    Energy,
    LMTokens,
    LogMel,
    Pitch,
    WithLens,
)
from nemo.collections.tts.torch.tts_tokenizers import BaseTokenizer, EnglishCharsTokenizer, EnglishPhonemesTokenizer
from nemo.core.classes import Dataset
from nemo.utils import logging


class TTSDataset(Dataset):
    def __init__(
        self,
        manifest_filepath: str,
        sample_rate: int,
        text_tokenizer: Union[BaseTokenizer, Callable[[str], List[int]]],
        tokens: Optional[List[str]] = None,
        text_normalizer: Optional[Union[Normalizer, Callable[[str], str]]] = None,
        text_normalizer_call_args: Optional[Dict] = None,
        text_tokenizer_pad_id: Optional[int] = None,
        sup_data_types: Optional[List[str]] = None,
        sup_data_path: Optional[Union[Path, str]] = None,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        ignore_file: Optional[str] = None,
        trim: bool = False,
        n_fft=1024,
        win_length=None,
        hop_length=None,
        window="hann",
        n_mels=80,
        lowfreq=0,
        highfreq=None,
        **kwargs,
    ):
        """Dataset that loads main data types (audio and text) and specified supplementary data types (e.g. log mel, durations, pitch).
        Most supplementary data types will be computed on the fly and saved in the supplementary_folder if they did not exist before.
        Arguments for supplementary data should be also specified in this class and they will be used from kwargs (see keyword args section).
        Args:
            manifest_filepath (str, Path, List[str, Path]): Path(s) to the .json manifests containing information on the
                dataset. Each line in the .json file should be valid json. Note: the .json file itself is not valid
                json. Each line should contain the following:
                    "audio_filepath": <PATH_TO_WAV>
                    "mel_filepath": <PATH_TO_LOG_MEL_PT> (Optional)
                    "duration": <Duration of audio clip in seconds> (Optional)
                    "text": <THE_TRANSCRIPT> (Optional)
            sample_rate (int): The sample rate of the audio. Or the sample rate that we will resample all files to.
            text_tokenizer (Optional[Union[BaseTokenizer, Callable[[str], List[int]]]]): BaseTokenizer or callable which represents text tokenizer.
            tokens (Optional[List[str]]): Tokens from text_tokenizer. Should be specified if text_tokenizer is not BaseTokenizer.
            text_normalizer (Optional[Union[Normalizer, Callable[[str], str]]]): Normalizer or callable which represents text normalizer.
            text_normalizer_call_args (Optional[Dict]): Additional arguments for text_normalizer function.
            text_tokenizer_pad_id (Optional[int]): Index of padding. Should be specified if text_tokenizer is not BaseTokenizer.
            sup_data_types (Optional[List[str]]): List of supplementary data types.
            sup_data_path (Optional[Union[Path, str]]): A folder that contains or will contain supplementary data (e.g. pitch).
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
            n_mels (Optional[int]): The number of mel filters. Defaults to 80.
            lowfreq (Optional[int]): The lowfreq input to the mel filter calculation. Defaults to 0.
            highfreq (Optional[int]): The highfreq input to the mel filter calculation. Defaults to None.
        Keyword Args:
            durs_file (Optional[str]): String path to pickled durations location.
            durs_type (Optional[str]): Type of durations. Currently supported only "aligned-based".
            pitch_fmin (Optional[float]): The fmin input to librosa.pyin. Defaults to librosa.note_to_hz('C2').
            pitch_fmax (Optional[float]): The fmax input to librosa.pyin. Defaults to librosa.note_to_hz('C7').
            pitch_avg (Optional[float]): The mean that we use to normalize the pitch.
            pitch_std (Optional[float]): The std that we use to normalize the pitch.
            pitch_norm (Optional[bool]): Whether to normalize pitch (via pitch_avg and pitch_std) or not.
        """
        super().__init__()

        self.text_normalizer = text_normalizer
        self.text_normalizer_call = (
            self.text_normalizer.normalize if isinstance(self.text_normalizer, Normalizer) else self.text_normalizer
        )
        self.text_normalizer_call_args = text_normalizer_call_args

        self.text_tokenizer = text_tokenizer

        if isinstance(self.text_tokenizer, BaseTokenizer):
            self.text_tokenizer_pad_id = text_tokenizer.pad
            self.tokens = text_tokenizer.tokens
        else:
            if text_tokenizer_pad_id is None:
                raise ValueError(f"text_tokenizer_pad_id must be specified if text_tokenizer is not BaseTokenizer")

            if tokens is None:
                raise ValueError(f"tokens must be specified if text_tokenizer is not BaseTokenizer")

            self.text_tokenizer_pad_id = text_tokenizer_pad_id
            self.tokens = tokens

        if isinstance(manifest_filepath, str):
            manifest_filepath = [manifest_filepath]
        self.manifest_filepath = manifest_filepath

        if sup_data_path is not None:
            Path(sup_data_path).mkdir(parents=True, exist_ok=True)
            self.sup_data_path = sup_data_path

        self.sup_data_types = (
            [DATA_STR2DATA_CLASS[d_as_str] for d_as_str in sup_data_types] if sup_data_types is not None else []
        )
        self.sup_data_types_set = set(self.sup_data_types)

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
                        file_info["raw_text"] = item["text"]
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

        self.stft = lambda x: torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_length,
            window=window_fn(self.win_length, periodic=False).to(torch.float) if window_fn else None,
        )

        for data_type in self.sup_data_types:
            if data_type not in VALID_SUPPLEMENTARY_DATA_TYPES:
                raise NotImplementedError(f"Current implementation of TTSDataset doesn't support {data_type} type.")

            getattr(self, f"add_{data_type.name}")(**kwargs)

    def add_log_mel(self, **kwargs):
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
        self.pitch_fmin = kwargs.pop("pitch_fmin", librosa.note_to_hz('C2'))
        self.pitch_fmax = kwargs.pop("pitch_fmax", librosa.note_to_hz('C7'))
        self.pitch_avg = kwargs.pop("pitch_avg", None)
        self.pitch_std = kwargs.pop("pitch_std", None)
        self.pitch_norm = kwargs.pop("pitch_norm", False)

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
        if LogMel in self.sup_data_types_set:
            mel_path = sample["mel_filepath"]

            if mel_path is not None and Path(mel_path).exists():
                log_mel = torch.load(mel_path)
            else:
                mel_path = Path(self.sup_data_path) / f"mel_{audio_stem}.pt"

                if mel_path.exists():
                    log_mel = torch.load(mel_path)
                else:
                    log_mel = self.get_log_mel(audio)
                    torch.save(log_mel, mel_path)

            log_mel = log_mel.squeeze(0)
            log_mel_length = torch.tensor(log_mel.shape[1]).long()

        durations = None
        if Durations in self.sup_data_types_set:
            durations = self.durs[index]

        duration_prior = None
        if DurationPrior in self.sup_data_types_set:
            prior_path = Path(self.sup_data_path) / f"pr_{audio_stem}.pt"

            if prior_path.exists():
                duration_prior = torch.load(prior_path)
            else:
                log_mel_length = torch.tensor(self.get_log_mel(audio).squeeze(0).shape[1]).long()
                duration_prior = beta_binomial_prior_distribution(text_length, log_mel_length)
                duration_prior = torch.from_numpy(duration_prior)
                torch.save(duration_prior, prior_path)

        pitch, pitch_length = None, None
        if Pitch in self.sup_data_types_set:
            pitch_name = (
                f"{audio_stem}_pitch_pyin_"
                f"fmin{self.pitch_fmin}_fmax{self.pitch_fmax}_"
                f"fl{self.win_length}_hs{self.hop_len}.pt"
            )

            pitch_path = Path(self.sup_data_path) / pitch_name
            if pitch_path.exists():
                pitch = torch.load(pitch_path).float()
            else:
                pitch, _, _ = librosa.pyin(
                    audio.numpy(),
                    fmin=self.pitch_fmin,
                    fmax=self.pitch_fmax,
                    frame_length=self.win_length,
                    sr=self.sample_rate,
                    fill_na=0.0,
                )
                pitch = torch.from_numpy(pitch).float()
                torch.save(pitch, pitch_path)

            if self.pitch_avg is not None and self.pitch_std is not None and self.pitch_norm:
                pitch -= self.pitch_avg
                pitch[pitch == -self.pitch_avg] = 0.0  # Zero out values that were perviously zero
                pitch /= self.pitch_std

            pitch_length = torch.tensor(len(pitch)).long()

        energy, energy_length = None, None
        if Energy in self.sup_data_types_set:
            energy_path = Path(self.sup_data_path) / f"{audio_stem}_energy_wl{self.win_length}_hs{self.hop_len}.pt"
            if energy_path.exists():
                energy = torch.load(energy_path).float()
            else:
                spec = self.get_spec(audio)
                energy = torch.linalg.norm(spec.squeeze(0), axis=0).float()
                torch.save(energy, energy_path)

            energy_length = torch.tensor(len(energy)).long()

        return (
            audio,
            audio_length,
            text,
            text_length,
            log_mel,
            log_mel_length,
            durations,
            duration_prior,
            pitch,
            pitch_length,
            energy,
            energy_length,
        )

    def __len__(self):
        return len(self.data)

    def join_data(self, data_dict):
        result = []
        for data_type in MAIN_DATA_TYPES + self.sup_data_types:
            result.append(data_dict[data_type.name])

            if issubclass(data_type, WithLens):
                result.append(data_dict[f"{data_type.name}_lens"])

        return tuple(result)

    def general_collate_fn(self, batch):
        (
            _,
            audio_lengths,
            _,
            tokens_lengths,
            _,
            log_mel_lengths,
            durations_list,
            duration_priors_list,
            pitches,
            pitches_lengths,
            energies,
            energies_lengths,
        ) = zip(*batch)

        max_audio_len = max(audio_lengths).item()
        max_tokens_len = max(tokens_lengths).item()
        max_log_mel_len = max(log_mel_lengths) if LogMel in self.sup_data_types_set else None
        max_durations_len = max([len(i) for i in durations_list]) if Durations in self.sup_data_types_set else None
        max_pitches_len = max(pitches_lengths).item() if Pitch in self.sup_data_types_set else None
        max_energies_len = max(energies_lengths).item() if Energy in self.sup_data_types_set else None

        if LogMel in self.sup_data_types_set:
            log_mel_pad = torch.finfo(batch[0][2].dtype).tiny

        duration_priors = (
            torch.zeros(
                len(duration_priors_list),
                max([prior_i.shape[0] for prior_i in duration_priors_list]),
                max([prior_i.shape[1] for prior_i in duration_priors_list]),
            )
            if DurationPrior in self.sup_data_types_set
            else []
        )
        audios, tokens, log_mels, durations_list, pitches, energies = [], [], [], [], [], []

        for i, sample_tuple in enumerate(batch):
            (
                audio,
                audio_len,
                token,
                token_len,
                log_mel,
                log_mel_len,
                durations,
                duration_prior,
                pitch,
                pitch_length,
                energy,
                energy_length,
            ) = sample_tuple

            audio = general_padding(audio, audio_len.item(), max_audio_len)
            audios.append(audio)

            token = general_padding(token, token_len.item(), max_tokens_len, pad_value=self.text_tokenizer_pad_id)
            tokens.append(token)

            if LogMel in self.sup_data_types_set:
                log_mels.append(general_padding(log_mel, log_mel_len, max_log_mel_len, pad_value=log_mel_pad))
            if Durations in self.sup_data_types_set:
                durations_list.append(general_padding(durations, len(durations), max_durations_len))
            if DurationPrior in self.sup_data_types_set:
                duration_priors[i, : duration_prior.shape[0], : duration_prior.shape[1]] = duration_prior
            if Pitch in self.sup_data_types_set:
                pitches.append(general_padding(pitch, pitch_length.item(), max_pitches_len))
            if Energy in self.sup_data_types_set:
                energies.append(general_padding(energy, energy_length.item(), max_energies_len))

        data_dict = {
            "audio": torch.stack(audios),
            "audio_lens": torch.stack(audio_lengths),
            "text": torch.stack(tokens),
            "text_lens": torch.stack(tokens_lengths),
            "log_mel": torch.stack(log_mels) if LogMel in self.sup_data_types_set else None,
            "log_mel_lens": torch.stack(log_mel_lengths) if LogMel in self.sup_data_types_set else None,
            "durations": torch.stack(durations_list) if Durations in self.sup_data_types_set else None,
            "duration_prior": duration_priors if DurationPrior in self.sup_data_types_set else None,
            "pitch": torch.stack(pitches) if Pitch in self.sup_data_types_set else None,
            "pitch_lens": torch.stack(pitches_lengths) if Pitch in self.sup_data_types_set else None,
            "energy": torch.stack(energies) if Energy in self.sup_data_types_set else None,
            "energy_lens": torch.stack(energies_lengths) if Energy in self.sup_data_types_set else None,
        }

        return data_dict

    def _collate_fn(self, batch):
        data_dict = self.general_collate_fn(batch)
        joined_data = self.join_data(data_dict)
        return joined_data


class MixerTTSDataset(TTSDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _albert(self):
        from transformers import AlbertTokenizer  # noqa pylint: disable=import-outside-toplevel

        self.lm_model_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.lm_padding_value = self.lm_model_tokenizer._convert_token_to_id('<pad>')
        space_value = self.lm_model_tokenizer._convert_token_to_id('▁')

        self.id2lm_tokens = {}
        for i, d in enumerate(self.data):
            raw_text = d["raw_text"]

            assert isinstance(self.text_tokenizer, EnglishPhonemesTokenizer) or isinstance(
                self.text_tokenizer, EnglishCharsTokenizer
            )
            if isinstance(self.text_tokenizer, EnglishPhonemesTokenizer):
                preprocess_text_as_tts_input = self.text_tokenizer.g2p.text_preprocessing_func(raw_text)
            else:
                preprocess_text_as_tts_input = self.text_tokenizer.text_preprocessing_func(raw_text)

            lm_tokens_as_ids = self.lm_model_tokenizer.encode(preprocess_text_as_tts_input, add_special_tokens=False)

            if self.text_tokenizer.pad_with_space:
                lm_tokens_as_ids = [space_value] + lm_tokens_as_ids + [space_value]

            self.id2lm_tokens[i] = lm_tokens_as_ids

    def add_lm_tokens(self, **kwargs):
        lm_model = kwargs.pop('lm_model')

        if lm_model == "albert":
            self._albert()
        else:
            raise NotImplementedError(
                f"{lm_model} lm model is not supported. Only albert is supported at this moment."
            )

    def __getitem__(self, index):
        (
            audio,
            audio_length,
            text,
            text_length,
            log_mel,
            log_mel_length,
            durations,
            duration_prior,
            pitch,
            pitch_length,
            energy,
            energy_length,
        ) = super().__getitem__(index)

        lm_tokens = None
        if LMTokens in self.sup_data_types_set:
            lm_tokens = torch.tensor(self.id2lm_tokens[index]).long()

        return (
            audio,
            audio_length,
            text,
            text_length,
            log_mel,
            log_mel_length,
            durations,
            duration_prior,
            pitch,
            pitch_length,
            energy,
            energy_length,
            lm_tokens,
        )

    def _collate_fn(self, batch):
        batch = list(zip(*batch))
        data_dict = self.general_collate_fn(list(zip(*batch[:12])))
        lm_tokens_list = batch[12]

        if LMTokens in self.sup_data_types_set:
            lm_tokens = torch.full(
                (len(lm_tokens_list), max([lm_tokens.shape[0] for lm_tokens in lm_tokens_list])),
                fill_value=self.lm_padding_value,
            )
            for i, lm_tokens_i in enumerate(lm_tokens_list):
                lm_tokens[i, : lm_tokens_i.shape[0]] = lm_tokens_i

            data_dict[LMTokens.name] = lm_tokens

        joined_data = self.join_data(data_dict)
        return joined_data
