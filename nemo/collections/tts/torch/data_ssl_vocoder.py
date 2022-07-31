import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import hydra.utils
import librosa
import numpy as np
import torch
from omegaconf import open_dict
from tqdm import tqdm

from nemo.collections.asr.models import ssl_models
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.tts.models import ssl_tts
from nemo.collections.tts.torch.helpers import get_base_dir
from nemo.core.classes import Dataset
from nemo.utils import logging


class SSLVocoderDataset(Dataset):
    def __init__(
        self,
        manifest_filepath: Union[str, Path, List[str], List[Path]],
        sample_rate: int,
        ssl_model_type: str,
        ssl_model_ckpt_path: Union[str, Path],
        ssl_content_emb_type: str,
        n_segments: Optional[int] = None,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        ignore_file: Optional[Union[str, Path]] = None,
        trim: Optional[bool] = False,
    ):
        """Dataset which can be used for training and fine-tuning vocoder with pre-computed mel-spectrograms.
        Args:
            manifest_filepath (Union[str, Path, List[str], List[Path]]): Path(s) to the .json manifests containing
            information on the dataset. Each line in the .json file should be valid json. Note: the .json file itself
            is not valid json. Each line should contain the following:
                "audio_filepath": <PATH_TO_WAV>,
                "duration": <Duration of audio clip in seconds> (Optional),
                "mel_filepath": <PATH_TO_LOG_MEL> (Optional, can be in .npy (numpy.save) or .pt (torch.save) format)
            sample_rate (int): The sample rate of the audio. Or the sample rate that we will resample all files to.
            n_segments (int): The length of audio in samples to load. For example, given a sample rate of 16kHz, and
                n_segments=16000, a random 1-second section of audio from the clip will be loaded. The section will
                be randomly sampled everytime the audio is batched. Can be set to None to load the entire audio.
                Must be specified if load_precomputed_mel is True.
            max_duration (Optional[float]): Max duration of audio clips in seconds. All samples exceeding this will be
                pruned prior to training. Note: Requires "duration" to be set in the manifest file. It does not load
                audio to compute duration. Defaults to None which does not prune.
            min_duration (Optional[float]): Min duration of audio clips in seconds. All samples lower than this will be
                pruned prior to training. Note: Requires "duration" to be set in the manifest file. It does not load
                audio to compute duration. Defaults to None which does not prune.
            trim (bool): Whether to apply librosa.effects.trim to the audio file. Defaults to False.
        """
        super().__init__()

        assert ssl_model_type in ["conformer", "conformer_multitask"]
        self.ssl_model_type = ssl_model_type

        assert ssl_content_emb_type in ["probs", "embedding", "log_probs"]
        self.ssl_content_emb_type = ssl_content_emb_type
        # Initialize and read manifest file(s), filter out data by duration and ignore_file
        if isinstance(manifest_filepath, str):
            manifest_filepath = [manifest_filepath]
        self.manifest_filepath = manifest_filepath

        data = []
        total_duration = 0
        for manifest_file in self.manifest_filepath:
            with open(Path(manifest_file).expanduser(), 'r') as f:
                logging.info(f"Loading dataset from {manifest_file}.")
                for line in tqdm(f):
                    item = json.loads(line)

                    file_info = {
                        "audio_filepath": item["audio_filepath"],
                        "duration": item["duration"] if "duration" in item else None,
                    }

                    data.append(file_info)

                    if file_info["duration"] is None:
                        logging.info(
                            "Not all audio files have duration information. Duration logging will be disabled."
                        )
                        total_duration = None

                    if total_duration is not None:
                        total_duration += item["duration"]

        logging.info(f"Loaded dataset with {len(data)} files.")
        if total_duration is not None:
            logging.info(f"Dataset contains {total_duration / 3600:.2f} hours.")

        self.data = SSLVocoderDataset.filter_files(data, ignore_file, min_duration, max_duration, total_duration)
        self.base_data_dir = get_base_dir([item["audio_filepath"] for item in self.data])

        # Initialize audio and mel related parameters
        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.n_segments = n_segments
        self.trim = trim

        if ssl_model_type == "conformer":
            self.ssl_model = ssl_models.SpeechEncDecSelfSupervisedModel.from_pretrained(
                model_name='ssl_en_conformer_large'
            ).cpu()

        elif ssl_model_type == "conformer_multitask":
            self.ssl_model = ssl_tts.SSLDisentangler.load_from_checkpoint(ssl_model_ckpt_path).cpu()

        with open_dict(self.ssl_model.cfg):
            self.ssl_model.cfg.preprocessor.exact_pad = True
        self.ssl_model.preprocessor = hydra.utils.instantiate(self.ssl_model.cfg.preprocessor)

        self.ssl_model.eval()

        ssl_cfg = self.ssl_model.cfg
        ssl_sample_rate = ssl_cfg.preprocessor.sample_rate
        self.ssl_sample_rate = ssl_sample_rate
        ssl_window_stride_seconds = ssl_cfg.preprocessor.window_stride
        downsampling_rate_wav_to_mel = int(ssl_window_stride_seconds * ssl_sample_rate)  # 160
        downsampling_rate_mel_to_ssl = int(ssl_cfg.encoder.subsampling_factor)  # 4
        self.pad_multiple = downsampling_rate_wav_to_mel * downsampling_rate_mel_to_ssl

    def _collate_fn(self, batch):
        return torch.utils.data.dataloader.default_collate(batch)

    def filter_files(data, ignore_file, min_duration, max_duration, total_duration):
        if ignore_file:
            logging.info(f"Using {ignore_file} to prune dataset.")
            with open(Path(ignore_file).expanduser(), "rb") as f:
                wavs_to_ignore = set(pickle.load(f))

        filtered_data: List[Dict] = []
        pruned_duration = 0 if total_duration is not None else None
        pruned_items = 0
        for item in data:
            audio_path = item['audio_filepath']

            # Prune data according to min/max_duration & the ignore file
            if total_duration is not None:
                if (min_duration and item["duration"] < min_duration) or (
                    max_duration and item["duration"] > max_duration
                ):
                    pruned_duration += item["duration"]
                    pruned_items += 1
                    continue

            if ignore_file and (audio_path in wavs_to_ignore):
                pruned_items += 1
                pruned_duration += item["duration"]
                wavs_to_ignore.remove(audio_path)
                continue

            filtered_data.append(item)

        logging.info(f"Pruned {pruned_items} files. Final dataset contains {len(filtered_data)} files")
        if pruned_duration is not None:
            logging.info(
                f"Pruned {pruned_duration / 3600:.2f} hours. Final dataset contains "
                f"{(total_duration - pruned_duration) / 3600:.2f} hours."
            )

        return filtered_data

    def __getitem__(self, index):
        sample = self.data[index]

        features = AudioSegment.segment_from_file(
            sample["audio_filepath"],
            target_sr=self.sample_rate,
            n_segments=self.n_segments if self.n_segments is not None else -1,
            trim=self.trim,
        )
        audio_samples = features.samples
        audio_samples_forssl = librosa.core.resample(
            audio_samples, orig_sr=self.sample_rate, target_sr=self.ssl_sample_rate
        )
        audio_samples_forssl = torch.tensor(audio_samples_forssl)

        audio_ssl, audio_ssl_length = audio_samples_forssl, torch.tensor(audio_samples_forssl.shape[0]).long()
        audio, audio_length = torch.tensor(audio_samples), torch.tensor(audio_samples.shape[0]).long()

        # pad audio to a multiple of self.pad_multiple
        if audio_ssl.shape[0] % self.pad_multiple != 0:
            audio_ssl = torch.cat(
                [audio_ssl, torch.zeros(self.pad_multiple - audio_ssl.shape[0] % self.pad_multiple, dtype=torch.float)]
            )
            audio_ssl_length = torch.tensor(audio_ssl.shape[0]).long()

            target_audio_length = int(audio_ssl.shape[0] * (self.sample_rate / self.ssl_sample_rate))
            audio = torch.cat([audio, torch.zeros(target_audio_length - audio.shape[0], dtype=torch.float)])
            audio_length = torch.tensor(audio.shape[0]).long()

        if self.ssl_model_type == "conformer":
            with torch.no_grad():
                processed_signal, processed_signal_length = self.ssl_model.preprocessor(
                    input_signal=audio_ssl[None], length=audio_ssl_length[None],
                )
                encoded, encoded_len = self.ssl_model.encoder.forward_for_export(
                    audio_signal=processed_signal, length=processed_signal_length
                )
                encoded = encoded[0].detach()
                encoded_len = encoded_len[0].detach()
                encoded = encoded[:, : encoded_len.item()]

            return audio, audio_length, encoded, encoded_len

        elif self.ssl_model_type == "conformer_multitask":
            with torch.no_grad():
                (
                    _,
                    speaker_embedding_normalized,
                    content_embedding,
                    content_log_probs,
                    encoded_len,
                ) = self.ssl_model.forward_for_export(
                    input_signal=audio_ssl[None], input_signal_length=audio_ssl_length[None]
                )
                speaker_embedding_normalized = speaker_embedding_normalized[0].detach()
                content_embedding = content_embedding[0].detach()
                content_log_probs = content_log_probs[:, 0, :].detach()  # (content lob prob is (t, b, c))
                encoded_len = encoded_len[0].detach()
                content_embedding = content_embedding[: encoded_len.item()]
                content_embedding = content_embedding.t()
                content_log_probs = content_log_probs[: encoded_len.item()]
                content_log_probs = content_log_probs.t()
                content_probs = torch.exp(content_log_probs)
                if self.ssl_content_emb_type == "probs":
                    final_content_embedding = content_probs
                elif self.ssl_content_emb_type == "embedding":
                    final_content_embedding = content_embedding
                elif self.ssl_content_emb_type == "log_probs":
                    final_content_embedding = content_log_probs

            return audio, audio_length, final_content_embedding, encoded_len, speaker_embedding_normalized

    def __len__(self):
        return len(self.data)
