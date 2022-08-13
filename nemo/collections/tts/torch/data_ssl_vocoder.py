import json
import os
import pickle
import shutil
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
        pitch_conditioning: Optional[bool] = False,
        pitch_mean: Optional[float] = None,
        pitch_std: Optional[float] = None,
        pitch_normalization: Optional[str] = None,
        sup_data_dir: Optional[Union[str, Path]] = None,
        data_caching: Optional[bool] = True,
        normalize_content: Optional[bool] = True,
        speaker_stats_pitch_fp: Optional[Union[str, Path]] = None,
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
                    if "speaker" not in item:
                        item["speaker"] = 0
                    file_info = {
                        "audio_filepath": item["audio_filepath"],
                        "duration": item["duration"] if "duration" in item else None,
                        "speaker": item["speaker"],
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
        assert self.n_segments % self.pad_multiple == 0, "suggested n_segments: {}".format(
            self.pad_multiple * (self.n_segments // self.pad_multiple)
        )
        self.n_segments_at_target_sr = n_segments * self.sample_rate / self.ssl_sample_rate
        assert self.n_segments_at_target_sr.is_integer()
        self.n_segments_at_target_sr = int(self.n_segments_at_target_sr)
        self.ssl_frame_length = int(0.025 * ssl_sample_rate)
        self.ssl_hop_length = int(0.01 * ssl_sample_rate)
        self.pitch_conditioning = pitch_conditioning
        self.pitch_mean = pitch_mean
        self.pitch_std = pitch_std
        self.pitch_normalization = pitch_normalization
        self.data_caching = data_caching

        if sup_data_dir is None:
            sup_data_dir = os.path.join(self.base_data_dir, "sup_data")

        self.sup_data_dir = sup_data_dir
        if not os.path.exists(self.sup_data_dir):
            os.makedirs(self.sup_data_dir)

        self.normalize_content = normalize_content

        if self.pitch_normalization == "speaker_wise":
            self.speaker_stats = {}
            with open(speaker_stats_pitch_fp, "r") as f:
                speaker_stats_raw = json.load(f)
                for key in speaker_stats_raw:
                    self.speaker_stats[int(key)] = speaker_stats_raw[key]

    def _collate_fn(self, batch):
        final_batch = {}
        for row in batch:
            for key in row:
                if key not in final_batch:
                    final_batch[key] = []
                final_batch[key].append(row[key])

        for key in final_batch:
            if final_batch[key][0] is None:
                final_batch[key] = None
            else:
                final_batch[key] = torch.stack(final_batch[key])

        return final_batch

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

    def get_pitch_contour(self, wav, wav_text_id):
        pitch_contour_fn = f"pitch_contour_{wav_text_id}.pt"
        pitch_contour_fp = os.path.join(self.sup_data_dir, pitch_contour_fn)
        if os.path.exists(pitch_contour_fp) and self.data_caching:
            return torch.load(pitch_contour_fp)
        else:
            f0, _, _ = librosa.pyin(
                wav.numpy(),
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                frame_length=self.ssl_hop_length * 16,
                hop_length=self.ssl_hop_length * 4,
                sr=self.ssl_sample_rate,
                center=True,
                fill_na=0.0,
            )
            pitch_contour = torch.tensor(f0, dtype=torch.float32)
            torch.save(pitch_contour, pitch_contour_fp)
            return pitch_contour

    def get_ssl_features(self, audio_ssl, audio_ssl_length, wav_text_id):
        content_emb_fn = f"{self.ssl_content_emb_type}_content_embedding_{wav_text_id}.pt"
        speaker_emb_fn = f"speaker_embedding_{wav_text_id}.pt"
        content_emb_fp = os.path.join(self.sup_data_dir, content_emb_fn)
        speaker_emb_fp = os.path.join(self.sup_data_dir, speaker_emb_fn)
        if os.path.exists(content_emb_fp) and self.data_caching:
            content_embedding = torch.load(content_emb_fp)
            if os.path.exists(speaker_emb_fp):
                speaker_embedding = torch.load(speaker_emb_fp)
            else:
                speaker_embedding = None
                assert self.ssl_model_type == "conformer"
            encoded_len = torch.tensor(content_embedding.shape[1]).long()
            return content_embedding, speaker_embedding, encoded_len
        else:
            if self.ssl_model_type == "conformer_multitask":
                with torch.no_grad():
                    (
                        _,
                        speaker_embedding_normalized,
                        content_embedding,
                        content_log_probs,
                        encoded_len,
                    ) = self.ssl_model.forward_for_export(
                        input_signal=audio_ssl[None],
                        input_signal_length=audio_ssl_length[None],
                        normalize_content=self.normalize_content,
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

                    torch.save(final_content_embedding, content_emb_fp)
                    torch.save(speaker_embedding_normalized, speaker_emb_fp)

                    return final_content_embedding, speaker_embedding_normalized, encoded_len

            elif self.ssl_model_type == "conformer":
                with torch.no_grad():
                    processed_signal, processed_signal_length = self.ssl_model.preprocessor(
                        input_signal=audio_ssl[None], length=audio_ssl_length[None],
                    )
                    encoded, encoded_len = self.ssl_model.encoder.forward_for_export(
                        audio_signal=processed_signal, length=processed_signal_length
                    )
                    encoded = encoded[0].detach()
                    encoded_len = encoded_len[0].detach()
                    torch.save(encoded, content_emb_fp)

                    return encoded, None, encoded_len

    def _segment_item(self, item):
        """
        item is the dict returned by __getitem__
        """
        segment_len = self.n_segments
        encoded_segment_len = segment_len // self.pad_multiple

        assert encoded_segment_len < item['encoded_len'], "{} < {}, {}".format(
            encoded_segment_len, item['encoded_len'], len(item['audio'])
        )
        encoded_sidx = np.random.randint(0, item['encoded_len'] - encoded_segment_len)
        encoded_eidx = encoded_sidx + encoded_segment_len

        audio_sidx = encoded_sidx * self.pad_multiple * self.sample_rate // self.ssl_sample_rate
        audio_eidx = audio_sidx + self.n_segments_at_target_sr

        audio_segment = item['audio'][audio_sidx:audio_eidx]
        encoded_segment = item['content_embedding'][:, encoded_sidx:encoded_eidx]

        if item['pitch_contour'] is not None:
            segment_pitch = item['pitch_contour'][encoded_sidx:encoded_eidx]
        else:
            segment_pitch = None

        new_item = {
            'audio': audio_segment,
            'content_embedding': encoded_segment,
            'audio_len': torch.tensor(self.n_segments_at_target_sr).long(),
            'encoded_len': torch.tensor(encoded_segment_len).long(),
            'pitch_contour': segment_pitch,
        }

        # add remaining fields
        for key in item:
            if key not in new_item:
                new_item[key] = item[key]

        return new_item

    def _get_wav_from_filepath(self, audio_filepath):
        features = AudioSegment.segment_from_file(
            audio_filepath, target_sr=self.sample_rate, n_segments=-1, trim=self.trim,
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

        return audio_ssl, audio_ssl_length, audio, audio_length

    def _is_valid_pitch_contour(self, pitch_contour):
        if pitch_contour.dtype != torch.float32:
            return False
        return True

    def __getitem__(self, index):
        sample = self.data[index]
        rel_audio_path = Path(sample["audio_filepath"]).relative_to(self.base_data_dir).with_suffix("")
        rel_audio_path_as_text_id = str(rel_audio_path).replace("/", "_")
        speaker = torch.tensor(sample["speaker"]).long()
        audio_ssl, audio_ssl_length, audio, audio_length = self._get_wav_from_filepath(sample["audio_filepath"])

        pitch_contour = None
        if self.pitch_conditioning:
            pitch_contour = self.get_pitch_contour(audio_ssl, rel_audio_path_as_text_id)
            # print(rel_audio_path_as_text_id, pitch_contour)
        content_embedding, speaker_embedding, encoded_len = self.get_ssl_features(
            audio_ssl, audio_ssl_length, rel_audio_path_as_text_id
        )

        if pitch_contour is not None:
            pitch_contour = pitch_contour[: encoded_len.item()]
            assert pitch_contour.shape[0] == content_embedding.shape[1] == encoded_len.item()

            if self.pitch_normalization in ["speaker_wise", "global"]:
                if self.pitch_normalization == "speaker_wise":
                    mean = self.speaker_stats[sample["speaker"]]["pitch_mean"]
                    std = self.speaker_stats[sample["speaker"]]["pitch_std"]
                    if np.isnan(mean) or np.isnan(std) or mean == 0 or std == 0:
                        logging.warning("NaN found in pitch mean/std for speaker {}".format(sample["speaker"]))
                        mean = self.pitch_mean
                        std = self.pitch_std
                elif self.pitch_normalization == "global":
                    mean = self.pitch_mean
                    std = self.pitch_std

                pitch_contour = pitch_contour - mean
                pitch_contour[pitch_contour == -mean] = 0.0
                pitch_contour = pitch_contour / std

            if not self._is_valid_pitch_contour(pitch_contour):
                print("invalid pitch contour for", sample["audio_filepath"])
                print("Setting pitch contour to 0")
                pitch_contour = torch.zeros(encoded_len.item())

        return self._segment_item(
            {
                'audio': audio,
                'audio_len': audio_length,
                'content_embedding': content_embedding,
                'speaker_embedding': speaker_embedding,
                'encoded_len': encoded_len,
                'pitch_contour': pitch_contour,
                'speaker': speaker,
            }
        )

    def __len__(self):
        return len(self.data)
