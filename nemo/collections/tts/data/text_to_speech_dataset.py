# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import webdataset as wd

from nemo.collections.asr.data.audio_to_text import expand_sharded_filepaths
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer
from nemo.collections.tts.parts.preprocessing.feature_processors import FeatureProcessor
from nemo.collections.tts.parts.preprocessing.features import FeatureReader
from nemo.collections.tts.parts.utils.tarred_dataset_utils import (
    create_tarred_dataset,
    process_tarred_manifest,
    FileFilterIterator,
    TarredMetadata
)
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    filter_dataset_by_duration,
    get_audio_filepaths,
    get_weighted_sampler,
    stack_tensors,
)
from nemo.core.classes import Dataset
from nemo.utils import logging
from nemo.utils.decorators import experimental
from torch.utils.data import IterableDataset, Sampler


@dataclass
class DatasetMeta:
    manifest_path: Path
    audio_dir: Path
    feature_dir: Path
    sample_weight: float = 1.0


@dataclass
class DatasetSample:
    dataset_name: str
    manifest_entry: Dict[str, Any]
    audio_dir: Path
    feature_dir: Path
    text: str
    speaker: str
    speaker_index: int = None


def text_to_speech_collate_fn(
    batch: List[dict],
    feature_readers: List[FeatureReader],
    feature_processors: List[FeatureProcessor],
    text_pad_value: int,
    include_speaker: bool
):
    dataset_name_list = []
    audio_filepath_list = []
    token_list = []
    token_len_list = []
    speaker_list = []

    for example in batch:
        dataset_name_list.append(example["dataset_name"])
        audio_filepath_list.append(example["audio_filepath"])

        token_list.append(example["tokens"])
        token_len_list.append(example["text_len"])

        if include_speaker:
            speaker_list.append(example["speaker_index"])

    batch_token_len = torch.IntTensor(token_len_list)
    token_max_len = int(batch_token_len.max().item())
    batch_tokens = stack_tensors(token_list, max_lens=[token_max_len], pad_value=text_pad_value)

    batch_dict = {
        "dataset_names": dataset_name_list,
        "audio_filepaths": audio_filepath_list,
        "text": batch_tokens,
        "text_lens": batch_token_len,
    }

    if include_speaker:
        batch_dict["speaker_id"] = torch.IntTensor(speaker_list)

    for feature_reader in feature_readers:
        feature_dict = feature_reader.collate_fn(batch)
        batch_dict.update(feature_dict)

    for feature_processor in feature_processors:
        processor_dict = feature_processor.collate_fn(batch)
        batch_dict.update(processor_dict)

    return batch_dict


@experimental
class TextToSpeechDataset(Dataset):
    """
    Class for processing and loading text to speech training examples.

    Args:
        dataset_meta: Dict of dataset names (string) to dataset metadata.
        text_tokenizer: Tokenizer to apply to the text field.
        weighted_sampling_steps_per_epoch: Optional int, If provided, then data will be sampled (with replacement) based on
            the sample weights provided in the dataset metadata. If None, then sample weights will be ignored.
        speaker_path: Optional, path to JSON file with speaker indices, for multi-speaker training. Can be created with
            scripts.dataset_processing.tts.create_speaker_map.py
        featurizers: Optional, list of featurizers to load feature data from. Should be the same config provided
            when running scripts.dataset_processing.tts.compute_features.py before training.
        feature_processors: Optional, list of feature processors to run on training examples.
        min_duration: Optional float, if provided audio files in the training manifest shorter than 'min_duration'
            will be ignored.
        max_duration: Optional float, if provided audio files in the training manifest longer than 'max_duration'
            will be ignored.
        volume_norm: Whether to apply volume normalization to loaded audio.
    """

    def __init__(
        self,
        dataset_meta: Dict,
        text_tokenizer: BaseTokenizer,
        weighted_sampling_steps_per_epoch: Optional[int] = None,
        speaker_path: Optional[Path] = None,
        feature_readers: Optional[Dict[str, FeatureReader]] = None,
        feature_processors: Optional[Dict[str, FeatureProcessor]] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        volume_norm: bool = True
    ):
        super().__init__()

        self.text_tokenizer = text_tokenizer
        self.weighted_sampling_steps_per_epoch = weighted_sampling_steps_per_epoch
        self.volume_norm = volume_norm

        if speaker_path:
            self.include_speaker = True
            with open(speaker_path, 'r', encoding="utf-8") as speaker_f:
                speaker_index_map = json.load(speaker_f)
        else:
            self.include_speaker = False
            speaker_index_map = None

        if feature_readers:
            logging.info(f"Found feature readers {feature_readers.keys()}")
            self.feature_readers = list(feature_readers.values())
        else:
            self.feature_readers = []

        if feature_processors:
            logging.info(f"Found feature processors {feature_processors.keys()}")
            self.feature_processors = list(feature_processors.values())
        else:
            self.feature_processors = []

        self.data_samples = []
        self.sample_weights = []
        for dataset_name, dataset_info in dataset_meta.items():
            dataset = DatasetMeta(**dataset_info)
            samples, weights = self._preprocess_manifest(
                dataset_name=dataset_name,
                dataset=dataset,
                min_duration=min_duration,
                max_duration=max_duration,
                speaker_index_map=speaker_index_map,
            )
            self.data_samples += samples
            self.sample_weights += weights

    def get_sampler(self, batch_size: int, world_size: int) -> Optional[Sampler]:
        if not self.weighted_sampling_steps_per_epoch:
            return None

        sampler = get_weighted_sampler(
            sample_weights=self.sample_weights,
            batch_size=batch_size,
            num_steps=self.weighted_sampling_steps_per_epoch,
            world_size=world_size
        )
        return sampler

    def _preprocess_manifest(
        self,
        dataset_name: str,
        dataset: DatasetMeta,
        min_duration: float,
        max_duration: float,
        speaker_index_map: Dict[str, int],
    ):
        entries = read_manifest(dataset.manifest_path)
        filtered_entries, total_hours, filtered_hours = filter_dataset_by_duration(
            entries=entries, min_duration=min_duration, max_duration=max_duration
        )

        logging.info(dataset_name)
        logging.info(f"Original # of files: {len(entries)}")
        logging.info(f"Filtered # of files: {len(filtered_entries)}")
        logging.info(f"Original duration: {total_hours:.2f} hours")
        logging.info(f"Filtered duration: {filtered_hours:.2f} hours")

        samples = []
        sample_weights = []
        for entry in filtered_entries:

            if "normalized_text" in entry:
                text = entry["normalized_text"]
            else:
                text = entry["text"]

            if self.include_speaker:
                speaker = entry["speaker"]
                speaker_index = speaker_index_map[speaker]
            else:
                speaker = None
                speaker_index = 0

            sample = DatasetSample(
                dataset_name=dataset_name,
                manifest_entry=entry,
                audio_dir=Path(dataset.audio_dir),
                feature_dir=Path(dataset.feature_dir),
                text=text,
                speaker=speaker,
                speaker_index=speaker_index,
            )
            samples.append(sample)
            sample_weights.append(dataset.sample_weight)

        return samples, sample_weights

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        data = self.data_samples[index]

        _, audio_filepath_rel = get_audio_filepaths(manifest_entry=data.manifest_entry, audio_dir=data.audio_dir)

        tokens = self.text_tokenizer(data.text)
        tokens = torch.tensor(tokens, dtype=torch.int32)
        text_len = tokens.shape[0]

        example = {
            "dataset_name": data.dataset_name,
            "audio_filepath": audio_filepath_rel,
            "tokens": tokens,
            "text_len": text_len,
        }

        if data.speaker is not None:
            example["speaker"] = data.speaker
            example["speaker_index"] = data.speaker_index

        for feature_reader in self.feature_readers:
            feature_dict = feature_reader.load(
                manifest_entry=data.manifest_entry, audio_dir=data.audio_dir, feature_dir=data.feature_dir
            )
            example.update(feature_dict)

        for processor in self.feature_processors:
            processor.process(example)

        return example

    def collate_fn(self, batch: List[dict]):
        return text_to_speech_collate_fn(
            batch,
            feature_readers=self.feature_readers,
            feature_processors=self.feature_processors,
            text_pad_value=self.text_tokenizer.pad,
            include_speaker=self.include_speaker
        )


class TarredTextToSpeechDataset(IterableDataset):
    """
    """
    def __init__(
        self,
        dataset_meta: Dict,
        text_tokenizer: BaseTokenizer = None,
        sample_type: str = "concat",
        sample_args: Optional[Dict] = None,
        speaker_path: Optional[Path] = None,
        feature_readers: Optional[Dict[str, FeatureReader]] = None,
        feature_processors: Optional[Dict[str, FeatureProcessor]] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        volume_norm: bool = True,
        shuffle_n: int = 0,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
    ):
        super().__init__()
        self.text_tokenizer = text_tokenizer
        self.volume_norm = volume_norm

        if speaker_path:
            self.include_speaker = True
            with open(speaker_path, 'r', encoding="utf-8") as speaker_f:
                self.speaker_index_map = json.load(speaker_f)
        else:
            self.include_speaker = False
            self.speaker_index_map = None

        if feature_readers:
            logging.info(f"Found feature readers for {feature_readers.keys()}")
            self.feature_readers = list(feature_readers.values())
        else:
            self.feature_readers = []

        if feature_processors:
            logging.info(f"Found feature processors {feature_processors.keys()}")
            self.feature_processors = list(feature_processors.values())
        else:
            self.feature_processors = []

        web_datasets = []
        dataset_lengths = []
        self.file_to_sample_map = {}
        for dataset_name, dataset_info in dataset_meta.items():
            dataset_meta = TarredMetadata(**dataset_info)

            dataset_entries = read_manifest(dataset_meta.manifest_path)
            sample_map, unfiltered_file_count, unfiltered_hours, filtered_hours = process_tarred_manifest(
                dataset_name=dataset_name, entries=dataset_entries, min_duration=min_duration, max_duration=max_duration
            )
            self.file_to_sample_map.update(sample_map)

            dataset_length = len(sample_map)
            if dataset_length == 0:
                raise ValueError(f"Found empty dataset {dataset_name} after filtering.")

            logging.info(dataset_name)
            logging.info(f"Original # of files: {len(dataset_entries)}")
            logging.info(f"Filtered # of files: {dataset_length}")
            logging.info(f"Original duration: {unfiltered_hours:.2f} hours")
            logging.info(f"Filtered duration: {unfiltered_hours:.2f} hours")

            web_dataset = self._create_web_dataset(
                tar_filepath=dataset_meta.tar_filepath,
                shuffle_n=shuffle_n,
                shard_strategy=shard_strategy,
                global_rank=global_rank,
                world_size=world_size
            )
            if web_dataset is not None:
                web_datasets.append(web_dataset)
                dataset_lengths.append(dataset_length)

        self.dataset = create_tarred_dataset(
            datasets=web_datasets,
            dataset_lengths=dataset_lengths,
            sample_type=sample_type,
            sample_args=sample_args
        )

        if len(self.dataset) == 0:
            raise ValueError(f"Final dataset is empty.")

    def _create_web_dataset(
        self, tar_filepath: str, shuffle_n: int, shard_strategy: str, global_rank: int, world_size: int
    ):
        tar_filepaths = expand_sharded_filepaths(
            sharded_filepaths=tar_filepath,
            global_rank=global_rank,
            world_size=world_size,
            shard_strategy=shard_strategy,
        )
        logging.info(f"Expanded {tar_filepath} to {len(tar_filepaths)} files")

        if len(tar_filepaths) == 0:
            # When using scatter shard_strategy, some workers might have no shards for a dataset
            return None

        dataset = wd.WebDataset(tar_filepaths, nodesplitter=None)

        key_names = ["key", "audio"]
        rename_args = {
            "key": "__key__"
        }
        for feature_reader in self.feature_readers:
            key_names.append(feature_reader.feature_name)
            rename_args[feature_reader.feature_name] = feature_reader.get_tarred_suffixes()

        file_ids = set(self.file_to_sample_map.keys())
        dataset = (
            dataset.rename(**rename_args)
            .pipe(lambda iterator: FileFilterIterator(iterator=iterator, file_ids=file_ids))
        )

        if shuffle_n > 0:
            logging.info(f"Using shuffle buffer of size {shuffle_n}")
            dataset = dataset.shuffle(size=shuffle_n, initial=shuffle_n)
        else:
            logging.info("WebDataset will not shuffle data. Consider setting shuffle_n > 0.")

        dataset = dataset.map(self._build_sample)

        return dataset

    def _build_sample(self, inputs):
        file_id = inputs["key"]
        data = self.file_to_sample_map[file_id]
        entry = data.manifest_entry

        if "normalized_text" in entry:
            text = entry["normalized_text"]
        else:
            text = entry["text"]

        tokens = self.text_tokenizer(text)
        tokens = torch.tensor(tokens, dtype=torch.int32)
        text_len = tokens.shape[0]

        audio_filepath = Path(data.manifest_entry["audio_filepath"])
        example = {
            "dataset_name": data.dataset_name,
            "audio_filepath": audio_filepath,
            "tokens": tokens,
            "text_len": text_len,
        }

        if self.include_speaker:
            speaker = entry["speaker"]
            example["speaker"] = speaker
            example["speaker_index"] = self.speaker_index_map[speaker]

        for feature_reader in self.feature_readers:
            feature_bytes = inputs[feature_reader.feature_name]
            feature_bytes_io = io.BytesIO(feature_bytes)
            feature_dict = feature_reader.deserialize(feature_bytes_io)
            example.update(feature_dict)

        for processor in self.feature_processors:
            processor.process(example)

        return example

    def get_sampler(self, batch_size: int, world_size: int) -> Optional[Sampler]:
        return None

    def collate_fn(self, batch):
        return text_to_speech_collate_fn(
            batch,
            feature_readers=self.feature_readers,
            feature_processors=self.feature_processors,
            text_pad_value=self.text_tokenizer.pad,
            include_speaker=self.include_speaker,
        )

    def __iter__(self):
        return self.dataset.__iter__()

    def __len__(self):
        return len(self.dataset)