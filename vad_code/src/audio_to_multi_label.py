# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import webdataset as wd
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_text_dataset import convert_to_config_list, get_chain_dataset
from nemo.collections.asr.data.audio_to_text import expand_audio_filepaths
from nemo.collections.asr.data.audio_to_label import _speech_collate_fn, count_occurence
from nemo.collections.asr.parts.preprocessing.segment import available_formats as valid_sf_formats
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common.parts.preprocessing import collections
from nemo.core.classes import Dataset, IterableDataset
from nemo.utils import logging

__all__ = ["AudioToMultiLabelDataset"]

# List of valid file formats (prioritized by order of importance)
VALID_FILE_FORMATS = ';'.join(['wav', 'mp3', 'flac'] + [fmt.lower() for fmt in valid_sf_formats.keys()])


class AudioToMultiLabelDataset(Dataset):
    def __init__(
        self,
        *,
        manifest_filepath: Union[str, List[str]],
        sample_rate: int,
        labels: Optional[List[str]] = None,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim_silence: bool = False,
        is_regression_task: bool = False,
        delimiter: str = " ",
    ):
        super().__init__()
        if isinstance(manifest_filepath, str):
            manifest_filepath = manifest_filepath.split(',')

        self.collection = collections.ASRSpeechLabel(
            manifests_files=manifest_filepath,
            min_duration=min_duration,
            max_duration=max_duration,
            is_regression_task=is_regression_task,
        )

        self.collection = self.filter_audio_files(self.collection)

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim_silence
        self.is_regression_task = is_regression_task
        self.delimiter = delimiter

        if not is_regression_task:
            self.labels = labels if labels else self._get_label_set()
            self.num_classes = len(self.labels) if self.labels is not None else 1
            self.label2id, self.id2label = {}, {}
            for label_id, label in enumerate(self.labels):
                self.label2id[label] = label_id
                self.id2label[label_id] = label
            for idx in range(len(self.labels[:5])):
                logging.debug(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))
        else:
            self.labels = []
            self.num_classes = 1

    def _get_label_set(self):
        labels = []
        for sample in self.collection:
            label_str = sample.label
            if label_str:
                label_str_list = label_str.split(self.delimiter) if self.delimiter else label_str.split()
                labels.extend(label_str_list)
        return sorted(set(labels))

    def _label_str_to_tensor(self, label_str: str):
        labels = label_str.split(self.delimiter) if self.delimiter else label_str.split()

        if self.is_regression_task:
            labels = [float(s) for s in labels]
            labels = torch.tensor(labels).float()
        else:
            labels = [self.label2id[s] for s in labels]
            labels = torch.tensor(labels).long()
        return labels

    def filter_audio_files(self, data_list):
        results = []
        cnt = 0
        duration = 0.0
        discarded = []
        for sample in data_list:
            if Path(sample.audio_file).is_file():
                results.append(sample)
                duration += sample.duration
            else:
                cnt += 1
                discarded.append(sample.audio_file)
        logging.info(f"{cnt} audio files were discarded since not found.")
        logging.info(discarded[:5])
        logging.info(f"Total duration after filtering: {duration / 3600: .2f} hours.")
        logging.info("--------------------------------")
        return results

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        sample = self.collection[index]

        offset = sample.offset

        if offset is None:
            offset = 0

        features = self.featurizer.process(sample.audio_file, offset=offset, duration=sample.duration, trim=self.trim)
        f, fl = features, torch.tensor(features.size(0)).long()

        t = self._label_str_to_tensor(sample.label)

        tl = torch.tensor(t.size(0)).long()

        return f, fl, t, tl

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=0)


class TarredAudioToMultiLabelDataset(IterableDataset):
    def __init__(
        self,
        *,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: Union[str, List[str]],
        sample_rate: int,
        labels: Optional[List[str]] = None,
        shuffle_n: int = 0,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim_silence: bool = False,
        is_regression_task: bool = False,
        delimiter: str = " ",
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
    ):
        super().__init__()
        if isinstance(manifest_filepath, str):
            manifest_filepath = manifest_filepath.split(',')

        self.collection = collections.ASRSpeechLabel(
            manifests_files=manifest_filepath,
            min_duration=min_duration,
            max_duration=max_duration,
            is_regression_task=is_regression_task,
            index_by_file_id=True,
        )
        self.file_occurence = count_occurence(self.collection.mapping)

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim_silence
        self.is_regression_task = is_regression_task
        self.delimiter = delimiter

        if not is_regression_task:
            self.labels = labels if labels else self._get_label_set()
            self.num_classes = len(self.labels) if self.labels is not None else 1
            self.label2id, self.id2label = {}, {}
            for label_id, label in enumerate(self.labels):
                self.label2id[label] = label_id
                self.id2label[label_id] = label
            for idx in range(len(self.labels[:5])):
                logging.debug(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))
        else:
            self.labels = []
            self.num_classes = 1

        audio_tar_filepaths = expand_audio_filepaths(
            audio_tar_filepaths=audio_tar_filepaths,
            shard_strategy=shard_strategy,
            world_size=world_size,
            global_rank=global_rank,
        )
        # Put together WebDataset
        self._dataset = wd.WebDataset(urls=audio_tar_filepaths, nodesplitter=None)

        if shuffle_n > 0:
            self._dataset = self._dataset.shuffle(shuffle_n)
        else:
            logging.info("WebDataset will not shuffle files within the tar files.")

        self._dataset = (
            self._dataset.rename(audio=VALID_FILE_FORMATS, key='__key__')
            .to_tuple('audio', 'key')
            .pipe(self._filter)
            .map(f=self._build_sample)
        )

    def _get_label_set(self):
        labels = []
        for sample in self.collection:
            label_str = sample.label
            if label_str:
                label_str_list = label_str.split(self.delimiter) if self.delimiter else label_str.split()
                labels.extend(label_str_list)
        return sorted(set(labels))

    def _label_str_to_tensor(self, label_str: str):
        labels = label_str.split(self.delimiter) if self.delimiter else label_str.split()

        if self.is_regression_task:
            labels = [float(s) for s in labels]
            labels = torch.tensor(labels).float()
        else:
            labels = [self.label2id[s] for s in labels]
            labels = torch.tensor(labels).long()
        return labels

    def _filter(self, iterator):
        """This function is used to remove samples that have been filtered out by ASRSpeechLabel already.
        Otherwise, we would get a KeyError as _build_sample attempts to find the manifest entry for a sample
        that was filtered out (e.g. for duration).
        Note that if using multi-GPU training, filtering may lead to an imbalance in samples in each shard,
        which may make your code hang as one process will finish before the other.
        """

        class TarredAudioFilter:
            def __init__(self, collection, file_occurence):
                self.iterator = iterator
                self.collection = collection
                self.file_occurence = file_occurence
                self._iterable = self._internal_generator()

            def __iter__(self):
                self._iterable = self._internal_generator()
                return self

            def __next__(self):
                try:
                    values = next(self._iterable)
                except StopIteration:
                    # reset generator
                    self._iterable = self._internal_generator()
                    values = next(self._iterable)

                return values

            def _internal_generator(self):
                """
                WebDataset requires an Iterator, but we require an iterable that yields 1-or-more
                values per value inside self.iterator.

                Therefore wrap the iterator with a generator function that will yield 1-or-more
                values per sample in the iterator.
                """
                for _, tup in enumerate(self.iterator):
                    audio_bytes, audio_filename = tup

                    file_id, _ = os.path.splitext(os.path.basename(audio_filename))
                    if audio_filename in self.file_occurence:
                        for j in range(0, self.file_occurence[file_id]):
                            if j == 0:
                                audio_filename = file_id
                            else:
                                audio_filename = file_id + "-sub" + str(j)
                            yield audio_bytes, audio_filename

        return TarredAudioFilter(self.collection, self.file_occurence)

    def _build_sample(self, tup):
        """Builds the training sample by combining the data from the WebDataset with the manifest info.
        """
        audio_bytes, audio_filename = tup
        # Grab manifest entry from self.collection
        file_id, _ = os.path.splitext(os.path.basename(audio_filename))

        manifest_idx = self.collection.mapping[file_id]
        manifest_entry = self.collection[manifest_idx]

        offset = manifest_entry.offset
        if offset is None:
            offset = 0

        # Convert audio bytes to IO stream for processing (for SoundFile to read)
        audio_filestream = io.BytesIO(audio_bytes)
        features = self.featurizer.process(
            audio_filestream, offset=offset, duration=manifest_entry.duration, trim=self.trim,
        )

        audio_filestream.close()

        # Audio features
        f, fl = features, torch.tensor(features.shape[0]).long()

        t = self._label_str_to_tensor(manifest_entry.label)

        tl = torch.tensor(t.size(0)).long()

        return f, fl, t, tl

    def __iter__(self):
        return self._dataset.__iter__()

    def __len__(self):
        return len(self.collection)

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=0)


def get_audio_multi_label_dataset(cfg: DictConfig) -> AudioToMultiLabelDataset:
    if "augmentor" in cfg:
        augmentor = process_augmentations(cfg.augmentor)
    else:
        augmentor = None

    dataset = AudioToMultiLabelDataset(
        manifest_filepath=cfg.get("manifest_filepath"),
        sample_rate=cfg.get("sample_rate"),
        labels=cfg.get("labels", None),
        int_values=cfg.get("int_values", False),
        augmentor=augmentor,
        min_duration=cfg.get("min_duration", None),
        max_duration=cfg.get("max_duration", None),
        trim_silence=cfg.get("trim_silence", False),
        is_regression_task=cfg.get("is_regression_task", False),
        delimiter=cfg.get("delimiter", None),
    )
    return dataset


def get_tarred_audio_multi_label_dataset(
    cfg: DictConfig, shuffle_n: int, global_rank: int, world_size: int
) -> TarredAudioToMultiLabelDataset:

    if "augmentor" in cfg:
        augmentor = process_augmentations(cfg.augmentor)
    else:
        augmentor = None

    tarred_audio_filepaths = cfg['tarred_audio_filepaths']
    manifest_filepaths = cfg['manifest_filepath']
    datasets = []
    tarred_audio_filepaths = convert_to_config_list(tarred_audio_filepaths)
    manifest_filepaths = convert_to_config_list(manifest_filepaths)

    bucketing_weights = cfg.get('bucketing_weights', None)  # For upsampling buckets
    if bucketing_weights:
        for idx, weight in enumerate(bucketing_weights):
            if not isinstance(weight, int) or weight <= 0:
                raise ValueError(f"bucket weights must be positive integers")

    if len(manifest_filepaths) != len(tarred_audio_filepaths):
        raise ValueError(
            f"manifest_filepaths (length={len(manifest_filepaths)}) and tarred_audio_filepaths (length={len(tarred_audio_filepaths)}) need to have the same number of buckets."
        )

    for dataset_idx, (tarred_audio_filepath, manifest_filepath) in enumerate(
        zip(tarred_audio_filepaths, manifest_filepaths)
    ):
        if len(tarred_audio_filepath) == 1:
            tarred_audio_filepath = tarred_audio_filepath[0]

        dataset = TarredAudioToMultiLabelDataset(
            audio_tar_filepaths=tarred_audio_filepath,
            manifest_filepath=manifest_filepath,
            sample_rate=cfg["sample_rate"],
            labels=cfg['labels'],
            shuffle_n=shuffle_n,
            int_values=cfg.get("int_values", False),
            augmentor=augmentor,
            min_duration=cfg.get('min_duration', None),
            max_duration=cfg.get('max_duration', None),
            trim_silence=cfg.get('trim_silence', False),
            is_regression_task=cfg.get('is_regression_task', False),
            delimiter=cfg.get("delimiter", None),
            shard_strategy=cfg.get('tarred_shard_strategy', 'scatter'),
            global_rank=global_rank,
            world_size=world_size,
        )

        if bucketing_weights:
            [datasets.append(dataset) for _ in range(bucketing_weights[dataset_idx])]
        else:
            datasets.append(dataset)

    return get_chain_dataset(datasets=datasets, ds_config=cfg)
