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

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_label import _speech_collate_fn
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common.parts.preprocessing import collections
from nemo.core.classes import Dataset
from nemo.utils import logging

__all__ = ["AudioToMultiLabelDataset"]


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
        discarded = []
        for sample in data_list:
            if Path(sample.audio_file).is_file():
                results.append(sample)
            else:
                cnt += 1
                discarded.append(sample.audio_file)
        logging.info(f"{cnt} audio files were discarded since not found.")
        logging.info(discarded[:5])
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
