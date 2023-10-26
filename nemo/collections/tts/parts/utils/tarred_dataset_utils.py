from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set

import numpy as np
import webdataset as wd

from nemo.collections.asr.parts.preprocessing.segment import available_formats as valid_sf_formats
from nemo.collections.tts.parts.utils.tts_dataset_utils import filter_dataset_by_duration
from torch.utils.data import IterableDataset


VALID_AUDIO_FORMATS = ';'.join(['wav', 'mp3', 'flac'] + [fmt.lower() for fmt in valid_sf_formats.keys()])


@dataclass
class TarredMetadata:
    manifest_path: Path
    tar_filepath: str = None
    sample_weight: float = 1.0


@dataclass
class TarredSample:
    dataset_name: str
    manifest_entry: Dict[str, Any]


def get_file_id(audio_filepath: Path):
    file_id = audio_filepath.with_suffix("")
    file_id = str(file_id).replace("/", "_").replace(".", "_")
    return file_id


def process_tarred_manifest(
    dataset_name: str, entries: List[Dict[str, Any]], min_duration: float, max_duration: float,
):
    unfiltered_file_count = len(entries)
    filtered_entries, unfiltered_hours, filtered_hours = filter_dataset_by_duration(
        entries=entries, min_duration=min_duration, max_duration=max_duration
    )

    file_to_sample_map = {}
    for entry in filtered_entries:
        sample = TarredSample(dataset_name=dataset_name, manifest_entry=entry)
        audio_filepath = Path(entry["audio_filepath"])
        file_id = get_file_id(audio_filepath=audio_filepath)
        file_to_sample_map[file_id] = sample

    return file_to_sample_map, unfiltered_file_count, unfiltered_hours, filtered_hours


class FileFilterIterator:
    def __init__(self, iterator: Iterator, file_ids: Set[str], id_field="key"):
        self.iterator = iterator
        self.file_ids = file_ids
        self.id_field = id_field

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            data_dict = next(self.iterator)
            file_id = data_dict[self.id_field]
            if file_id in self.file_ids:
                return data_dict


def create_tarred_dataset(
    sample_type: str,
    sample_args: Optional[Dict],
    datasets: List[IterableDataset],
    dataset_lengths: List[int],
):
    if sample_args is None:
        sample_args = {}

    if sample_type == "concat":
        return TarredConcatDataset(datasets=datasets, dataset_lengths=dataset_lengths)
    elif sample_type == "weighted_random":
        return TarredWeightedRandomDataset(datasets=datasets, dataset_lengths=dataset_lengths, **sample_args)
    else:
        raise ValueError(f"Unknown sampling type {sample_type}")


class TarredConcatDataset(IterableDataset, wd.Shorthands, wd.Composable):
    def __init__(
        self,
        datasets: List[IterableDataset],
        dataset_lengths: List[int]
    ):
        super().__init__()
        self.datasets = datasets
        self.dataset_lengths = dataset_lengths
        self.iterators = [iter(ds) for ds in self.datasets]

    def __len__(self):
        return sum(self.dataset_lengths)

    def __iter__(self):
        iterators = [iter(dataset) for dataset in self.datasets]
        for iterator in iterators:
            for sample in iterator:
                yield sample


class TarredWeightedRandomDataset(IterableDataset, wd.Shorthands, wd.Composable):
    def __init__(
        self,
        datasets: List[IterableDataset],
        dataset_lengths: List[int],
        dataset_weights: List[float],
        batch_size: int,
        steps_per_epoch: int
    ):
        super().__init__()
        assert len(datasets) == len(dataset_lengths) == len(dataset_weights)

        self.datasets = datasets
        self.dataset_lengths = dataset_lengths
        self.dataset_weights = dataset_weights
        self.samples_per_epoch = batch_size * steps_per_epoch
        self.dataset_indices = list(range(len(self.datasets)))
        self.iterators = [iter(dataset) for dataset in self.datasets]

    def __len__(self):
        return self.samples_per_epoch

    def __iter__(self):
        for _ in range(self.samples_per_epoch):
            dataset_index = np.random.choice(self.dataset_indices, size=1, p=self.dataset_weights)[0]

            try:
                sample = next(self.iterators[dataset_index])
            except StopIteration:
                self.iterators[dataset_index] = iter(self.datasets[dataset_index])
                sample = next(self.iterators[dataset_index])

            yield(sample)
