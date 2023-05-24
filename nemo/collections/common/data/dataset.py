# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import torch.utils.data as pt_data
from torch.utils.data import Dataset, IterableDataset

__all__ = ['ConcatDataset', 'ConcatMapDataset']

# toggle on / off
_CONCAT_SAMPLES = os.getenv('CONCAT_SAMPLES')
if _CONCAT_SAMPLES is None:
    _CONCAT_SAMPLES = False

# if true, concatenated samples are counted as just one sample. 
# otherwise, they are counted "normally", N joined
# samples are N samples. But,  dataset length will not be observed!
_CONCAT_SAMPLES_COUNT_AS_ONE = os.getenv('_CONCAT_SAMPLES_COUNT_AS_ONE')
if _CONCAT_SAMPLES_COUNT_AS_ONE is None:
    _CONCAT_SAMPLES_COUNT_AS_ONE = True

_CONCAT_SAMPLES_MAX_LENGTH = os.getenv('CONCAT_SAMPLES_MAX_LENGTH')
if _CONCAT_SAMPLES_MAX_LENGTH is None:
    _CONCAT_SAMPLES_MAX_LENGTH = 20
else:
    _CONCAT_SAMPLES_MAX_LENGTH = int(_CONCAT_SAMPLES_MAX_LENGTH)



_CONCAT_SAMPLES_MIN_LENGTH = os.getenv('CONCAT_SAMPLES_MIN_LENGTH')
if _CONCAT_SAMPLES_MIN_LENGTH is None:
    _CONCAT_SAMPLES_MIN_LENGTH = 16
else:
    _CONCAT_SAMPLES_MIN_LENGTH = int(_CONCAT_SAMPLES_MIN_LENGTH)

_CONCAT_SAMPLES_JOINING_PAUSE_MSEC = os.getenv('CONCAT_SAMPLES_JOINING_PAUSE_MSEC')
if _CONCAT_SAMPLES_JOINING_PAUSE_MSEC is None:
    _CONCAT_SAMPLES_JOINING_PAUSE_MSEC = 100
else:
    _CONCAT_SAMPLES_JOINING_PAUSE_MSEC = int(_CONCAT_SAMPLES_JOINING_PAUSE_MSEC)

# sizes of tokenizers in the aggregate tokenizer
# 256,256,256,256,256,256,256,256,256,256
AGG_TOK_SIZES = os.getenv('AGGREGATE_TOKENIZER_SIZES')
if AGG_TOK_SIZES is not None:
    AGG_TOK_SIZES = AGG_TOK_SIZES.split(',')
# would need to be able to compute the token id of the space based on some token id.
SPACE_ID_LOOKUP_TABLE = {}
_offset = 0
for c in AGG_TOK_SIZES:
    for i in range(c):
        # the space token id is the first one in the tokenizer.
        SPACE_ID_LOOKUP_TABLE[_offset + i] = _offset


class ConcatDataset(IterableDataset):
    """
    A dataset that accepts as argument multiple datasets and then samples from them based on the specified 
    sampling technique.
    Args:
        datasets (list): A list of datasets to sample from.
        shuffle (bool): Whether to shuffle individual datasets. Only works with non-iterable datasets. 
            Defaults to True.
        sampling_technique (str): Sampling technique to choose which dataset to draw a sample from.
            Defaults to 'temperature'. Currently supports 'temperature', 'random' and 'round-robin'.
        sampling_temperature (int): Temperature value for sampling. Only used when sampling_technique = 'temperature'.
            Defaults to 5.
        sampling_scale: Gives you the ability to upsample / downsample the dataset. Defaults to 1.
        sampling_probabilities (list): Probability values for sampling. Only used when sampling_technique = 'random'.
        seed: Optional value to seed the numpy RNG.
        global_rank (int): Worker rank, used for partitioning map style datasets. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning map style datasets. Defaults to 1.
    """

    def __init__(
        self,
        datasets: List[Any],
        shuffle: bool = True,
        sampling_technique: str = 'temperature',
        sampling_temperature: int = 5,
        sampling_scale: int = 1,
        sampling_probabilities: List[float] = None,
        seed: Optional[int] = None,
        global_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()

        supported_sampling_techniques = ['temperature', 'random', 'round-robin']
        self.datasets = datasets
        self.iterables = [None] * len(datasets)
        self.shuffle = shuffle
        self.global_rank = global_rank
        self.world_size = world_size
        self.sampling_kwargs = {}
        self.sampling_scale = sampling_scale

        if sampling_technique == 'temperature':
            self.index_generator = ConcatDataset.temperature_generator
            self.sampling_kwargs['temperature'] = sampling_temperature
            self.sampling_kwargs['seed'] = seed
        elif sampling_technique == 'random':
            self.index_generator = ConcatDataset.random_generator
            self.sampling_kwargs['p'] = sampling_probabilities
            self.sampling_kwargs['seed'] = seed
        elif sampling_technique == 'round-robin':
            self.index_generator = ConcatDataset.round_robin_generator
        else:
            raise ValueError(f"Currently we only support sampling techniques in {supported_sampling_techniques}.")
        self.length = 0

        if isinstance(datasets[0], IterableDataset):
            self.kind = 'iterable'
        else:
            self.kind = 'map'

        for idx, dataset in enumerate(datasets):
            isiterable = isinstance(dataset, IterableDataset)
            if (isiterable and not self.kind == 'iterable') or (not isiterable and self.kind == 'iterable'):
                raise ValueError("All datasets in ConcatDataset must be of the same kind (Iterable or Map).")

            if self.kind == 'map':
                self.length += len(dataset) // world_size
            else:
                self.length += len(dataset)

        if self.sampling_scale != 1:
            self.length = int(self.length * self.sampling_scale)
            logging.info(f'applying {sampling_scale} sampling scale, concat ds len: {self.length}')

    def get_iterable(self, dataset):
        if isinstance(dataset, IterableDataset):
            return dataset.__iter__()
        else:
            indices = np.arange(len(dataset))
            if self.shuffle:
                np.random.shuffle(indices)
            return iter(indices)

    def __iter__(self):
        worker_info = pt_data.get_worker_info()
        if worker_info is None:
            max_elements = self.length
            wid = 0
            wnum = 1
        else:
            wid = worker_info.id
            wnum = worker_info.num_workers
            max_elements = len(range(wid, self.length, wnum))

        if self.kind == 'map':
            for idx in range(len(self.datasets)):
                start_idx = (len(self.datasets[idx]) // self.world_size) * self.global_rank
                end_idx = start_idx + (len(self.datasets[idx]) // self.world_size)
                if self.global_rank == self.world_size - 1:
                    end_idx = len(self.datasets[idx])
                indices = range(start_idx + wid, end_idx, wnum)
                self.datasets[idx] = pt_data.Subset(self.datasets[idx], indices)

        for idx, dataset in enumerate(self.datasets):
            iterable = self.get_iterable(dataset)
            self.iterables[idx] = iterable

        n = 0
        ind_gen = self.index_generator(self.datasets, **self.sampling_kwargs)
        while n < max_elements:

            if _CONCAT_SAMPLES:
                samp, incr = self.pull_concatenated_sample(ind_gen)
            else:
                samp, incr = self.pull_sample(ind_gen)

            if samp is not None:
                n += incr
                yield samp
            else:
                if incr is None: # ind gen iterator ended... hmm
                    return
                # otherwise we do nothing and just retry
                # this means we are restarting one of the datasets

# The original logic
#            n += 1
#            try:
#                ind = next(ind_gen)
#            except StopIteration:
#                return
#            try:
#                val = next(self.iterables[ind])
#                if self.kind == 'map':
#                    val = self.datasets[ind][val]
#                yield val
#            except StopIteration:
#                self.iterables[ind] = self.get_iterable(self.datasets[ind])
#                n -= 1

    def pull_concatenated_sample(self, ind_gen):
        """
        Return a concatenated sample as well the number of samples pulled
        Merge a bunch of samples into one.
        """
        # f, fl, torch.tensor(t).long(), torch.tensor(tl).long()
        # features, features length, tokens, tokens length
        # need a spectrogram of random noise.. 
        # need also a space token from each sample
        # we can artifically add a space token to each manifest sample. then here we just need to learn to delete the last token of a sample.
        # the spectrogram though.. has shape of torch.Size([16 , 80, 1063]),
        # which one am i using? use that!

        # SPACE_ID_LOOKUP_TABLE
        _f = None
        _fl = 0
        _t = None
        _tl = 0
        
        while _fl < _CONCAT_SAMPLES_MIN_LENGTH:
            f, fl, t, tl = self.pull_sample()
            if _fl + fl > _CONCAT_SAMPLES_MAX_LENGTH:  # need to check if these are compatible lengths obviously.
                # just try another sample if this one is too long.
                continue

            # the length needs to be converted because of the sampling.. ughhh 
            _f, _fl = self.concat_with_pause(_f, _fl, f, fl, _CONCAT_SAMPLES_JOINING_PAUSE_MSEC)
            _space_id = SPACE_ID_LOOKUP_TABLE[t[-1]] # careful here because it is a tensor.. 
            _t = _t + _space_id + t  # likely needs to be concat
            _tl += tl + 1 # adding the space token in the language of last sample.

        return _f, _fl, _t, _tl


    def pull_sample(self, ind_gen):
        """
        Return a sample as well the number of samples pulled
        If the index generator ended, we return None, None
        If one of the dataset iterators ended, we return None, 0
        """
        _sample = None

        while _sample is None:
            try:
                ind = next(ind_gen)
            except StopIteration:
                return None, None

            try:
                _sample = next(self.iterables[ind])
                if self.kind == 'map':
                    _sample = self.datasets[ind][val]

            except StopIteration:
                self.iterables[ind] = self.get_iterable(self.datasets[ind])

        return _sample, 1

    def __len__(self):
        return self.length

    @staticmethod
    def temperature_generator(datasets, **kwargs):
        temp = kwargs.get('temperature')
        if not temp:
            raise ValueError("Temperature generator expects a 'temperature' keyword argument.")

        seed = kwargs.get('seed', None)
        np_rng = np.random.RandomState(seed)
        lengths = []
        num = len(datasets)
        for dataset in datasets:
            lengths.append(len(dataset))

        p = np.array(lengths) / np.sum(lengths)
        p = np.power(p, 1 / temp)
        p = p / np.sum(p)

        while True:
            ind = np_rng.choice(np.arange(num), p=p)
            yield ind

    @staticmethod
    def round_robin_generator(datasets, **kwargs):
        num = len(datasets)
        while True:
            for i in range(num):
                yield i

    @staticmethod
    def random_generator(datasets, **kwargs):
        p = kwargs.get('p')
        if not p:
            raise ValueError("Random generator expects a 'p' keyowrd argument for sampling probabilities.")

        seed = kwargs.get('seed', None)
        np_rng = np.random.RandomState(seed)
        num = len(datasets)
        if len(p) != num:
            raise ValueError("Length of probabilities list must be equal to the number of datasets.")

        while True:
            ind = np_rng.choice(np.arange(num), p=p)
            yield ind


class ConcatMapDataset(Dataset):
    """
    A dataset that accepts as argument multiple datasets and then samples from them based on the specified 
    sampling technique.
    Args:
        datasets (list): A list of datasets to sample from.
        sampling_technique (str): Sampling technique to choose which dataset to draw a sample from.
            Defaults to 'temperature'. Currently supports 'temperature', 'random' and 'round-robin'.
        sampling_temperature (int): Temperature value for sampling. Only used when sampling_technique = 'temperature'.
            Defaults to 5.
        sampling_probabilities (list): Probability values for sampling. Only used when sampling_technique = 'random'.
        seed: Optional value to seed the numpy RNG.
    """

    def __init__(
        self,
        datasets: List[Any],
        sampling_technique: str = 'temperature',
        sampling_temperature: int = 5,
        sampling_probabilities: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.datasets = datasets
        self.lengths = [len(x) for x in self.datasets]
        self.sampling_technique = sampling_technique
        self.sampling_temperature = sampling_temperature
        self.sampling_probabilities = sampling_probabilities
        self.np_rng = np.random.RandomState(seed)

        # Build a list of size `len(self)`. Each tuple contains (dataset_id, dataset_index)
        self.indices: List[Tuple[int, int]] = []
        # Current position as we consume indices from each data set
        dataset_positions = [0] * len(self.datasets)
        # Random permutation of each dataset. Will be regenerated when exhausted.
        shuffled_indices = [self.np_rng.permutation(len(x)) for x in self.datasets]
        # Build the list of randomly-chosen datasets spanning the entire length, adhering to sampling technique
        if self.sampling_technique == "round-robin":
            # To exhaust longest dataset, need to draw `num_datasets * max_dataset_len` samples
            total_length = max(self.lengths) * len(self.lengths)
            # For round robin, iterate through each dataset
            dataset_ids = np.arange(total_length) % len(self.datasets)
            for dataset_id in dataset_ids:
                position = dataset_positions[dataset_id]
                index = shuffled_indices[dataset_id][position]
                self.indices.append((dataset_id, index))
                dataset_positions[dataset_id] += 1
                if dataset_positions[dataset_id] == len(shuffled_indices[dataset_id]):
                    dataset_positions[dataset_id] = 0
                    shuffled_indices[dataset_id] = self.np_rng.permutation(len(self.datasets[dataset_id]))
        else:
            # Resolve probabilities of drawing from each data set
            if self.sampling_technique == "random":
                if sampling_probabilities is None or len(sampling_probabilities) != len(self.datasets):
                    raise ValueError(
                        f"Need {len(self.datasets)} probabilities; got "
                        f"{len(sampling_probabilities) if sampling_probabilities is not None else 'None'}"
                    )
                p = np.array(self.sampling_probabilities)
            elif self.sampling_technique == "temperature":
                p = np.array([len(x) for x in self.datasets])
                p = np.power(p, 1 / self.sampling_temperature)
            else:
                raise ValueError(f"Couldn't interpret sampling technique: {sampling_technique}")
            # Normalize probabilities
            p = p / np.sum(p)
            # Will randomly choose from datasets
            choices = np.arange(len(self.datasets))
            # Keep going until largest dataset is exhausted.
            exhausted_datasets = set()
            while len(exhausted_datasets) < len(self.datasets):
                # Randomly choose a dataset for each position in accordance with p
                dataset_id = self.np_rng.choice(a=choices, p=p)
                dataset = self.datasets[dataset_id]
                # Pick next index from dataset
                position = dataset_positions[dataset_id]
                index = shuffled_indices[dataset_id][position]
                self.indices.append((dataset_id, index))
                # Maybe reset this dataset's permutation
                dataset_positions[dataset_id] += 1
                if dataset_positions[dataset_id] >= len(dataset):
                    shuffled_indices[dataset_id] = self.np_rng.permutation(len(dataset))
                    dataset_positions[dataset_id] = 0
                    exhausted_datasets.add(dataset_id)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        dataset_id, dataset_index = self.indices[idx]
        return self.datasets[dataset_id][dataset_index]
