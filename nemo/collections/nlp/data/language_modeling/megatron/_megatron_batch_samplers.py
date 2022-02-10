import abc

import torch


__all__ = [
    "MegatronPretrainingBatchSampler",
    "MegatronPretrainingRandomBatchSampler",
]


class BaseMegatronBatchSampler:
    """Base class for Megatron style BatchSampler."""

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __iter__(self):
        ...

    @property
    @abc.abstractmethod
    def local_global_batch_size(self) -> int:
        """The size of global batch on each data parallel rank."""
        ...

    @local_global_batch_size.setter
    @abc.abstractclassmethod
    def local_global_batch_size(self) -> None:
        """The size of global batch on each data parallel rank."""
        ...


class MegatronPretrainingBatchSampler(BaseMegatronBatchSampler):

    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        local_global_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool = True,
    ) -> None:
        # Sanity checks.
        if total_samples <= 0:
            raise RuntimeError(
                'no sample to consume: {}'.format(self.total_samples))
        if consumed_samples >= total_samples:
            raise RuntimeError(
                'no samples left to consume: {}, {}'.format(self.consumed_samples, self.total_samples))
        if local_global_batch_size <= 0:
            raise RuntimeError(
                f"local global_batch size must be greater than 0: {local_global_batch_size}")
        if data_parallel_size <= 0:
            raise RuntimeError(
                f"data parallel size must be greater than 0: {data_parallel_size}")
        if data_parallel_rank >= data_parallel_size:
            raise RuntimeError(
                'data_parallel_rank should be smaller than data size: {}, {}'.format(
                    self.data_parallel_rank, data_parallel_size))
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self._local_global_batch_size = local_global_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.local_global_batch_times_data_parallel_size = self._local_global_batch_size * data_parallel_size
        self.drop_last = drop_last

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self._local_global_batch_size
        end_idx = start_idx + self._local_global_batch_size
        return start_idx, end_idx

    @property
    def local_global_batch_size(self) -> int:
        return self._local_global_batch_size

    @local_global_batch_size.setter
    def local_global_batch_size(self, new_local_global_batch_size) -> None:
        self._local_global_batch_size = new_local_global_batch_size
        self.local_global_batch_times_data_parallel_size = self._local_global_batch_size * self.data_parallel_size

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self._local_global_batch_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class MegatronPretrainingRandomBatchSampler(BaseMegatronBatchSampler):
    """Megatron style Random Batch Sampler.
    Major difference is that `__iter__` yields a local global_batch, not a microbatch.
    A local global_batch consists of `global_batch_size / data_parallel_size`
    Args:
        total_samples: The number of data samples, i.e. ``len(dataset)``.
        consumed_samples: The number of samples already consumed in pretraining.
        local_global_batch_size: The number of data in each batch returned from `__iter__`. Basically
            `local_global_batch_size = global_batch_size / data_parallel_size`.
        data_parallel_rank:
        data_parallel_size:
    """

    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        local_global_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
    ) -> None:
        if total_samples <= 0:
            raise ValueError(f"no sample to consume: total_samples of {total_samples}")
        if local_global_batch_size <= 0:
            raise ValueError(f"Invalid local_global_batch_size: {local_global_batch_size}")
        if data_parallel_size <= 0:
            raise ValueError(f"Invalid data_parallel_size: {data_parallel_size}")
        if data_parallel_rank >= data_parallel_size:
            raise ValueError(
                f"data_parallel_rank should be smaller than data parallel size: {data_parallel_rank} < {data_parallel_size}"
            )
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self._local_global_batch_size = local_global_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.local_global_batch_times_data_parallel_size = self._local_global_batch_size * self.data_parallel_size
        self.last_batch_size = self.total_samples % self.local_global_batch_times_data_parallel_size

    def __len__(self) -> int:
        return self.total_samples

    @property
    def local_global_batch_size(self) -> int:
        return self._local_global_batch_size

    @local_global_batch_size.setter
    def local_global_batch_size(self, new_local_global_batch_size) -> None:
        self._local_global_batch_size = new_local_global_batch_size
        self.local_global_batch_times_data_parallel_size = self._local_global_batch_size * self.data_parallel_size

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        # note(mkozuki): might be better to uncomment
        # assert current_epoch_samples % (self.data_parallel_size * apex.transformer.pipeline_parallel.utils.get_micro_batch_size()) == 0

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.local_global_batch_times_data_parallel_size) * self._local_global_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        g = torch.Generator()
        g.manual_seed(self.epoch)
        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        idx_range = [start_idx + x for x in random_idx[bucket_offset:]]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self._local_global_batch_size:
                self.consumed_samples += self.local_global_batch_times_data_parallel_size
                yield batch
                batch = []
