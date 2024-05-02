import threading
from queue import Queue
from typing import Any

import torch


class DataLoaderPrefetcher(threading.Thread):
    """
    Starts a background thread that immediately begins iterating the dataloader.
    This is useful if dataloader initialization takes a while and there is other work
    that needs to be performed between dataloader initialization and the start of the training loop,
    in which case it will eliminate (or reduce) the lag between the start of the training script
    and the actual start of training.

    .. note:: This class uses a Python thread for concurrency. It doesn't matter that it gets blocked by GIL,
        because the dataloader typically spawns worker subprocesses that perform actual work.
        The main point of this class it to trigger ``DataLoader.__iter__`` so that it can spawn the workers
        and have them begin loading data immediately, so that the mini-batches are ready by the time
        the script execution reaches the training loop.

    Example::

        >>> dl = torch.utils.data.DataLoader(...)
        ... dl = DataLoaderPrefetcher(dl)
        ... # ... do some work while prefetcher starts dataloader iteration ...
        ... for batch in dl:
        ...     train_step(batch)

    https://github.com/pytorch/audio/blob/e4e171a51714b2b2bd79e1aea199c3f658eddf9a/torchaudio/datasets/utils.py#L238
    """

    def __init__(self, dataloader: torch.utils.data.DataLoader, maxsize: int = 2) -> None:
        super().__init__()
        self.queue = Queue(maxsize)
        self.dataloader = dataloader
        self.daemon = True
        self.start()

    @property
    def dataset(self):
        return self.dataloader.dataset

    @property
    def sampler(self):
        return self.dataloader.sampler

    def run(self) -> None:
        for item in self.dataloader:
            self.queue.put(item)
        self.queue.put(_End)

    def __iter__(self) -> "DataLoaderPrefetcher":
        return self

    def __next__(self) -> Any:
        next_item = self.queue.get()
        if next_item == _End:
            raise StopIteration
        return next_item


class _End:
    pass
