import os
from typing import List, Optional

import torch
from pytorch_lightning.callbacks.callback import Callback
from nemo.lightning import io

from nemo.utils import logging
from nemo.utils.get_rank import get_rank


class MemoryProfileCallback(Callback, io.IOMixin):
    """
    This callback enables recording a timeline of memory allocations during training.
    The generated .pickle profiles can be analyzed at https://pytorch.org/memory_viz

    More info about the profiles can be found [here](https://pytorch.org/blog/understanding-gpu-memory-1/).

    Args:
        dir (Optional[str]): Directory to store the memory profile dump

    Example:
        >>> callback = MemoryProfileCallback(dir="/mem_profile")
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(self, dir: str = "/mem_profile"):

        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)
        logging.info(f"Torch memory profiles will be written to: {self.dir},")

    def setup(self, trainer, pl_module, stage) -> None:
        """PyTorch Lightning hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-end
        We use it here to start recording the memory profiler.
        """

        if torch.distributed.is_initialized():
            torch.cuda.memory._record_memory_history(max_entries=100000)

    def on_train_end(self, trainer, pl_module) -> None:
        """PyTorch Lightning hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-train-end
        We use it here to finish memory profiling and write the snapshot.
        """

        logging.info(
            f"on_train_batch_end rank: {torch.distributed.get_rank()} mem: {torch.cuda.memory_allocated()/1024/1024/1024} / {torch.cuda.max_memory_reserved()/1024/1024/1024}"
        )

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            _snapshot_path = f"{self.dir}/memory_snapshot-rank{rank}.pickle"
            logging.info(f"Writing memory profile snapshot to {_snapshot_path}")
            torch.cuda.memory._dump_snapshot(f"{_snapshot_path}")
            torch.cuda.memory._record_memory_history(enabled=None)
            logging.info(f"Finished writing memory profile snapshot: {_snapshot_path}")
