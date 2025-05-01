import os
import tempfile
import torch
from contextlib import ContextDecorator

def is_first_rank_on_node() -> bool:
    env = os.environ
    if "LOCAL_RANK" in env:
        return int(env["LOCAL_RANK"]) == 0
    if "SLURM_LOCALID" in env:
        return int(env["SLURM_LOCALID"]) == 0
    if "OMPI_COMM_WORLD_LOCAL_RANK" in env:
        return int(env["OMPI_COMM_WORLD_LOCAL_RANK"]) == 0
    return True

class FirstRankPerNode(ContextDecorator):
    """
    Context manager for ensuring only LOCAL_RANK==0 runs the code first
    on each node. Creates and destroys a temporary process group if needed.
    """

    def __enter__(self):
        self._created_pg = False
        self._temp_group = None
        if not torch.distributed.is_initialized():
            self._temp_group = self._bootstrap_pg()
            self._created_pg = True
            self._group = self._temp_group
        else:
            self._group = None  # Use the default global group

        self._first = is_first_rank_on_node()
        if not self._first:
            torch.distributed.barrier(group=self._group)
        return self._first

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            pass
        finally:
            if self._first:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier(group=self._group)
                if exc_type is not None:
                    torch.distributed.abort()
            if self._created_pg and self._temp_group is not None:
                torch.distributed.destroy_process_group(self._temp_group)
        # propagate exception outside context manager
        return False

    def _bootstrap_pg(self):
        env = os.environ
        if all(k in env for k in ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT")):
            # Use env:// with the correct world size/rank
            group = torch.distributed.new_group(backend="gloo")
            return group
        else:
            torch.distributed.init_process_group(
                backend="gloo",
                init_method=None,
                world_size=int(env.get('WORLD_SIZE', '1')),
                rank=int(env.get('RANK', '0')),
            )
            return None  # The default group is used in single-process

