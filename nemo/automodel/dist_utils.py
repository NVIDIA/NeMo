#!/usr/bin/python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
from contextlib import ContextDecorator
import torch
import torch.distributed as dist

class FirstRankPerNode(ContextDecorator):
    """
    Context manager that:
      • Lets LOCAL_RANK==0 run the protected code first on each node.
      • Inserts an extra barrier across *only* the node‑local rank‑0 processes.
      • Works on a single GPU (no env flags, no distributed initialisation).

    Note: it is assumed the scoped code is not torch.distributed heavy.
    """

    def __enter__(self):
        self._created_pg = False
        self._node0_group = None
        self._first = True          # default for single‑GPU / no‑dist case

        # ------------------------------------------------------------------ #
        # 1. Make sure there is at least *some* process‑group initialised
        # ------------------------------------------------------------------ #
        if not dist.is_initialized():
            self._created_pg = self._try_bootstrap_pg()

        if not dist.is_initialized():                     # pure single GPU
            return True                                   # I am “first”

        # ------------------------------------------------------------------ #
        # 2. Figure out local/global ranks
        # ------------------------------------------------------------------ #
        env = os.environ
        world_size   = dist.get_world_size()
        global_rank  = dist.get_rank()
        local_rank   = int(env.get("LOCAL_RANK", global_rank))  # fallback
        self._first  = local_rank == 0

        # ------------------------------------------------------------------ #
        # 3. Build a subgroup that contains exactly one rank per node
        #    (those where local_rank == 0)
        # ------------------------------------------------------------------ #
        # Gather all local_ranks so every process can derive the same subgroup
        gathered_local = [None] * world_size
        dist.all_gather_object(gathered_local, local_rank)
        node0_ranks = [idx for idx, lr in enumerate(gathered_local) if lr == 0]

        # new_group must be called by *all* ranks with identical `ranks` list
        self._node0_group = dist.new_group(ranks=node0_ranks, backend="gloo")

        # ------------------------------------------------------------------ #
        # 4. Synchronisation logic
        # ------------------------------------------------------------------ #
        if not self._first:
            # Non‑rank‑0 processes wait for their node‑rank-0
            dist.barrier()
        else:
            # Rank‑0 processes wait for their *peer* rank‑0s on other nodes
            if dist.get_world_size(self._node0_group) > 1:
                dist.barrier(group=self._node0_group)

        return self._first

    # ====================================================================== #
    # Exit
    # ====================================================================== #
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._first and dist.is_initialized():
                # Ensure all node‑0 peers leave the context together
                if self._node0_group is not None and dist.get_world_size(self._node0_group) > 1:
                    dist.barrier(group=self._node0_group)
                # Re‑sync the whole world so that non‑rank‑0s can proceed
                dist.barrier()
                if exc_type is not None:
                    dist.abort()      # propagate failure to the entire job
        finally:
            if self._created_pg:
                dist.destroy_process_group()

        # propagate any exception to outer scope
        return False

    # ====================================================================== #
    # Helper
    # ====================================================================== #
    def _try_bootstrap_pg(self) -> bool:
        """Try to create a (single‑node) default pg from env:// variables."""
        env = os.environ
        required = ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT")
        if all(k in env for k in required):
            dist.init_process_group(
                backend="gloo",
                world_size=int(env.get("WORLD_SIZE")),
                rank=int(env.get("RANK")),
            )
            return True
        return False
