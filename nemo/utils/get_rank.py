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

from nemo.utils.env_var_parsing import get_envint


def is_global_rank_zero():
    """ Helper function to determine if the current process is global_rank 0 (the main process)
    """
    # Try to get the pytorch RANK env var
    # RANK is set by torch.distributed.launch
    rank = get_envint("RANK", None)
    if rank:
        return rank == 0

    # on SLURM variables SLURM_NODEID and SLURM_PROCID will be defined
    # SLURM_PROCID holds the global rank and computing the node_rank is
    # not strictly necessary, but we first check LOCAL_RANK to retain 
    # the behaviour before the change  
    node_rank = get_envint("SLURM_NODEID", get_envint("NODE_RANK", get_envint("GROUP_RANK", 0)))
    local_rank = get_envint("LOCAL_RANK", get_envint("SLURM_PROCID", 0))
    return node_rank == 0 and local_rank == 0
