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

    # If not set by pytorch, we need to determine node_rank
    def get_node_rank():
        # Use an equivalent of pytorch lightning's determine_ddp_node_rank()
        node_rank = 0
        # First check if running on a slurm cluster
        # TODO: This check could probably be better
        num_slurm_tasks = get_envint("SLURM_NTASKS", 0)
        if num_slurm_tasks > 0:
            node_rank = get_envint("SLURM_NODEID", 0)
        else:
            node_rank_env = get_envint("NODE_RANK", None)
            group_rank = get_envint("GROUP_RANK", None)
            if group_rank:
                node_rank = group_rank
            # Take from NODE_RANK whenever available
            if node_rank_env:
                node_rank = node_rank_env
        return node_rank

    node_rank = get_node_rank()
    local_rank = get_envint("LOCAL_RANK", 0)
    return node_rank == 0 and local_rank == 0
