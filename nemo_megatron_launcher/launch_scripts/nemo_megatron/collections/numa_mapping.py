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

import os

import hydra


def numa_mapping(local_rank, devices, numa_cfg):
    """Sets the GPU affinity for the NUMA mapping for the current GPU passed as local_rank.
    It sets the NUMA mapping following the parameters in numa_cfg.

    Arguments:
        local_rank: int, local_rank as it will be passed to PyTorch.
        devices: int, number of GPUs per node, or nproc_per_node.
        numa_cfg: OmegaConf, config to set the numa mapping parameters.
    """
    enable = numa_cfg.get("enable")
    mode = numa_cfg.get("mode")
    scope = numa_cfg.get("scope")
    cores = numa_cfg.get("cores")
    balanced = numa_cfg.get("balanced")
    min_cores = numa_cfg.get("min_cores")
    max_cores = numa_cfg.get("max_cores")

    if enable:
        from gpu_affinity import set_affinity

        affinity = set_affinity(
            gpu_id=int(local_rank),
            nproc_per_node=devices,
            mode=mode,
            scope=scope,
            cores=cores,
            balanced=balanced,
            min_cores=min_cores,
            max_cores=max_cores,
        )
        print(f"Setting NUMA mapping (GPU Affinity) for rank {local_rank}: {affinity}")
    else:
        print("No NUMA mapping was enabled, performance might be affected.")


@hydra.main(config_path="conf", config_name="numa_mapping")
def main(cfg):
    rank = int(os.environ.get("LOCAL_RANK"))
    devices = int(os.environ.get("SLURM_NTASKS_PER_NODE"))  # TODO: Check BCP, interactive

    numa_mapping(
        local_rank=rank, devices=devices, numa_cfg=cfg,
    )


if __name__ == "__main__":
    main()
