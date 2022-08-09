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
    devices = int(os.environ.get("SLURM_TASKS_PER_NODE")) # TODO: Check BCP, interactive

    numa_mapping(
        local_rank=rank,
        devices=devices,
        numa_cfg=cfg,
    )

if __name__ == "__main__":
    main()