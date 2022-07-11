import os
import sys

import hydra

from bignlp.bignlp_utils import convert_to_cli
from bignlp.train_scripts.train_utils import generate_mt5_data_blend
from bignlp.train_scripts.train_utils import (
    pause_and_prime_dns_connections,
    generate_cmd_prefix,
    numa_mapping,
    convert_args_to_hydra_train_args,
)


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):
    args = sys.argv[1:]
    hydra_train_args = convert_args_to_hydra_train_args(args)

    bignlp_path = cfg.get("bignlp_path")
    train_cfg = cfg.get("training")
    trainer_cfg = train_cfg.get("trainer")
    devices = trainer_cfg.get("devices")
    numa_cfg = cfg.get("numa_mapping")

    # Run parameters
    run_cfg = train_cfg.get("run")
    results_dir = run_cfg.get("results_dir")

    rank = int(os.environ.get("LOCAL_RANK"))

    training_config = cfg.get("training_config").rsplit("/", 1)[1]
    training_config_path = os.path.join(
        bignlp_path, "conf/training", cfg.get("training_config").rsplit("/", 1)[0]
    )
    flags = f"--config-path={training_config_path} --config-name={training_config} "

    # Re-build the hydra args to add dataset blend
    if "mt5" in cfg.get("training_config"):
        model_cfg = train_cfg.get("model")
        model_data_cfg = model_cfg.get("data")
        if model_data_cfg.get("data_prefix") is None:
            cfg.get("training").model.data.data_prefix = generate_mt5_data_blend(cfg)
            hydra_args = convert_to_cli(cfg)
            hydra_train_args = convert_args_to_hydra_train_args(hydra_args.split())

    cuda_visible_devices = numa_mapping(local_rank=rank, devices=devices, numa_cfg=numa_cfg)

    code_dir = "/opt/bignlp/NeMo"
    code_path = f"{code_dir}/examples/nlp/language_modeling/megatron_t5_pretraining.py"
    cmd_prefix = generate_cmd_prefix(cfg, code_dir)

    # Write command to launch training.
        if cfg.get("profile", False):
        slurm_node = os.environ.get("SLURM_NODEID", "0")
        slurm_rank = os.environ.get("SLURM_PROCID", "0")
        slurm_jobid = os.environ.get("SLURM_JOB_ID", "0")
        profile_out_path = os.path.join(results_dir, "profile_logs")
        os.makedirs(profile_out_path, exist_ok=True)
        nsys = f"nsys profile -s none -o -t " \
               f"cuda,nvtx {profile_out_path}/profile_{slurm_jobid}_node{slurm_node}_rank{slurm_rank} " \
               f"--force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
    else:
        nsys = ""

    if cfg.get("cluster_type") == "bcm":
        cmd = f"{cmd_prefix} {cuda_visible_devices} {nsys} python3 {code_path} {hydra_train_args} {flags}"
    elif cfg.get("cluster_type") == "bcp":
        pause_and_prime_dns_connections()
        cmd = f'{cmd_prefix} {cuda_visible_devices} python3 {code_path} +cluster_type=BCP +rank={os.environ.get("RANK")}  {hydra_train_args} {flags}'
    os.system(f"{cmd}")


if __name__ == "__main__":
    main()
