import os
import sys
import re

import hydra

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

    rank = int(os.environ.get("LOCAL_RANK"))

    training_config = cfg.training_config.rsplit("/", 1)[1]
    training_config_path = os.path.join(
        bignlp_path, "conf/training", cfg.training_config.rsplit("/", 1)[0]
    )
    flags = f"--config-path={training_config_path} --config-name={training_config} "

    cuda_visible_devices = numa_mapping(local_rank=rank, devices=devices, numa_cfg=numa_cfg)

    code_dir = "/opt/bignlp/NeMo"
    code_path = f"{code_dir}/examples/nlp/language_modeling/megatron_gpt_pretraining.py"
    cmd_prefix = generate_cmd_prefix(cfg, code_dir)

    # Write command to launch training.
    if cfg.cluster_type == "bcm":
        cmd = f"{cmd_prefix} {cuda_visible_devices} python3 {code_path} {hydra_train_args} {flags}"
    elif cfg.cluster_type == "bcp":
        pause_and_prime_dns_connections()
        cmd = f'{cmd_prefix} {cuda_visible_devices} python3 {code_path} +cluster_type=bcp +rank={os.environ.get("RANK")}  {hydra_train_args} {flags}'
    os.system(f"{cmd}")


if __name__ == "__main__":
    main()
