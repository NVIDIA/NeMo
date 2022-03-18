import os
import sys

import hydra

from bignlp.train_scripts.train_utils import pause_and_prime_dns_connections, generate_cmd_prefix, numa_mapping, \
    convert_args_to_hydra_train_args


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):
    args = sys.argv[1:]
    hydra_train_args = convert_args_to_hydra_train_args(args)

    bignlp_path = cfg.get("bignlp_path")
    dgxa100_gpu2core = cfg.get("dgxa100_gpu2core")
    dgxa100_gpu2mem = cfg.get("dgxa100_gpu2mem")
    training_config = cfg.training_config.rsplit('/', 1)[1]
    training_config_path = os.path.join(bignlp_path, "conf/training", cfg.training_config.rsplit('/', 1)[0])
    flags = f"--config-path={training_config_path} --config-name={training_config} "

    gpu_mapping, core_mapping = numa_mapping(dgxa100_gpu2core, dgxa100_gpu2mem)

    code_dir = "/opt/bignlp/NeMo"
    code_path = (
        f"{code_dir}/examples/nlp/language_modeling/megatron_gpt_pretraining.py"
    )
    cmd_prefix = generate_cmd_prefix(cfg, code_dir)
    # Write command to launch training.
    if cfg.cluster_type == "bcm":
        cmd = f'{cmd_prefix} {gpu_mapping} {core_mapping} python3 {code_path} {hydra_train_args} {flags}'
    elif cfg.cluster_type == "bcp":
        pause_and_prime_dns_connections()
        cmd = f'{cmd_prefix} {gpu_mapping} {core_mapping} python3 {code_path} +cluster_type=BCP +rank={os.environ.get("RANK")}  {hydra_train_args} {flags}'
    os.system(f"{cmd}")


if __name__ == "__main__":
    main()
