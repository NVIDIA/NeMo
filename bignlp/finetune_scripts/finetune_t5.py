import os
import sys

import hydra
import datasets

from bignlp.train_scripts.train_utils import pause_and_prime_dns_connections, generate_cmd_prefix, numa_mapping, \
    convert_args_to_hydra_train_args


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):
    args = sys.argv[1:]
    hydra_train_args = convert_args_to_hydra_train_args(args, prefix="finetuning.")

    bignlp_path = cfg.bignlp_path
    finetuning_config = cfg.finetuning_config.rsplit('/', 1)[1]
    finetuning_config_path = os.path.join(bignlp_path, "conf/finetuning", cfg.finetuning_config.rsplit('/', 1)[0])
    flags = f"--config-path={finetuning_config_path} --config-name={finetuning_config} "

    gpu_mapping, core_mapping = numa_mapping(cfg.dgxa100_gpu2core, cfg.dgxa100_gpu2mem)

    code_dir = "/opt/bignlp/NeMo"
    if "mt5" in cfg.finetuning_config:
        code_path = (
            f"{code_dir}/examples/nlp/language_modeling/megatron_t5_xnli.py"
        )
    else:
        code_path = (
            f"{code_dir}/examples/nlp/language_modeling/megatron_t5_glue.py"
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
