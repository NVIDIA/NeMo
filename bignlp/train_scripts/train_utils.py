import os
import sys
import re
import socket
import time
import math
from collections import defaultdict

from gpu_affinity import set_affinity


rank2gpu = [0, 4, 2, 6, 1, 5, 3, 7]


def pause_and_prime_dns_connections() -> None:
    if int(os.environ.get("GROUP_RANK")) > 0:
        time.sleep(20)
        prime_dns_connections()
    elif int(os.environ.get("LOCAL_RANK")) != 0:
        time.sleep(10)


def prime_dns_connections() -> None:
    me = "worker" + os.environ.get("GROUP_RANK") + ":" + os.environ.get("RANK")
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = int(os.environ.get("MASTER_PORT"))
    print(f"SPDNS: {me} Connecting to {master_addr}:{master_port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (master_addr, master_port)
    timeout = time.time() + 300
    connected = False
    while not connected:
        try:
            sock.connect(server_address)
            connected = True
        except Exception:
            time.sleep(2)
        if time.time() > timeout:
            print(f"{me} couldnt connect to {master_addr}:{master_port} timed out! (300s)")
            sys.exit(110)
    print(f"SPDNS: {me} connected to {master_addr}:{master_port}")
    sock.close()


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
        print("No NUMA mapping was enabled, performance might suffer without it.")

    cuda_visible_devices = "CUDA_VISIBLE_DEVICES={}".format(re.sub("[\[\] ]", "", str(rank2gpu)))
    return cuda_visible_devices


def generate_cmd_prefix(cfg, code_dir):
    nccl_topo_file = cfg.nccl_topology_xml_file
    nccl_cmd = ""
    if nccl_topo_file is not None:
        nccl_cmd = f"export NCCL_TOPO_FILE={nccl_topo_file}; "

    # W&B Api Key file.
    wandb_cmd = ""
    if cfg.wandb_api_key_file is not None:
        with open(cfg.wandb_api_key_file, "r") as f:
            wandb_api_key = f.readline().rstrip()
        wandb_cmd = f"wandb login {wandb_api_key}; "

    # Write command to launch training.
    cmd_prefix = f'{wandb_cmd} cd {code_dir}; git rev-parse HEAD; cd {code_dir}/nemo/collections/nlp/data/language_modeling/megatron; make; export PYTHONPATH="{code_dir}/.:$PYTHONPATH"; export TRANSFORMERS_CACHE="/tmp/.cache/"; {nccl_cmd}'
    return cmd_prefix


def convert_args_to_hydra_train_args(args, prefix="training."):
    for index, arg in enumerate(args):
        k, v = arg.split("=", 1)
        if "splits_string" in k and "\\" not in v:
            args[index] = "{}={}".format(k, v.replace("'", "\\'"))

    train_args = [x.replace(prefix, "") for x in args if x.startswith(prefix)]
    train_args = [x.replace("None", "null") for x in train_args if "run." not in x]
    hydra_train_args = " ".join(train_args)
    return hydra_train_args


def generate_mt5_data_blend(cfg):
    train_cfg = cfg.training
    if train_cfg.model.data.data_prefix is not None:
        return train_cfg.model.data.data_prefix

    data_dir = train_cfg.run.get("preprocessed_dir")
    alpha = train_cfg.run.get("blending_alpha")

    data_files = os.listdir(data_dir)
    lang_size = defaultdict(int)
    file_size = defaultdict(list)
    for f in data_files:
        if f.endswith(".bin"):
            f_path = os.path.join(data_dir, f)
            f_size = os.path.getsize(f_path)

            elements = f.split("_")
            lang = elements[0]
            lang_size[lang] += f_size
            file_size[lang].append((f_path.strip(".bin"), f_size))

    lang_ratio = {lang: math.pow(lang_size[lang], alpha) for lang in lang_size}
    total = sum(lang_ratio.values())
    lang_ratio = {lang: lang_ratio[lang] / total for lang in lang_ratio}

    res = []
    for lang in file_size:
        for prefix, size in file_size[lang]:
            res.extend([round(size / lang_size[lang] * lang_ratio[lang], 6), prefix])
    return res
