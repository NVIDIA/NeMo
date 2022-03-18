import os
import sys
import re
import socket
import time
import math
from collections import defaultdict

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
    print(f'SPDNS: {me} Connecting to {master_addr}:{master_port}')
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
            print(f'{me} couldnt connect to {master_addr}:{master_port} timed out! (300s)')
            sys.exit(110)
    print(f'SPDNS: {me} connected to {master_addr}:{master_port}')
    sock.close()


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
    cmd_prefix = f'{wandb_cmd} cd {code_dir}; git rev-parse HEAD; cd {code_dir}/nemo/collections/nlp/data/language_modeling/megatron; make; export PYTHONPATH="{code_dir}/.:$PYTHONPATH"; export TRANSFORMERS_CACHE="/temp_root/.cache/"; {nccl_cmd}'
    return cmd_prefix


def numa_mapping(dgxa100_gpu2core, dgxa100_gpu2mem):
    gpu_mapping = "CUDA_VISIBLE_DEVICES={}".format(re.sub('[\[\] ]', '', str(rank2gpu)))
    core_mapping = f"exec numactl --physcpubind={dgxa100_gpu2core[rank2gpu[int(os.environ.get('LOCAL_RANK'))]]} --membind={dgxa100_gpu2mem[rank2gpu[int(os.environ.get('LOCAL_RANK'))]]} -- "
    return gpu_mapping, core_mapping


def convert_args_to_hydra_train_args(args, prefix="training."):
    for index, arg in enumerate(args):
        k, v = arg.split("=", 1)
        if "splits_string" in k and "\\" not in v:
            args[index] = "{}={}".format(k, v.replace("'", "\\'"))

    train_args = [x.replace(prefix, "") for x in args if x.startswith(prefix)]
    train_args = [x.replace("None", "null") for x in train_args if "run." not in x]
    hydra_train_args = " ".join(train_args)
    return hydra_train_args


def generate_mt5_data_blend(cfg, alpha=0.3):
    train_cfg = cfg.training
    if train_cfg.model.data.data_prefix is not None:
        return train_cfg.model.data.data_prefix

    data_cfg = cfg.get("data_preparation")
    data_dir = data_cfg.get("preprocessed_dir")

    data_files = os.listdir(data_dir)
    lang_size = defaultdict(int)
    file_size = defaultdict(list)
    for f in data_files:
        if f.endswith(".bin"):
            f_path = os.path.join(data_dir, f)
            f_size = os.path.getsize(f_path)

            elements = f.split('_')
            lang = elements[0]
            lang_size[lang] += f_size
            file_size[lang].append((f_path.strip(".bin"), f_size))

    lang_ratio = {lang: math.pow(lang_size[lang], alpha) for lang in lang_size}
    total = sum(lang_ratio.values())
    lang_ratio = {lang: lang_ratio[lang] / total for lang in lang_ratio}

    res = []
    for lang in file_size:
        for prefix, size in file_size[lang]:
            res.extend([round(size / lang_size[lang] * lang_ratio[lang], 6),
                        prefix])
    return res
