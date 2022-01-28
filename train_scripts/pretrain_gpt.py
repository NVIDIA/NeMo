import os
import sys
import re
import socket
import time

import hydra


dgxa100_gpu2core = {0:'48-51,176-179', 1:'60-63,188-191', 2:'16-19,144-147', 3:'28-31,156-159', 4:'112-115,240-243', 5:'124-127,252-255', 6:'80-83,208-211', 7:'92-95,220-223'}
dgxa100_gpu2mem = {0:'3', 1:'3', 2:'1', 3:'1', 4:'7', 5:'7', 6:'5', 7:'5'}
rank2gpu = [0, 4, 2, 6, 1, 5, 3, 7]

def pause_and_prime_dns_connections() -> None:
    if int(os.environ.get("GROUP_RANK")) > 0:
        time.sleep(20)
        prime_dns_connections()
    elif int(os.environ.get("LOCAL_RANK")) != 0:
        time.sleep(10)

def prime_dns_connections() -> None:
    me = "worker"+os.environ.get("GROUP_RANK")+":"+os.environ.get("RANK")
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

@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        k, v = arg.split("=")
        if "splits_string" in k:
            args[index] = "{}={}".format(k, v.replace("'", "\\'"))

    train_args = [x.replace("training.", "") for x in args if x[:9] == "training."]
    train_args = [x.replace("None", "null") for x in train_args if "run." not in x]
    hydra_train_args = " ".join(train_args)

    bignlp_path = cfg.bignlp_path
    training_config = cfg.training_config
    nccl_topo_file = cfg.nccl_topology_xml_file
    if nccl_topo_file is not None:
        nccl_cmd = f"export NCCL_TOPO_FILE={nccl_topo_file}; "

    code_dir = "/opt/bignlp/NeMo"
    code_path = (
        f"{code_dir}/examples/nlp/language_modeling/megatron_gpt_pretraining.py"
    )
    training_config_path = os.path.join(bignlp_path, "conf/training")
    gpu_mapping = "CUDA_VISIBLE_DEVICES={}".format(re.sub('[\[\] ]', '', str(rank2gpu)))
    core_mapping = f"exec numactl --physcpubind={dgxa100_gpu2core[rank2gpu[int(os.environ.get('LOCAL_RANK'))]]} --membind={dgxa100_gpu2mem[rank2gpu[int(os.environ.get('LOCAL_RANK'))]]} -- "
    flags = f"--config-path={training_config_path} --config-name={training_config} "

    # W&B Api Key file.
    wandb_cmd = ""
    if cfg.wandb_api_key_file is not None:
        with open(cfg.wandb_api_key_file, "r") as f:
            wandb_api_key = f.readline().rstrip()
        wandb_cmd = f"wandb login {wandb_api_key}; "
    
    # Write command to launch training.
    cmd_prefix = f'{wandb_cmd} cd {code_dir}; git rev-parse HEAD; cd {code_dir}/nemo/collections/nlp/data/language_modeling/megatron; make; export PYTHONPATH="{code_dir}/.:$PYTHONPATH"; export TRANSFORMERS_CACHE="/temp_root/.cache/"; {nccl_cmd}'
    if cfg.cluster_type == "bcm":
        cmd = f'{cmd_prefix} {gpu_mapping} {core_mapping} python3 {code_path} {hydra_train_args} {flags}'
    elif cfg.cluster_type == "bcp":
        pause_and_prime_dns_connections()
        cmd = f'{cmd_prefix} {gpu_mapping} {core_mapping} python3 {code_path} +cluster_type=BCP +rank={os.environ.get("RANK")}  {hydra_train_args} {flags}'
    os.system(f"{cmd}")


if __name__ == "__main__":
    main()
