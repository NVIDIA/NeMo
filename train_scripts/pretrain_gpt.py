import os
import sys
import re

import hydra


dgxa100_gpu2core = {0:'48-51,176-179', 1:'60-63,188-191', 2:'16-19,144-147', 3:'28-31,156-159', 4:'112-115,240-243', 5:'124-127,252-255', 6:'80-83,208-211', 7:'92-95,220-223'}
dgxa100_gpu2mem = {0:'3', 1:'3', 2:'1', 3:'1', 4:'7', 5:'7', 6:'5', 7:'5'}
rank2gpu = [0, 4, 2, 6, 1, 5, 3, 7]


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        k, v = arg.split("=")
        if "splits_string" in k:
            args[index] = "{}={}".format(k, v.replace("'", "\\'"))

    train_args = [x.replace("training.", "") for x in args if x[:9] == "training."]
    train_args = [x.replace("None", "null") for x in train_args if "run." not in x and "slurm." not in x and "bcp." not in x]
    hydra_train_args = " ".join(train_args)

    bignlp_path = cfg["bignlp_path"]
    training_config = cfg["training_config"]
    code_dir = cfg["code_dir"]
    code_path = (
        f"{code_dir}/examples/nlp/language_modeling/megatron_gpt_pretraining.py"
    )
    training_config_path = os.path.join(bignlp_path, "conf/training")
    gpu_mapping = "CUDA_VISIBLE_DEVICES={}".format(re.sub('[\[\] ]', '', str(rank2gpu)))
    core_mapping = f"exec numactl --physcpubind={dgxa100_gpu2core[rank2gpu[int(os.environ.get('LOCAL_RANK'))]]} --membind={dgxa100_gpu2mem[rank2gpu[int(os.environ.get('LOCAL_RANK'))]]} -- "
    flags = f"--config-path={training_config_path} --config-name={training_config} "
    # cmd = f'cd /opt/bignlp/NeMo; git rev-parse HEAD; cd /opt/bignlp/NeMo/nemo/collections/nlp/data/language_modeling/megatron; make; export PYTHONPATH="/opt/bignlp/NeMo/.:$PYTHONPATH"; export TRANSFORMERS_CACHE="/temp_root/.cache/"; {gpu_mapping} {core_mapping} python3 {code_path} {hydra_train_args} {flags}'
    cmd = f'cd {code_dir}; git rev-parse HEAD; cd {code_dir}/nemo/collections/nlp/data/language_modeling/megatron; make; export PYTHONPATH="{code_dir}/.:$PYTHONPATH"; export TRANSFORMERS_CACHE="/temp_root/.cache/"; cp {bignlp_path}/megatron_gpt_pretraining.py {code_path}; python3 {code_path} +cluster_type=BCP {hydra_train_args} {flags}'
    print(f" Command is: {cmd}")
    os.system(f"{cmd}")

if __name__ == "__main__":
    main()
