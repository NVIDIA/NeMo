import os
import sys

import hydra


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        k, v = arg.split("=")
        if "splits_string" in k:
            args[index] = "{}={}".format(k, v.replace("'", "\\'"))

    train_args = [x.replace("training.", "") for x in args if x[:9] == "training."]
    train_args = [x.replace("None", "null") for x in train_args if "run." not in x and "slurm." not in x]
    hydra_train_args = " ".join(train_args)

    bignlp_path = cfg["bignlp_path"]
    training_config = cfg["training_config"]
    code_path = (
        "/opt/bignlp/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py"
    )
    training_config_path = os.path.join(bignlp_path, "conf/training")
    flags = f"--config-path={training_config_path} --config-name={training_config} "
    cmd = f'cd /opt/bignlp/NeMo; git rev-parse HEAD; cd /opt/bignlp/NeMo/nemo/collections/nlp/data/language_modeling/megatron; make; export PYTHONPATH="/opt/bignlp/NeMo/.:$PYTHONPATH"; export TRANSFORMERS_CACHE="/temp_root/.cache/"; CUDA_VISIBLE_DEVICES=0 python3 {code_path} {hydra_train_args} {flags}'
    os.system(f"{cmd}")


if __name__ == "__main__":
    main()
