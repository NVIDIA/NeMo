import sys
import os
import subprocess

import hydra
import omegaconf

from data_preparation import data_preparation
from train_scripts import train
from eval_scripts import evaluate


def convert_to_absolute_path(cfg):
    base = cfg.bignlp_path

    data_cfg = cfg.data_preparation
    for k, v in data_cfg.items():
        if "_dir" in k and v is not None and v[0] != "/":
            data_cfg[k] = os.path.join(base, v)

    train_cfg = cfg.training
    
    run_cfg = train_cfg.run
    for k, v in run_cfg.items():
        if "log_dir" in k and v is not None and v[0] != "/":
            run_cfg[k] = os.path.join(base, v)
    
    exp_manager_cfg = train_cfg.exp_manager
    for k, v in exp_manager_cfg.items():
        if "_dir" in k and v is not None and v[0] != "/":
            exp_manager_cfg[k] = os.path.join(base, v)

    model_cfg = train_cfg.model
    for k, v in model_cfg.items():
        if k == "tokenizer":
            for k2, v2 in v.items():
                if "_file" in k2 and v2 is not None and v2[0] != "/":
                    model_cfg[k][k2] = os.path.join(base, v2)
        if k == "data":
            for k2, v2 in v.items():
                if k2 == "data_prefix" and v2 is not None:
                    for index, elem in enumerate(v2):
                        if isinstance(elem, str) and elem[0] != "/":
                            v2[index] = os.path.join(base, elem)
        if "_path" in k and v is not None and v[0] != "/":
            model_cfg[k] = os.path.join(base, v)
    return cfg

def convert_to_cli(cfg):
    result = ""
    for k, v in cfg.items():
        if isinstance(v, omegaconf.dictconfig.DictConfig):
            output = convert_to_cli(v).split(" ")
            result += " ".join([f"{k}.{x}" for x in output if x != ""]) + " "
        elif isinstance(v, omegaconf.listconfig.ListConfig):
            result += f"{k}={str(v).replace(' ', '')} "
        elif isinstance(v, str) and "{" in v:
            continue
        elif k == "splits_string":
            result += f"{k}=\\'{v}\\' "
        else:
            result += f"{k}={v} "
    return result



@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    hydra_args = " ".join(sys.argv[1:])
    cfg = convert_to_absolute_path(cfg)
    hydra_args = convert_to_cli(cfg)

    # Read config
    run_data_preparation = cfg.get("run_data_preparation")
    run_training = cfg.get("run_training")
    run_evaluation = cfg.get("run_evaluation")


    dependency = None
    if run_data_preparation:
        dependency = data_preparation.run_data_preparation(cfg, hydra_args=hydra_args, dependency=dependency)

    if run_training:
        dependency = train.run_training(cfg, hydra_args=hydra_args, dependency=dependency)

    if run_evaluation:
        dependency = evaluate.run_evaluation(cfg, dependency=dependency)


if __name__ == "__main__":
    main()
