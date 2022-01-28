import sys
import os
import subprocess

import copy
import hydra
import omegaconf

from data_preparation import data_preparation
from train_scripts import train
from conversion_scripts import convert
from eval_scripts import evaluate


omegaconf.OmegaConf.register_new_resolver("multiply", lambda x, y: x*y)

def convert_to_cli(cfg):
    result = ""
    for k, v in cfg.items():
        if isinstance(v, omegaconf.dictconfig.DictConfig):
            output = convert_to_cli(v).split(" ")
            result += " ".join([f"{k}.{x}" for x in output if x != ""]) + " "
        elif isinstance(v, omegaconf.listconfig.ListConfig):
            if k == "data_prefix":
                v = [x for x in v] # Needed because of lazy omegaconf interpolation.
            result += f"{k}={str(v).replace(' ', '')} "
        elif isinstance(v, str) and "{" in v:
            continue
        elif k == "splits_string":
            result += f"{k}=\\'{v}\\' "
        elif k == "file_numbers":
            result += f"{k}=\\'{v}\\' "
        else:
            result += f"{k}={convert_to_null(v)} "
    return result


def convert_to_null(val):
    if val is None:
        return "null"
    return val


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    hydra_args = " ".join(sys.argv[1:])
    hydra_args = convert_to_cli(cfg)

    # Read config
    run_data_preparation = cfg.run_data_preparation
    run_training = cfg.run_training
    run_conversion = cfg.run_conversion
    run_evaluation = cfg.run_evaluation

    cfg_copy = copy.deepcopy(cfg)
    dependency = None
    if run_data_preparation:
        dependency = data_preparation.run_data_preparation(cfg, hydra_args=hydra_args, dependency=dependency)
    else:
        cfg_copy._content.pop("data_preparation")

    if run_training:
        dependency = train.run_training(cfg, hydra_args=hydra_args, dependency=dependency)
    else:
        cfg_copy._content.pop("training")

    if run_conversion:
        dependency = convert.convert_ckpt(cfg, hydra_args=hydra_args, dependency=dependency)
    else:
        cfg_copy._content.pop("conversion")

    if run_evaluation:
        dependency = evaluate.run_evaluation(cfg, dependency=dependency)
    else:
        cfg_copy._content.pop("evaluation")

    print(omegaconf.OmegaConf.to_yaml(cfg_copy))


if __name__ == "__main__":
    main()
