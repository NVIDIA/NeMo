import sys

import copy
import hydra
import omegaconf
import subprocess

# from data_preparation.pile_dataprep_scripts import data_preparation
from bignlp.train_scripts import train
from bignlp.conversion_scripts import convert
from bignlp.eval_scripts import evaluate

omegaconf.OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)


def convert_to_cli(cfg):
    result = ""
    for k, v in cfg.items():
        if isinstance(v, omegaconf.dictconfig.DictConfig):
            output = convert_to_cli(v).split(" ")
            result += " ".join([f"{k}.{x}" for x in output if x != ""]) + " "
        elif isinstance(v, omegaconf.listconfig.ListConfig):
            if k == "data_prefix":
                v = [x for x in v]  # Needed because of lazy omegaconf interpolation.
            result += f"{k}={str(v).replace(' ', '')} "
        elif isinstance(v, str) and "{" in v:
            continue
        elif k == "splits_string":
            result += f"{k}=\\'{v}\\' "
        elif k == "file_numbers":
            result += f"{k}=\\'{v}\\' "
        elif k == "checkpoint_name":
            v = v.replace('=', '\=')
            result += f"{k}=\'{v}\' "
        else:
            result += f"{k}={convert_to_null(v)} "
    return result


def convert_to_null(val):
    if val is None:
        return "null"
    return val


def fake_submit(*args, **kwargs):
    print(args, kwargs)
    fake_id = 123456
    return str(fake_id).encode()


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    hydra_args = " ".join(sys.argv[1:])
    hydra_args = convert_to_cli(cfg)

    if cfg.debug:
        subprocess.check_output = fake_submit

    # Read config
    run_data_preparation = cfg.run_data_preparation
    run_training = cfg.run_training
    run_conversion = cfg.run_conversion
    run_evaluation = cfg.run_evaluation

    # TODO: build a mapping from dataset name to modules
    if "pile" in cfg.data_config:
        from bignlp.data_preparation import data_preparation_pile as data_preparation
    elif "mc4" in cfg.data_config:
        from bignlp.data_preparation import data_preparation_mc4 as data_preparation
    else:
        raise ValueError(f"Unrecognized dataset in data config `{cfg.data_config}`.")

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

    if cfg.run_search_train:
        from bignlp.search_train_scripts.run_config_generator import search_config
        dependency = search_config(cfg, dependency=dependency)

    print(omegaconf.OmegaConf.to_yaml(cfg_copy))


if __name__ == "__main__":
    main()
