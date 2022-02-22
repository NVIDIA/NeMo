import sys

import copy
import math
import hydra
import omegaconf
import subprocess

from bignlp.bignlp_utils import convert_to_cli, fake_submit
from bignlp.train_scripts import train
from bignlp.conversion_scripts import convert
from bignlp.finetune_scripts import finetune
from bignlp.eval_scripts import evaluate

omegaconf.OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
omegaconf.OmegaConf.register_new_resolver("divide_ceil", lambda x, y: int(math.ceil(x / y)), replace=True)


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
    run_finetuning = cfg.run_finetuning
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

    if run_finetuning:
        dependency = finetune.run_finetuning(cfg, hydra_args=hydra_args, dependency=dependency)
    else:
        cfg_copy._content.pop("finetuning")

    if run_evaluation:
        dependency = evaluate.run_evaluation(cfg, dependency=dependency)
    else:
        cfg_copy._content.pop("evaluation")

    print(omegaconf.OmegaConf.to_yaml(cfg_copy))


if __name__ == "__main__":
    main()
