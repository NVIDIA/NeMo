"""Entry point, main file to run to launch jobs with the HP tool."""

import sys
import math

import hydra
import omegaconf
from omegaconf import OmegaConf

from hp_tool.search_config import search_config


OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("divide_ceil", lambda x, y: int(math.ceil(x / y)), replace=True)
OmegaConf.register_new_resolver("divide_floor", lambda x, y: int(x // y), replace=True)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: omegaconf.dictconfig.DictConfig) -> None:
    """
    Main function in the entire pipeline, it reads the config using hydra and calls search_config.

    :param omegaconf.dictconfig.DictConfig cfg: OmegaConf object, read using the @hydra.main decorator.
    :return: None
    """
    hydra_args = " ".join(sys.argv[1:])
    search_config(cfg=cfg)


if __name__ == "__main__":
    main()
