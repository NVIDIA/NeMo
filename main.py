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
def main(cfg):
    hydra_args = " ".join(sys.argv[1:])
    search_config(cfg=cfg)


if __name__ == "__main__":
    main()
