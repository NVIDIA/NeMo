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
    hydra_args = convert_to_cli(cfg)
    search_config(cfg=cfg, hydra_args=hydra_args)


def convert_to_cli(cfg, root=True):
    result = []
    if cfg.get("search_config_value") is not None:
        result.append(f"search_config={cfg['search_config_value']}")

    for k, v in cfg.items():
        if k in ["training_container", "inference_container", "training_container_image", "inference_container_image"]:
            continue
        if isinstance(v, omegaconf.dictconfig.DictConfig):
            output = convert_to_cli(v, False)
            result.extend([f"{k}.{x}" for x in output if x != ""])
        elif isinstance(v, omegaconf.listconfig.ListConfig):
            result.append(f"{k}={str(v).replace(' ', '')}")
        elif isinstance(v, str) and "{" in v:
            continue
        else:
            result.append(f"{k}={convert_to_null(v)}")
    return " \\\n  ".join(result) if root else result

def convert_to_null(val):
    if val is None:
        return "null"
    return val


if __name__ == "__main__":
    main()
