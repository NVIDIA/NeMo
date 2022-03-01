import hydra
from omegaconf import OmegaConf

from hp_tool.search_config import search_config



@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # Read config
    cluster_cfg = cfg.cluster
    search_cfg = cfg.search_config

    search_config(cfg)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("multiply", lambda x, y: x*y)
    main()

