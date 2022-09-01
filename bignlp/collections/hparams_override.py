import os
import math
import hydra
import time
from omegaconf import OmegaConf
from collections import defaultdict
from nemo.utils.get_rank import is_global_rank_zero

@hydra.main(config_path="conf", config_name="hparams_override")
def hparams_override(cfg):
    hparams_file = cfg.get("hparams_file")
    if hparams_file is not None:
        output_path = cfg.get("output_path")
        hparams_override_file = os.path.join(output_path, "hparams_override.yaml")

        vocab_file = cfg.get("vocab_file")
        merge_file = cfg.get("merge_file")
        tokenizer_model = cfg.get("tokenizer_model")
        conf = OmegaConf.load(hparams_file)
        if vocab_file is not None:
            conf.cfg.tokenizer.vocab_file = vocab_file
        if merge_file is not None:
            conf.cfg.tokenizer.merge_file = merge_file
        if tokenizer_model is not None:
            conf.cfg.tokenizer.model = tokenizer_model
        if "activations_checkpoint_granularity" in conf.cfg:
            conf.cfg.activations_checkpoint_granularity = None
        if "activations_checkpoint_method" in conf.cfg:
            conf.cfg.activations_checkpoint_method = None
        if "sequence_parallel" in conf.cfg:
            conf.cfg.sequence_parallel = False
        if conf.cfg.optim.name == "distributed_fused_adam":
            conf.cfg.optim.name = "fused_adam"

        if is_global_rank_zero():
            with open(hparams_override_file, "w") as f:
                OmegaConf.save(config=conf, f=f)

        wait_time = 0
        while not os.path.exists(hparams_override_file):
            time.sleep(1)
            wait_time += 1
            if wait_time > 60:
                raise TimeoutError('Timeout waiting for config file to be created.')


if __name__ == "__main__":
    hparams_override()