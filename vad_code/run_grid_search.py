import json
import os
from pathlib import Path
from pprint import pprint
from typing import List, Tuple, Union

import torch
from omegaconf import OmegaConf
from src.vad_utils import vad_tune_threshold_on_dev

from nemo.core.config import hydra_runner
from nemo.utils import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@hydra_runner(config_path="./configs", config_name="post_processing_grid.yaml")
def main(cfg):
    params = OmegaConf.to_container(cfg.params, resolve=True)
    pred_frames_dir = cfg.pred_dir
    gt_frames_dir = cfg.gt_dir
    num_workders = cfg.num_workers
    best_params, optimal_scores = vad_tune_threshold_on_dev(
        params, pred_frames_dir, gt_frames_dir, num_workers=num_workders
    )
    pprint(best_params)
    pprint(optimal_scores)


if __name__ == "__main__":
    main()
