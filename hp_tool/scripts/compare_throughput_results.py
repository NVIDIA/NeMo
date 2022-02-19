import os
import sys
import re
from shutil import copyfile

import hydra
import pandas as pd
from omegaconf import OmegaConf
from tensorboard.backend.event_processing import event_accumulator


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    # model_size = sys.argv[1]

    bignlp_path = cfg.bignlp_path
    hp_cfg = cfg.search_train_config
    settings_cfg = hp_cfg.settings
    model_size = settings_cfg.model_size_in_b
    candidate_log_dir = os.path.join(settings_cfg.candidate_logs, f"{model_size}b")
    candidate_config_dir = os.path.join(
        settings_cfg.candidate_configs, f"{model_size}b"
    )

    min_avg_time = float("inf")
    best_model = []
    dirs = os.listdir(candidate_log_dir)
    for candidate_dir in dirs:
        config_path = os.path.join(candidate_config_dir, f"{candidate_dir}.yaml")
        config = OmegaConf.load(config_path)
        files = os.listdir(os.path.join(candidate_log_dir, candidate_dir))
        for f in files:
            if f[:6] == "events":
                event_file = os.path.join(candidate_log_dir, candidate_dir, f)
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                timing_list = ea.Scalars("train_step_timing")
                half_timing_list = timing_list[len(timing_list) // 2 :]
                avg_step_time = calculate_average(half_timing_list)
                grad_accumul_steps = config.trainer.accumulate_grad_batches
                avg_global_step_time = avg_step_time * grad_accumul_steps
                if avg_global_step_time < min_avg_time:
                    min_avg_time = avg_global_step_time
                    best_model = [candidate_dir, min_avg_time]
                print(
                    f"Config {candidate_dir} achieves {avg_global_step_time}s per global step."
                )

    print("\n==================================================")
    print(f"Optimal config: {best_model[0]} with {best_model[1]}s per global step.")
    print(f"Saving config to search_train_config/optimal_config_{model_size}b.yaml.")
    print("==================================================\n")

    copyfile(
        os.path.join(candidate_config_dir, f"{best_model[0]}.yaml"),
        os.path.join(
            f"{bignlp_path}/search_train_config/optimal_config_{model_size}b.yaml"
        ),
    )


def calculate_average(ea_timing):
    time_avg = 0.0
    length = len(ea_timing)
    for step_time in ea_timing:
        time_avg += step_time.value / length
    return time_avg


if __name__ == "__main__":
    main()
