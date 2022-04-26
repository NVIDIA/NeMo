import os
import sys
import re
from shutil import copyfile

import hydra
import pandas as pd
from omegaconf import OmegaConf
from tensorboard.backend.event_processing import event_accumulator


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):
    bignlp_hp_tool_path = cfg.bignlp_hp_tool_path
    hp_cfg = cfg.search_config
    settings_cfg = hp_cfg.train_settings
    model_size = settings_cfg.model_size_in_b
    output_top_n = settings_cfg.output_top_n

    training_logs = os.path.join(settings_cfg.get("logs"), "training_logs")
    candidate_configs = os.path.join(settings_cfg.get("logs"), "candidate_configs")
    final_result_logs = os.path.join(settings_cfg.get("logs"), "final_result")

    result_models = []
    dirs = os.listdir(training_logs)
    for candidate_dir in dirs:
        config_path = os.path.join(candidate_configs, f"{candidate_dir}.yaml")
        config = OmegaConf.load(config_path)
        files = os.listdir(os.path.join(training_logs, candidate_dir))
        for f in files:
            if f[:6] == "events":
                event_file = os.path.join(training_logs, candidate_dir, f)
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                try:
                    timing_list = ea.Scalars("train_step_timing")
                    half_timing_list = timing_list[len(timing_list) // 2 :]
                    avg_global_step_time = calculate_average(half_timing_list)
                    result_models.append([candidate_dir, avg_global_step_time])
                finally:
                    continue

    result_models.sort(key=lambda x: x[-1])
    print(f"Top {output_top_n} configs sorted from fastest to slowest:")
    for i, (config, avg_time) in enumerate(result_models):
        print(f"Config #{i+1}: {config} with {avg_time:.4f}s per global step.")
        if i+1 == output_top_n:
            break

    print("\n==================================================")
    print(f"Optimal config: {result_models[0][0]} with {result_models[0][1]:.4f}s per global step.")
    print(f"Saving config to {final_result_logs}/optimal_config_{model_size}b.yaml.")
    print("==================================================\n")

    os.makedirs(final_result_logs, exist_ok=True)
    copyfile(
        os.path.join(candidate_configs, f"{result_models[0][0]}.yaml"),
        os.path.join(final_result_logs, f"optimal_config_{model_size}b.yaml"),
    )


def calculate_average(ea_timing):
    time_avg = 0.0
    length = len(ea_timing)
    for step_time in ea_timing:
        time_avg += step_time.value
    return time_avg / length


if __name__ == "__main__":
    main()
