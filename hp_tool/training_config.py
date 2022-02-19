from hp_tool.generate_config import (
    calculate_model_size,
    generate_config_for_model_size,
    generate_grid_search_configs,
    launch_grid_search_configs,
    launch_throughput_measure,
)


def search_training_conig(base_cfg, model_size, cfg):
    base_dir, results_cfgs = generate_grid_search_configs(
        base_cfg, gbs, model_size, cfg
    )
    job_ids = launch_grid_search_configs(base_dir, results_cfgs, cfg)
    launch_throughput_measure(job_ids, model_size, cfg)
