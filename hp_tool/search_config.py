import os

from hp_tool import utils
from hp_tool.generate_config import calculate_model_size, generate_config_for_model_size, generate_grid_search_configs, launch_grid_search_configs, launch_throughput_measure 
from hp_tool.inference_config import search_inference_config


def search_config(cfg):
    hp_cfg = cfg.search_train_config
    slurm_cfg = hp_cfg.slurm
    train_settings_cfg = hp_cfg.train_settings

    nodes = slurm_cfg.nodes
    ntasks_per_node = slurm_cfg.ntasks_per_node
    gpu_count = nodes * ntasks_per_node

    max_training_days = train_settings_cfg.max_training_days
    max_mins_per_run = train_settings_cfg.max_mins_per_run
    assert max_mins_per_run >= 15, "max_mins_per_run must be at least 15 minutes"
    model_size = calculate_model_size(gpu_count, max_training_days, train_settings_cfg.model_size_in_b)

    base_cfg, gbs = generate_config_for_model_size(model_size=model_size, nodes=nodes, gpus_per_node=ntasks_per_node, max_training_days=max_training_days, cfg=cfg)
    
    # Inference constraints
    search_inference_config(base_cfg=base_cfg, cfg=cfg)

    # Store yaml files in tmp dir and then delete after best is found.
    base_dir, results_cfgs = generate_grid_search_configs(base_cfg, gbs, model_size, cfg)

    job_ids = launch_grid_search_configs(base_dir, results_cfgs, cfg)

    launch_throughput_measure(job_ids, model_size, cfg)
