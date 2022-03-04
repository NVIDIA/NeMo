import os

from hp_tool import utils
from hp_tool.base_config import calculate_model_size, generate_base_config
from hp_tool.training_config import search_training_config
from hp_tool.inference_config import search_inference_config


def search_config(cfg):
    """
    Main function that implements the entire pipeline to search the optimal
    model config and launch the grid searches for both training and inference
    constraints.
    """
    # Read config
    hp_cfg = cfg.search_config
    cluster_cfg = cfg.cluster
    train_cfg = hp_cfg.train_settings
    nodes = train_cfg.num_nodes
    gpus_per_node = train_cfg.gpus_per_node
    gpu_count = nodes * gpus_per_node
    max_training_days = train_cfg.max_training_days
    max_minutes_per_run = train_cfg.max_minutes_per_run
    model_size_in_b = train_cfg.model_size_in_b
    tflops_per_gpu = train_cfg.tflops_per_gpu
    num_tokens_in_b = train_cfg.num_tokens_in_b

    os.makedirs(cfg.base_results_dir, exist_ok=True)
    os.makedirs(cfg.search_config.train_settings.candidate_configs, exist_ok=True)
    os.makedirs(cfg.search_config.train_settings.candidate_logs, exist_ok=True)
    os.makedirs(cfg.search_config.train_settings.final_result_logs, exist_ok=True)
    assert max_minutes_per_run >= 10, "max_minutes_per_run must be at least 10 minutes"

    # Calculate model size
    model_size_in_b = calculate_model_size(
        gpu_count=gpu_count,
        max_training_days=max_training_days,
        model_size_in_b=model_size_in_b,
        tflops_per_gpu=tflops_per_gpu,
        num_tokens_in_b=num_tokens_in_b,
    )
    cfg.search_config.train_settings.model_size_in_b = model_size_in_b

    # Generate base config for the given model size
    base_cfg = generate_base_config(
        model_size_in_b=model_size_in_b,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        max_training_days=max_training_days,
        num_tokens_in_b=num_tokens_in_b,
        cfg=cfg,
    )

    # Launch grid search for training constraints
    #search_training_config(base_cfg, model_size_in_b, cfg)

    # Launch grid search for inference constraints
    search_inference_config(base_cfg=base_cfg, cfg=cfg)
