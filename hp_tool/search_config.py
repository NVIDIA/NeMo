import os

from hp_tool import utils
from hp_tool.base_config import calculate_model_size, generate_base_config
from hp_tool.training_config import search_training_config
from hp_tool.inference_config import search_inference_config


SUPPORTED_MODELS = ["gpt3", "t5", "mt5"]

def search_config(cfg):
    """
    Main function that implements the entire pipeline to search the optimal
    model config and launch the grid searches for both training and inference
    constraints.
    """
    model_type = cfg.get("search_config_value")
    model_name, model_size = model_type.split("/")
    assert (
        model_name in SUPPORTED_MODELS
    ), f"search_config must be set to one of {SUPPORTED_MODELS}/<model_size>"

    # Read config
    hp_cfg = cfg.get("search_config")
    cluster_cfg = cfg.get("cluster")
    train_cfg = hp_cfg.get("train_settings")
    nodes = train_cfg.get("num_nodes")
    gpus_per_node = train_cfg.get("gpus_per_node")
    max_training_days = train_cfg.get("max_training_days")
    max_minutes_per_run = train_cfg.get("max_minutes_per_run")
    model_size_in_b = train_cfg.get("model_size_in_b")
    tflops_per_gpu = train_cfg.get("tflops_per_gpu")
    num_tokens_in_b = train_cfg.get("num_tokens_in_b")

    gpu_count = nodes * gpus_per_node
    assert (
        isinstance(gpu_count, int) and gpu_count > 0
    ), "nodes * gpus_per_node must be an int larger than zero."
    assert (
        isinstance(max_minutes_per_run, int) and max_minutes_per_run >= 10
    ), "max_minutes_per_run must be an int and be at least 10 minutes."

    # Logging config
    log_dir = train_cfg.get("logs")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "candidate_configs"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "training_logs"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "final_result"), exist_ok=True)

    # Calculate model size
    model_size_in_b = calculate_model_size(
        gpu_count=gpu_count,
        max_training_days=max_training_days,
        model_size_in_b=model_size_in_b,
        tflops_per_gpu=tflops_per_gpu,
        num_tokens_in_b=num_tokens_in_b,
        model_name=model_name,
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
        model_name=model_name,
    )

    # Launch grid search for training constraints
    search_training_config(base_cfg, model_size_in_b, model_name, cfg)

    # Launch grid search for inference constraints
    search_inference_config(base_cfg=base_cfg, cfg=cfg)
