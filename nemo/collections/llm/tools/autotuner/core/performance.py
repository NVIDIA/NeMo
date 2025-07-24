import os
import logging
import datetime
from typing import Dict, Any, Optional
from nemo.collections.llm.tools.autotuner.args import AutoTuneArgs
from nemo.collections.llm.tools.autotuner.core.predictive_config_builder import generate_recipe_configs
from nemo.collections.llm.tools.autotuner.core.display import display_performance_analysis
from nemo.collections.llm.tools.autotuner.core.utils import extract_all_values, _load_args_from_config_dir, update_args_with_performance_results, get_results

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def results(args: AutoTuneArgs, logs_path: str, log_prefix: str = '', top_n: int = 5, force_reconstruct: bool = False, cost_per_node_hour: float = 24.0, quiet: bool = False) -> Dict[str, Any]:
    """
    Collect, analyze, and display AutoConfigurator results in one step.
    Returns a dict with performance_dict and analysis_data.
    """
    if not os.path.exists(logs_path):
        logger.error(f"Logs directory not found: {logs_path}")
        raise FileNotFoundError(f"Logs directory not found: {logs_path}")

    # Load or reconstruct objects
    if not force_reconstruct and args.has_valid_objects():
        base_config = args.get_base_config()
        runner = args.get_runner()
        metadata = args.metadata
    else:
        config_result = generate_recipe_configs(args)
        base_config = config_result['base_config']
        runner = config_result['runner']
        metadata = args.metadata

    logger.info(f"Collecting AutoTune results from: {logs_path}")
    logger.info(f"Loaded configuration for model: {args.model}")
    logger.info(f"Resources: {args.nodes} nodes × {args.gpus_per_node} GPUs = {args.nodes * args.gpus_per_node} total GPUs")
    logger.info(f"Resource shape: {args.resource_shape}")
    logger.info(f"Batch sizes: micro={args.micro_batch_sizes}, global={args.global_batch_sizes}")
    logger.info(f"Sequence length: {args.seq_length}")
    logger.info(f"Training: max_steps={args.max_steps}, val_check_interval={args.val_check_interval}")
    
    performance_dict = get_results(
        base_config=base_config,
        train_config=runner,
        path_to_save=logs_path,
        output_top_n=top_n,
        log_file_prefix=log_prefix,
    )

    logger.info(f"Results collection completed. Total configs: {metadata.get('num_configs_generated')}, Base config matches: {len(metadata.get('base_config_matches', []))}")

    if performance_dict:
        update_args_with_performance_results(args.model, performance_dict, args.config_dir)
        logger.info("Performance results saved to args.json.")

    # --- Performance Analysis and Display ---
    analysis_data = None
    if performance_dict:
        total_tokens = args.num_tokens_in_b * 1_000_000_000
        # total_tokens =  15000 * 1_000_000_000
        analysis_data = calculate_performance_analysis(performance_dict, args, total_tokens, cost_per_node_hour)
        display_performance_analysis(analysis_data)

    return {
        'performance_dict': performance_dict,
        'analysis_data': analysis_data,
        'metadata': metadata
    }

def calculate_performance_analysis(performance_dict, args, total_tokens, cost_per_node_hour):
    """
    Calculate performance and cost analysis for all configurations.
    Returns a dict with config_analysis, sorted_configs, and args.
    """
    if not performance_dict:
        return None
    print(f"cost_per_node_hour {cost_per_node_hour}")
    total_gpus = args.nodes * args.gpus_per_node
    config_analysis = {}
    for config_name, config_data in performance_dict.items():
        time_per_step = config_data.get('time_per_global_step', 0)
        m_tflops_gpu = config_data.get('m_tflops_gpu', 0)
        extracted_values = extract_all_values(config_name)
        gbs = extracted_values.get('gbs')
        if gbs is None:
            gbs = args.global_batch_sizes[0] if args.global_batch_sizes else 512
        tokens_per_step = args.seq_length * gbs
        total_steps = total_tokens / tokens_per_step
        total_training_time_seconds = time_per_step * total_steps
        total_training_time_hours = total_training_time_seconds / 3600
        total_cost = total_training_time_hours * cost_per_node_hour * args.nodes
        config_analysis[config_name] = {
            **config_data,
            'gbs': gbs,
            'tokens_per_step': tokens_per_step,
            'total_steps': total_steps,
            'total_training_time_hours': total_training_time_hours,
            'total_training_time_days': total_training_time_hours / 24,
            'total_cost': total_cost,
            'cost_per_tflop': total_cost / (m_tflops_gpu * total_gpus) if m_tflops_gpu > 0 else float('inf')
        }
    sorted_configs = sorted(
        config_analysis.items(),
        key=lambda x: x[1].get('total_training_time_hours', float('inf')),
        reverse=False
    )
    return {
        'config_analysis': config_analysis,
        'sorted_configs': sorted_configs,
        'args': args
    }