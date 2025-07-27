# Usage note for developers:
# This module provides the main API entrypoints for the autotuner backend.
# Use the following functions:
#   generate(args: AutoTuneArgs) -> dict
#   run(args: AutoTuneArgs) -> dict
#   list_configs(config_dir: str, model_name: Optional[str]) -> None
#   list_models() -> list NeMo supported models
#   results(analysis_data: dict) -> None

import os
import json
import base64
import pickle
import logging
from typing import Dict, Any

from nemo.collections.llm.tools.autotuner.core.predictive_config_builder import generate as generate_impl, list_models as list_models_impl, list_configs as list_configs_impl
from nemo.collections.llm.tools.autotuner.core.pretraining import run_pretraining
from nemo.collections.llm.tools.autotuner.core.display import display_performance_analysis
from nemo.collections.llm.tools.autotuner.core.performance import results as performance_results

logger = logging.getLogger(__name__)

from nemo.collections.llm.tools.autotuner.args import AutoTuneArgs

def generate(args: 'AutoTuneArgs') -> dict:
    """
    Generate all configs for the given args. Returns a result dict with base_config, configs, runner, etc.
    """
    result = generate_impl(**args.to_dict())
    return result

def run(args: 'AutoTuneArgs') -> dict:
    """
    Run pretraining for the given args/configs. Returns a summary dict with run status.
    """
    return run_pretraining(
        base_config=args.get_base_config(),
        configs=args.metadata.get('configs', {}),
        base_config_matches=args.metadata.get('base_config_matches', []),
        sequential=args.sequential,
        executor_config=args.get_executor_config(),
        memory_analysis=args.get_memory_analysis(),
        run_all=args.metadata.get('run_all', False)
    )

def list_configs(config_dir: str, model_name: str = None) -> None:
    """
    Display a config table for a directory using rich output.
    """
    list_configs_impl(config_dir, model_name)

def list_models() -> list:
    """
    Return a list of supported models.
    """
    list_models_impl()

def results(args: 'AutoTuneArgs', logs_path: str, log_prefix: str = '', top_n: int = 5, force_reconstruct: bool = False, cost_per_node_hour: float = 3.0, quiet: bool = False) -> Dict[str, Any]:
    """
    Collect, analyze, and display AutoConfigurator results in one step.
    Returns a dict with performance_dict and analysis_data.
    """
    return performance_results(args, logs_path, log_prefix, top_n, force_reconstruct, cost_per_node_hour, quiet)
