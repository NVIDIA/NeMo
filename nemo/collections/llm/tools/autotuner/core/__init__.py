"""
AutoTuner core module.

This module contains the core functionality for the AutoTuner system:
- predictive_config_builder: Configuration generation and validation
- performance: Results analysis and performance calculation
- display: Rich console output and tables
- utils: Utility functions and extraction logic
- pretraining: Training execution with nemo_run
"""

from .display import _display_configs_table, display_performance_analysis
from .performance import results
from .predictive_config_builder import (
    generate,
    generate_recipe_configs,
    get_supported_models,
    list_models,
    validate_all_configs,
)
from .pretraining import run_pretraining
from .utils import (
    check_config_matches,
    create_log_dir_name,
    extract_all_values,
    extract_gpu_specs_unified,
    get_args_file_path,
    get_supported_models,
    update_args_with_generation_metadata,
    validate_all_configs,
)

__all__ = [
    'generate_recipe_configs',
    'validate_all_configs',
    'get_supported_models',
    'generate',
    'list_models',
    'results',
    'display_performance_analysis',
    '_display_configs_table',
    'extract_gpu_specs_unified',
    'extract_all_values',
    'get_args_file_path',
    'update_args_with_generation_metadata',
    'create_log_dir_name',
    'check_config_matches',
    'run_pretraining',
]
