# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from .predictive_config_builder import generate, generate_recipe_configs, get_supported_models, list_models
from .pretraining import run_pretraining
from .utils import (
    extract_all_values,
    extract_gpu_specs_unified,
    get_args_file_path,
    get_supported_models,
    update_args_with_generation_metadata,
)

__all__ = [
    'generate_recipe_configs',
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
    'run_pretraining',
]
