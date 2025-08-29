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
AutoTune utility functions

This module contains ALL extraction functions in one place to eliminate duplication:
- Model size, precision, and parameter extraction
- GPU type and memory specifications  
- Configuration parsing from strings, objects, and names
- Hardware resource parsing
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console

from nemo.collections import llm
from nemo.collections.llm.tools.autotuner.args import AutoTuneArgs

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

console = Console()

# ===================== PATTERNS =====================


class ExtractionPatterns:
    """Centralized regex patterns and data for all extraction operations."""

    # GPU resource patterns
    GPU_RESOURCE_PATTERNS = [
        r'gpu\.(\d+)x([a-zA-Z0-9\-]+)',  # gpu.8xh200, gpu.4xh100, gpu.2xa100-40gb
        r'gpu\.([a-zA-Z0-9\-]+)\.(\w+)',  # gpu.a10.6xlarge
        r'gpu\.([a-zA-Z0-9\-]+)',  # gpu.a10, gpu.a100-40gb, gpu.h100-sxm
        r'(\d+)x([a-zA-Z0-9\-]+)',  # 8xh200, 4xh100, 2xa100-40gb
        r'(\d+)x?',  # Just count: 8x, 8
    ]

    # Config name parsing patterns (format: model_8nodes_tp_2_pp_1_cp_1_ep_1_mbs_1_vp_None_seq_8192_gbs_64)
    CONFIG_NAME_PATTERNS = {
        'nodes': r'(\d+)nodes_',
        'tp': r'tp_(\d+)_',
        'pp': r'pp_(\d+)_',
        'cp': r'cp_(\d+)_',
        'ep': r'ep_(\d+)_',
        'vp': r'vp_(\w+?)_',
        'mbs': r'mbs_(\d+)_',
        'gbs': r'gbs_(\d+)(?:_|$)',
        'seq_length': r'seq_(\d+)_',
    }

    GPU_MEMORY_SPECS = {"h100": 80, "h200": 141, "a100": 80, "v100": 32, "l40s": 48, "gb200": 192, "b200": 180}


# ===================== CORE EXTRACTION FUNCTIONS =====================


def extract_all_values(config_name: str) -> Dict[str, Any]:
    config_values = {}
    for key, pattern in ExtractionPatterns.CONFIG_NAME_PATTERNS.items():
        if key == 'vp':
            vp_val = extract_value_with_patterns(config_name, [pattern], str)
            config_values['vp'] = None if not vp_val or vp_val.lower() == 'none' else int(vp_val)
        else:
            extracted = extract_value_with_patterns(config_name, [pattern], int)
            if extracted is not None:
                config_values[key] = extracted
    return config_values


def extract_value_with_patterns(text: str, patterns: List[str], convert_type: type = str, default: Any = None) -> Any:
    """Extract value using multiple regex patterns with type conversion."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = match.group(1)
                if convert_type == bool:
                    return value.lower() == 'true'
                elif convert_type == int:
                    return int(value)
                elif convert_type == float:
                    return float(value)
                else:
                    return value
            except (ValueError, IndexError):
                continue
    return default


def extract_gpu_specs_unified(resource_shape: str, memory_per_gpu: Optional[float] = None) -> Tuple[str, int, float]:
    """
    Unified GPU specification extraction.

    Args:
        resource_shape: Resource shape string like "gpu.8xh200", "gpu.4xh100", etc.
        memory_per_gpu: Optional custom memory per GPU in GB

    Returns:
        Tuple of (gpu_type, gpu_count, memory_per_gpu_gb)
    """
    gpu_type = "h100"
    gpu_count = 8

    for pattern in ExtractionPatterns.GPU_RESOURCE_PATTERNS:
        match = re.search(pattern, resource_shape.lower())
        if match:
            if len(match.groups()) >= 2:
                gpu_count = int(match.group(1))
                gpu_type = match.group(2).lower()
                break
            elif len(match.groups()) == 1:
                gpu_count = int(match.group(1))
                break

    if memory_per_gpu is not None:
        memory_gb = memory_per_gpu
        logger.info(f"Using custom GPU memory: {memory_per_gpu}GB")
    elif gpu_type in ExtractionPatterns.GPU_MEMORY_SPECS:
        memory_gb = ExtractionPatterns.GPU_MEMORY_SPECS[gpu_type]
    else:
        memory_gb = 80.0
        logger.warning(f"Unknown GPU type '{gpu_type}', defaulting to 80GB")

    return gpu_type, gpu_count, memory_gb


# ===================== MODEL SUPPORT FUNCTIONS =====================


def get_supported_models() -> List[str]:
    """Get list of supported models from NeMo's llm module."""
    supported_models = []
    try:
        for attr_name in dir(llm):
            if not attr_name.startswith("_"):
                attr = getattr(llm, attr_name)
                if hasattr(attr, "pretrain_recipe"):
                    supported_models.append(attr_name)
    except Exception as e:
        logger.warning(f"Error getting supported models: {e}")
    return sorted(supported_models)


# ===================== UTILITY FUNCTIONS =====================


def get_args_file_path(model, config_dir):
    """Get the standard path for the args file."""
    return os.path.join(config_dir, model, "args.json")


def update_args_with_generation_metadata(model_name, result, config_dir):
    """Update the args.json file with generation metadata."""
    args_file_path = get_args_file_path(model_name, config_dir)
    args = AutoTuneArgs.load_from_file(args_file_path)
    args.save_with_metadata(args_file_path, result)
    return args_file_path


def _load_args_from_config_dir(config_dir, model=None):
    """Load AutoTuneArgs from a config directory, optionally for a specific model."""
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    if model is None:
        subdirs = [d for d in os.listdir(config_dir) if os.path.isdir(os.path.join(config_dir, d))]
        if not subdirs:
            raise FileNotFoundError(f"No model directories found in: {config_dir}")
        model = subdirs[0]
    args_file_path = get_args_file_path(model, config_dir)
    if not os.path.exists(args_file_path):
        raise FileNotFoundError(f"Args file not found: {args_file_path}")
    return AutoTuneArgs.load_from_file(args_file_path)


def update_args_with_performance_results(model_name, performance_dict, config_dir):
    """Update the args.json file with performance results."""
    try:
        args_file_path = get_args_file_path(model_name, config_dir)
        if os.path.exists(args_file_path):
            args = AutoTuneArgs.load_from_file(args_file_path)
            # Update with performance results and save
            args.update_performance_results(performance_dict)
            args.save_to_file(args_file_path)
            logger.info(f"Performance results saved to {args_file_path}")
        else:
            logger.warning(f"Args file not found: {args_file_path}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to read args file: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to update performance results: {e}")
        raise
