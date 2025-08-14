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
AutoTune module for NeMo LLM training optimization.

This module provides a clean Python API for generating, running, and analyzing
AutoTune configurations for NeMo pretraining.

Main API:
- AutoTuneArgs: Configuration class for all autotune parameters
- generate(): Generate all configurations for given args
- run(): Run pretraining for given args/configs  
- list_configs(): Display configuration table
- list_models(): Get supported models
- results(): Analyze performance results
"""

from nemo.collections.llm.tools.autotuner.args import AutoTuneArgs
from nemo.collections.llm.tools.autotuner.core.performance import results
from nemo.collections.llm.tools.autotuner.core.predictive_config_builder import generate, list_configs
from nemo.collections.llm.tools.autotuner.core.pretraining import run_pretraining

__all__ = [
    'run_pretraining',
    'list_configs',
    'generate',
    'results',
    'AutoTuneArgs',
]
