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
