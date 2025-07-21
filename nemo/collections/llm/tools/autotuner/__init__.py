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

from .args import AutoTuneArgs
from .runner import generate, run, list_configs, list_models, results

__all__ = [
    'AutoTuneArgs',
    'generate', 
    'run',
    'list_configs',
    'list_models', 
    'results'
] 