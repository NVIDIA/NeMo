#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Dict, Any, Optional, List, Union
from omegaconf import OmegaConf


class MetaInfoManager:
    """
    Manager for abstracting metadata configuration across different systems.
    
    This class provides a standardized way to generate metadata from NeMo configs
    for use with various logging systems, monitoring tools, and other services.
    """
    
    def __init__(self, cfg=None):
        """
        Initialize the MetaInfoManager.
        
        Args:
            cfg: Configuration object (typically a NeMo hydra config)
        """
        self.cfg = cfg or OmegaConf.create({})
        
    def _get_config_value(self, path: str, default: Any) -> Any:
        """
        Safely extract a value from the config using dot notation path.
        
        Args:
            path: Dot-notation path to the config value (e.g., "model.batch_size")
            default: Default value if path doesn't exist
            
        Returns:
            The config value or default
        """
        if not self.cfg:
            return default
            
        parts = path.split('.')
        current = self.cfg
        
        try:
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                elif isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current
        except (AttributeError, KeyError):
            return default
            
    def _get_env(self, name: str, default: Any) -> Any:
        """Get environment variable with default value"""
        return os.environ.get(name, default)
            
    def get_metadata(self, 
                     run_type: str = "training", 
                     project_name: Optional[str] = None, 
                     run_suffix: str = "test",
                     **kwargs) -> Dict[str, Any]:
        """
        Generate standardized metadata for experiments based on config.
        
        Args:
            run_type: Type of run (e.g., "training", "inference", "evaluation")
            project_name: Project name (overrides config value if provided)
            run_suffix: Suffix to append to run name (e.g., "test", "prod")
            **kwargs: Additional metadata to include or override defaults
            
        Returns:
            Dictionary containing metadata
        """
        # Try to extract experiment name
        try:
            exp_name = self._get_config_value("exp_manager.name", "unnamed-experiment")
            model_name = str(exp_name).split("_")[1] if "_" in str(exp_name) else "model"
        except (AttributeError, IndexError):
            exp_name = "unnamed-experiment"
            model_name = "model"
        
        # Basic metadata common to all run types
        metadata = {
            # Run identification
            "app_tag": exp_name,
            "app_tag_run_name": f"{model_name}-{run_suffix}",
            "app_tag_run_version": "0.0.0",
            "app_run_type": run_type,
            
            # Project information
            "project_name": project_name or self._get_config_value("project_name", "default-project"),
            "run_name": exp_name,
            
            # Environment information
            "world_size": self._get_env("WORLD_SIZE", -1),
            "rank": self._get_env("RANK", "0"),
            "local_rank": self._get_env("LOCAL_RANK", "0"),
            "node_rank": self._get_env("NODE_RANK", "0"),
            
            # Schema versioning
            "metadata_schema_version": "1.0.0",
        }
        
        # Add run-type specific configuration
        if run_type == "training":
            metadata.update({
                # Batch size information
                "global_batch_size": self._get_config_value("model.global_batch_size", 1),
                "micro_batch_size": self._get_config_value("model.micro_batch_size", 1),
                
                # Training targets
                "max_steps": self._get_config_value("trainer.max_steps", 1),
                "max_epochs": self._get_config_value("trainer.max_epochs", None),
                "train_samples_target": (
                    self._get_config_value("trainer.max_steps", 1) *
                    self._get_config_value("model.global_batch_size", 1)
                ),
                
                # Logging frequency
                "log_every_n_steps": self._get_config_value("trainer.log_every_n_steps", 10),
                
                # Feature flags for training runs
                "precision": self._get_config_value("trainer.precision", "32"),
                "gradient_accumulation_steps": self._get_config_value(
                    "model.gradient_accumulation_steps", 
                    1
                ),
                
                # Checkpoint settings
                "save_checkpoint_enabled": True,
                "save_checkpoint_strategy": "steps",
            })
        elif run_type == "inference":
            metadata.update({
                # Inference specific settings
                "batch_size": self._get_config_value("model.batch_size", 1),
                "precision": self._get_config_value("model.precision", "32"),
            })
        elif run_type == "evaluation":
            metadata.update({
                # Evaluation specific settings
                "validation_ds": self._get_config_value("model.data.validation_ds.manifest_filepath", []),
                "test_ds": self._get_config_value("model.data.test_ds.manifest_filepath", []),
            })
            
        # Override with any provided kwargs
        metadata.update(kwargs)
        
        return metadata
        



