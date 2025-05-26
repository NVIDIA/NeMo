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

"""MetaInfoManager module for handling experiment metadata configuration."""

import os
from typing import Any, Dict, Optional

import nemo_run.config as run
from omegaconf import OmegaConf


class MetaInfoManager:
    """Manager for abstracting metadata configuration across different systems.

    This class provides a standardized way to generate metadata from NeMo configs
    for use with various logging systems, monitoring tools, and other services.
    """

    def __init__(self, cfg=None):
        """Initialize the MetaInfoManager.

        Args:
            cfg: Configuration object (typically a NeMo hydra config)
        """
        self.cfg = cfg or OmegaConf.create({})

    def _get_config_value(self, path: str, default: Any) -> Any:
        """Safely extract a value from the config using dot notation path.

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
                # Check if current is a run.Partial object
                if isinstance(current, run.Partial):
                    if part in current.keywords:
                        current = current.keywords[part]
                    else:
                        return default
                # Check for regular attribute access
                elif hasattr(current, part):
                    current = getattr(current, part)
                # Check for dictionary access
                elif isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current
        except (AttributeError, KeyError):
            return default

    def _get_env(self, name: str, default: Any) -> Any:
        """Get environment variable with default value.

        Args:
            name: Name of the environment variable
            default: Default value if environment variable is not set

        Returns:
            Value of environment variable or default
        """
        return os.environ.get(name, default)

    def get_metadata(
        self, run_type: str = "training", project_name: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Generate standardized metadata for experiments based on config.

        Args:
            run_type: Type of run (e.g., "training", "inference", "evaluation")
            project_name: Project name (overrides config value if provided)
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
            "session_tag": exp_name,
            "model_name": model_name,
            "perf_version_tag": self._get_env("PERF_VERSION_TAG", "0.0.0"),
            "workload_type": run_type,
            # Project information
            "app_name": project_name or self._get_config_value("project_name", "default-project"),
            # Environment information - check config first, then fallback to env vars
            "world_size": (
                self._get_config_value(
                    "trainer.world_size",
                    self._get_config_value("trainer.devices", -1)
                    * self._get_config_value("trainer.num_nodes", 1),
                )
                or self._get_env("WORLD_SIZE", -1)
            ),
            "rank": self._get_config_value("trainer.global_rank", None) or self._get_env("RANK", "0"),
        }

        # Add run-type specific configuration
        if run_type == "training":
            metadata.update(
                {
                    # Batch size information
                    "enable_for_current_rank": (
                        int(metadata.get('rank', 0)) == int(metadata.get('world_size', 1)) - 1
                    ),
                    "global_batch_size": self._get_config_value("model.global_batch_size", 1),
                    "micro_batch_size": self._get_config_value("model.micro_batch_size", 1),
                    "seq_length": self._get_config_value("model.seq_length", 1),
                    "train_iterations_target": self._get_config_value("trainer.max_steps", 1),
                    # Training targets
                    "max_steps": self._get_config_value("trainer.max_steps", 1),
                    "max_epochs": self._get_config_value("trainer.max_epochs", None),
                    "train_samples_target": (
                        self._get_config_value("trainer.max_steps", 1)
                        * self._get_config_value("model.global_batch_size", 1)
                    ),
                    # Logging frequency
                    "log_every_n_iterations": self._get_config_value("trainer.log_every_n_steps", 10),
                    "save_checkpoint_strategy": self._get_config_value(
                        "trainer.save_checkpoint_strategy", "async"
                    ),
                    # Construct perf_tag as a string with safely accessed variables
                    "perf_tag": (
                        f"{exp_name}_{metadata.get('perf_version_tag', '0.0.0')}_"
                        f"{self._get_config_value('model.global_batch_size', 1)}_"
                        f"{metadata.get('world_size', 1)}"
                    ),
                    # Feature flags - get from config when available
                    "is_train_iterations_enabled": self._get_config_value(
                        "exp_manager.track_train_iterations", True
                    ),
                    "is_test_iterations_enabled": self._get_config_value(
                        "exp_manager.track_test_iterations", True
                    ),
                    "is_validation_iterations_enabled": self._get_config_value(
                        "exp_manager.track_validation_iterations", True
                    ),
                    "is_save_checkpoint_enabled": self._get_config_value(
                        "exp_manager.create_checkpoint_callback", True
                    ),
                    "is_log_throughput_enabled": self._get_config_value(
                        "exp_manager.log_tflops_per_sec_per_gpu", True
                    ),
                }
            )

        # Override with any provided kwargs
        metadata.update(kwargs)

        return metadata
