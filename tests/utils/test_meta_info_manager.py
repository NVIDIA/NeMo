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
from unittest import mock

import pytest
from omegaconf import OmegaConf

from nemo.utils.meta_info_manager import MetaInfoManager


class TestMetaInfoManager:
    @pytest.mark.unit
    def test_init_with_none_config(self):
        """Test initialization with None config."""
        manager = MetaInfoManager()
        assert manager.cfg == OmegaConf.create({})

    @pytest.mark.unit
    def test_get_config_value(self):
        """Test getting config values with different access patterns."""
        config = OmegaConf.create({
            "model": {
                "batch_size": 32,
                "hidden_size": 768
            },
            "trainer": {
                "devices": 4,
                "num_nodes": 2
            }
        })
        manager = MetaInfoManager(config)

        # Test direct access
        assert manager._get_config_value("model.batch_size", 16) == 32
        assert manager._get_config_value("model.hidden_size", 512) == 768

        # Test non-existent path
        assert manager._get_config_value("model.nonexistent", 16) == 16

        # Test nested access
        assert manager._get_config_value("trainer.devices", 1) == 4

    @pytest.mark.unit
    def test_get_env(self):
        """Test getting environment variables."""
        manager = MetaInfoManager()
        
        # Test existing env var
        with mock.patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            assert manager._get_env("TEST_VAR", "default") == "test_value"

        # Test non-existent env var
        assert manager._get_env("NONEXISTENT_VAR", "default") == "default"

    @pytest.mark.unit
    def test_get_metadata_training(self):
        """Test metadata generation for training run type."""
        config = OmegaConf.create({
            "exp_manager": {
                "name": "test_experiment",
                "track_train_iterations": True,
                "track_test_iterations": True,
                "track_validation_iterations": True,
                "create_checkpoint_callback": True,
                "log_tflops_per_sec_per_gpu": True
            },
            "model": {
                "global_batch_size": 32,
                "micro_batch_size": 8,
                "seq_length": 128
            },
            "trainer": {
                "max_steps": 1000,
                "max_epochs": 10,
                "devices": 4,
                "num_nodes": 2,
                "log_every_n_steps": 50
            }
        })
        
        manager = MetaInfoManager(config)
        metadata = manager.get_metadata(run_type="training")

        # Test basic metadata
        assert metadata["session_tag"] == "test_experiment"
        assert metadata["model_name"] == "experiment"
        assert metadata["workload_type"] == "training"

        # Test training specific metadata
        assert metadata["global_batch_size"] == 32
        assert metadata["micro_batch_size"] == 8
        assert metadata["seq_length"] == 128
        assert metadata["max_steps"] == 1000
        assert metadata["max_epochs"] == 10
        assert metadata["log_every_n_iterations"] == 50

        # Test feature flags
        assert metadata["is_train_iterations_enabled"] is True
        assert metadata["is_test_iterations_enabled"] is True
        assert metadata["is_validation_iterations_enabled"] is True
        assert metadata["is_save_checkpoint_enabled"] is True
        assert metadata["is_log_throughput_enabled"] is True

    @pytest.mark.unit
    def test_get_metadata_with_env_vars(self):
        """Test metadata generation with environment variables."""
        manager = MetaInfoManager()
        
        env_vars = {
            "PERF_VERSION_TAG": "1.0.0",
            "WORLD_SIZE": "8",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "NODE_RANK": "0"
        }
        
        with mock.patch.dict(os.environ, env_vars):
            metadata = manager.get_metadata()
            
            assert metadata["perf_version_tag"] == "1.0.0"
            assert metadata["world_size"] == 8
            assert metadata["rank"] == "0"
            assert metadata["local_rank"] == "0"
            assert metadata["node_rank"] == "0" 