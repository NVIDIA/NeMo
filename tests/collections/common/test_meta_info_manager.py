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
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from nemo.utils.meta_info_manager import (
    _get_base_callback_config,
    _should_enable_for_current_rank,
    get_nemo_v1_callback_config,
    get_nemo_v2_callback_config,
    get_onelogger_init_config,
)


class TestMetaInfoManager:
    """Test cases for meta_info_manager module functions."""

    @pytest.mark.unit
    def test_get_onelogger_init_config(self):
        """Test get_onelogger_init_config returns correct minimal configuration."""
        with patch.dict(os.environ, {"SLURM_JOB_NAME": "test_job", "WORLD_SIZE": "4"}):
            config = get_onelogger_init_config()

            assert isinstance(config, dict)
            assert config["application_name"] == "nemo-application"
            assert config["session_tag_or_fn"] == "test_job"
            assert "enable_for_current_rank" in config
            assert config["world_size_or_fn"] == 4
            assert config["error_handling_strategy"] == "propagate_exceptions"

    @pytest.mark.unit
    def test_get_onelogger_init_config_no_slurm(self):
        """Test get_onelogger_init_config when SLURM_JOB_NAME is not set."""
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True):
            config = get_onelogger_init_config()

            assert config["session_tag_or_fn"] == "nemo-run"
            assert config["world_size_or_fn"] == 1

    @pytest.mark.unit
    def test_get_base_callback_config(self):
        """Test _get_base_callback_config with basic trainer setup."""
        trainer = MagicMock()
        trainer.max_steps = 1000
        trainer.callbacks = []
        trainer.val_check_interval = 1.0
        trainer.strategy = None
        trainer.log_every_n_steps = 10

        with patch.dict(os.environ, {"SLURM_JOB_NAME": "test_job", "WORLD_SIZE": "4", "PERF_VERSION_TAG": "1.0.0"}):
            config = _get_base_callback_config(trainer=trainer, global_batch_size=32, seq_length=512)

            assert config["perf_tag_or_fn"] == "test_job_1.0.0_bf32_se512_ws4"
            assert config["global_batch_size_or_fn"] == 32
            assert config["micro_batch_size_or_fn"] == 8
            assert config["seq_length_or_fn"] == 512
            assert config["train_iterations_target_or_fn"] == 1000
            assert config["train_samples_target_or_fn"] == 32000
            assert config["log_every_n_train_iterations"] == 10
            assert config["is_validation_iterations_enabled_or_fn"] is True
            assert config["is_save_checkpoint_enabled_or_fn"] is False
            assert config["save_checkpoint_strategy"] == "sync"

    @pytest.mark.unit
    def test_get_base_callback_config_with_checkpoint_callback(self):
        """Test _get_base_callback_config when checkpoint callback is present."""
        trainer = MagicMock()
        trainer.max_steps = 1000
        trainer.val_check_interval = 0

        # Mock checkpoint callback
        checkpoint_callback = MagicMock()
        checkpoint_callback.__class__.__name__ = "ModelCheckpoint"
        trainer.callbacks = [checkpoint_callback]

        with patch.dict(os.environ, {"SLURM_JOB_NAME": "test_job", "WORLD_SIZE": "2"}):
            config = _get_base_callback_config(trainer=trainer, global_batch_size=16, seq_length=256)

            assert config["is_save_checkpoint_enabled_or_fn"] is True
            assert config["is_validation_iterations_enabled_or_fn"] is False

    @pytest.mark.unit
    def test_get_base_callback_config_async_save(self):
        """Test _get_base_callback_config with async save strategy."""
        trainer = MagicMock()
        trainer.max_steps = 1000
        trainer.callbacks = []
        trainer.val_check_interval = 0  # Set to 0 to avoid validation

        # Mock strategy with async_save
        strategy = MagicMock()
        strategy.async_save = True
        trainer.strategy = strategy

        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            config = _get_base_callback_config(trainer=trainer, global_batch_size=8, seq_length=128)

            assert config["save_checkpoint_strategy"] == "async"

    @pytest.mark.unit
    def test_get_base_callback_config_dict_strategy(self):
        """Test _get_base_callback_config with dict strategy."""
        trainer = MagicMock()
        trainer.max_steps = 1000
        trainer.callbacks = []
        trainer.val_check_interval = 0  # Set to 0 to avoid validation
        trainer.strategy = {"async_save": True}

        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            config = _get_base_callback_config(trainer=trainer, global_batch_size=8, seq_length=128)

            assert config["save_checkpoint_strategy"] == "async"

    @pytest.mark.unit
    def test_get_nemo_v1_callback_config(self):
        """Test get_nemo_v1_callback_config with model configuration."""
        trainer = MagicMock()
        trainer.max_steps = 500
        trainer.val_check_interval = 0  # Set to 0 to avoid validation

        # Mock lightning module with config
        pl_module = MagicMock()
        pl_module.cfg = OmegaConf.create({"train_ds": {"batch_size": 8}, "encoder": {"d_model": 768}})
        trainer.lightning_module = pl_module

        with patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            config = get_nemo_v1_callback_config(trainer)

            assert config["global_batch_size_or_fn"] == 16  # 8 * 2
            assert config["seq_length_or_fn"] == 768
            assert config["train_iterations_target_or_fn"] == 500

    @pytest.mark.unit
    def test_get_nemo_v1_callback_config_bucket_batch_size(self):
        """Test get_nemo_v1_callback_config with bucket batch sizes (ASR case)."""
        trainer = MagicMock()
        trainer.max_steps = 1000
        trainer.val_check_interval = 0  # Set to 0 to avoid validation

        # Mock lightning module with bucket batch sizes
        pl_module = MagicMock()
        pl_module.cfg = OmegaConf.create({"train_ds": {"bucket_batch_size": [4, 8, 12]}, "encoder": {"d_model": 512}})
        trainer.lightning_module = pl_module

        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            config = get_nemo_v1_callback_config(trainer)

            # Average bucket batch size is (4+8+12)/3 = 8
            assert config["global_batch_size_or_fn"] == 8
            assert config["seq_length_or_fn"] == 512

    @pytest.mark.unit
    def test_get_nemo_v1_callback_config_fallback(self):
        """Test get_nemo_v1_callback_config with fallback values."""
        trainer = MagicMock()
        trainer.max_steps = 100
        trainer.val_check_interval = 0  # Set to 0 to avoid validation

        # Mock lightning module without required config
        pl_module = MagicMock()
        pl_module.cfg = OmegaConf.create({})
        trainer.lightning_module = pl_module

        config = get_nemo_v1_callback_config(trainer)

        assert config["global_batch_size_or_fn"] == 1  # fallback
        assert config["seq_length_or_fn"] == 1  # fallback
        assert config["train_iterations_target_or_fn"] == 100

    @pytest.mark.unit
    def test_get_nemo_v2_callback_config(self):
        """Test get_nemo_v2_callback_config with data module."""
        trainer = MagicMock()
        trainer.max_steps = 200
        trainer.val_check_interval = 0  # Set to 0 to avoid validation

        # Mock data module
        data = MagicMock()
        data.global_batch_size = 64
        data.seq_length = 1024

        with patch.dict(os.environ, {"WORLD_SIZE": "4"}):
            config = get_nemo_v2_callback_config(trainer=trainer, data=data)

            assert config["global_batch_size_or_fn"] == 64
            assert config["seq_length_or_fn"] == 1024
            assert config["train_iterations_target_or_fn"] == 200

    @pytest.mark.unit
    def test_get_nemo_v2_callback_config_no_data(self):
        """Test get_nemo_v2_callback_config without data module."""
        trainer = MagicMock()
        trainer.max_steps = 300
        trainer.val_check_interval = 0  # Set to 0 to avoid validation

        config = get_nemo_v2_callback_config(trainer=trainer, data=None)

        assert config["global_batch_size_or_fn"] == 1  # fallback
        assert config["seq_length_or_fn"] == 1  # fallback
        assert config["train_iterations_target_or_fn"] == 300

    @pytest.mark.unit
    def test_should_enable_for_current_rank_single_process(self):
        """Test _should_enable_for_current_rank in single process training."""
        with patch('torch.distributed.is_initialized', return_value=False):
            result = _should_enable_for_current_rank()
            assert result is True

    @pytest.mark.unit
    def test_should_enable_for_current_rank_distributed_rank0(self):
        """Test _should_enable_for_current_rank for rank 0 in distributed training."""
        with patch('torch.distributed.is_initialized', return_value=True):
            with patch('torch.distributed.get_world_size', return_value=4):
                with patch('torch.distributed.get_rank', return_value=0):
                    result = _should_enable_for_current_rank()
                    assert result is True

    @pytest.mark.unit
    def test_should_enable_for_current_rank_distributed_last_rank(self):
        """Test _should_enable_for_current_rank for last rank in distributed training."""
        with patch('torch.distributed.is_initialized', return_value=True):
            with patch('torch.distributed.get_world_size', return_value=4):
                with patch('torch.distributed.get_rank', return_value=3):
                    result = _should_enable_for_current_rank()
                    assert result is True

    @pytest.mark.unit
    def test_should_enable_for_current_rank_distributed_middle_rank(self):
        """Test _should_enable_for_current_rank for middle rank in distributed training."""
        with patch('torch.distributed.is_initialized', return_value=True):
            with patch('torch.distributed.get_world_size', return_value=4):
                with patch('torch.distributed.get_rank', return_value=1):
                    result = _should_enable_for_current_rank()
                    assert result is False

    @pytest.mark.unit
    def test_should_enable_for_current_rank_exception_handling(self):
        """Test _should_enable_for_current_rank handles exceptions gracefully."""
        with patch('torch.distributed.is_initialized', side_effect=Exception("Test exception")):
            result = _should_enable_for_current_rank()
            assert result is False
