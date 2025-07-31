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

import os
import shutil

import pytest
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from nemo.collections import llm
from nemo.tron.api import megatron_pretrain
from nemo.tron.config import (CheckpointConfig, ConfigContainer,
                              GPTDatasetConfig, LoggerConfig, RNGConfig,
                              SchedulerConfig, TokenizerConfig, TrainingConfig)
from nemo.tron.llm.gpt import forward_step


class TestMockTrain:
    @pytest.mark.run_only_on('GPU')  # Standard pattern in NeMo tests for GPU-only tests
    def test_mock_training_checkpoint(self, tmp_path):
        # Use tmp_path fixture from pytest for temporary directory
        checkpoint_dir = str(tmp_path / "checkpoints")
        tensorboard_dir = str(tmp_path / "tensorboard")

        try:
            # Training parameters
            global_batch_size = 8
            micro_batch_size = 1
            seq_length = 512
            total_iters = 20

            # Model - very small for testing
            model_cfg = llm.Llama32Config1B(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=False,
                attention_softmax_in_fp32=True,
                pipeline_dtype=torch.bfloat16,
                bf16=True,
                seq_length=seq_length,
                make_vocab_size_divisible_by=128,
            )

            # Config Container
            cfg = ConfigContainer(
                model_config=model_cfg,
                train_config=TrainingConfig(
                    train_iters=total_iters,
                    eval_interval=5,
                    eval_iters=2,
                    global_batch_size=global_batch_size,
                    micro_batch_size=micro_batch_size,
                    exit_signal_handler=True,
                ),
                optimizer_config=OptimizerConfig(
                    optimizer="adam",
                    bf16=True,
                    fp16=False,
                    adam_beta1=0.9,
                    adam_beta2=0.95,
                    adam_eps=1e-5,
                    use_distributed_optimizer=True,
                    clip_grad=1.0,
                    lr=3e-3,
                    weight_decay=0.01,
                    min_lr=1e-6,
                ),
                scheduler_config=SchedulerConfig(
                    start_weight_decay=0.033,
                    end_weight_decay=0.033,
                    weight_decay_incr_style="constant",
                    lr_decay_style="cosine",
                    lr_warmup_iters=2,
                    lr_warmup_init=0.0,
                    lr_decay_iters=total_iters,
                    override_opt_param_scheduler=True,
                ),
                ddp_config=DistributedDataParallelConfig(
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                    average_in_collective=True,
                    use_distributed_optimizer=True,
                ),
                dataset_config=GPTDatasetConfig(
                    random_seed=1234,
                    reset_attention_mask=False,
                    reset_position_ids=False,
                    eod_mask_loss=False,
                    sequence_length=seq_length,
                    num_dataset_builder_threads=1,
                    data_sharding=True,
                    dataloader_type="single",
                    num_workers=1,
                ),
                logger_config=LoggerConfig(
                    log_interval=1,
                    tensorboard_dir=tensorboard_dir,
                ),
                tokenizer_config=TokenizerConfig(
                    tokenizer_type="NullTokenizer",
                    vocab_size=10000,
                ),
                checkpoint_config=CheckpointConfig(
                    save_interval=5,
                    save=checkpoint_dir,
                    ckpt_format="torch_dist",
                    fully_parallel_save=True,
                ),
                rng_config=RNGConfig(seed=1234),
            )

            # Run training
            megatron_pretrain(cfg, forward_step)

            # Check for the latest checkpoint tracker file
            latest_tracker_file = os.path.join(checkpoint_dir, "latest_train_state.pt")
            assert os.path.exists(latest_tracker_file), "Latest checkpoint tracker file not found"

            # Check for the final checkpoint directory (should be iter_0000020)
            final_iter_dir = os.path.join(checkpoint_dir, f"iter_{total_iters:07d}")
            assert os.path.exists(final_iter_dir), f"Final checkpoint directory not found at {final_iter_dir}"

            # For distributed checkpoints, check for the metadata file
            metadata_file = os.path.join(final_iter_dir, ".metadata")
            assert os.path.exists(metadata_file), "Checkpoint metadata file not found"

        finally:
            # pytest's tmp_path fixture doesn't clean up immediately.
            # Clean up manually.
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            if os.path.exists(tensorboard_dir):
                shutil.rmtree(tensorboard_dir)
