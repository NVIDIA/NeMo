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

import math

import torch
import torch.distributed
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from nemo.collections import llm
from nemo.tron.api import megatron_pretrain
from nemo.tron.config import (
    CheckpointConfig,
    ConfigContainer,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from nemo.tron.data.dataset import get_blend_and_blend_per_split
from nemo.tron.llm.gpt import forward_step

if __name__ == "__main__":
    global_batch_size = 256
    micro_batch_size = 1
    seq_length = 4096

    # Model
    model_cfg = llm.Llama32Config1B(
        num_layers=25,
        hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=16,
        ffn_hidden_size=6144,
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

    # Dataset
    blend, blend_per_split = get_blend_and_blend_per_split(
        data_paths=[f"/path/to/data/dclm_{i:02d}_text_document" for i in range(1, 51)]
    )

    tokens = 60_000_000_000
    max_steps = math.ceil(tokens / model_cfg.seq_length / global_batch_size)

    # Config Container
    cfg = ConfigContainer(
        model_config=model_cfg,
        train_config=TrainingConfig(
            train_iters=max_steps,
            eval_interval=2000,
            eval_iters=32,
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
            lr_warmup_iters=5000,
            lr_warmup_init=0.0,
            lr_decay_iters=max_steps,
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
            blend=blend,
            blend_per_split=blend_per_split,
            random_seed=2788,
            sequence_length=model_cfg.seq_length,
            path_to_cache="/hubs/data/index_mapping_dclm_1_0",
            reset_position_ids=False,
            create_attention_mask=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            num_dataset_builder_threads=1,
            split="9999,8,2",
            data_sharding=True,
            dataloader_type="single",
            num_workers=2,
        ),
        logger_config=LoggerConfig(
            wandb_project="nemo_custom_pretraining_loop",
            wandb_entity="nvidia",
            wandb_exp_name=f"lingua_dclm_full_1b_gbs_{global_batch_size}_20250303",
            wandb_save_dir="/nemo_run/wandb",
            tensorboard_dir="/nemo_run/tensorboard",
            log_timers_to_tensorboard=True,
            log_validation_ppl_to_tensorboard=True,
            tensorboard_log_interval=10,
            timing_log_level=2,
            log_progress=True,
            log_interval=10,
            logging_level="INFO",
        ),
        tokenizer_config=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="/path/to/hf-tokenizer/",
        ),
        checkpoint_config=CheckpointConfig(
            save_interval=10000,
            save="/nemo_run/checkpoints",
            load="/nemo_run/checkpoints",
            async_save=True,
            fully_parallel_save=True,
        ),
        rng_config=RNGConfig(seed=2888),
    )

    megatron_pretrain(cfg, forward_step)
