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

from functools import partial

import torch
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.multimodal_dataset import MockMultimodalDataset, MultimodalDatasetConfig
from megatron.core.optimizer import OptimizerConfig

from nemo.collections import llm, vlm
from nemo.collections.vlm.neva.model.base import neva_data_step, neva_forward_step
from nemo.tron.api import megatron_pretrain
from nemo.tron.config import CheckpointConfig, ConfigContainer, LoggerConfig, SchedulerConfig, TrainingConfig
from nemo.tron.losses import masked_next_token_loss
from nemo.tron.state import GlobalState
from nemo.tron.utils.common_utils import print_rank_0


def forward_step(state: GlobalState, data_iterator, model):
    timers = state.timers

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    batch = neva_data_step(data_iterator)
    loss_mask = batch.get("loss_mask", None)
    timers("batch-generator").stop()

    output_tensor = neva_forward_step(model, batch)

    return output_tensor, partial(masked_next_token_loss, loss_mask)


def neva_dataset_provider(train_val_test_num_samples: list[int], dataset_config: MultimodalDatasetConfig):
    print_rank_0("> building train, validation, and test datasets for multimodal ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        MockMultimodalDataset,
        train_val_test_num_samples,
        lambda: parallel_state.get_tensor_model_parallel_rank() == 0,
        dataset_config,
    ).build()

    print_rank_0("> finished creating multimodal datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    gbs = 8
    mbs = 2
    seq_length = 576
    decoder_seq_length = 1024

    # Transformer configurations
    language_transformer_config = llm.Llama2Config7B(seq_length=decoder_seq_length, num_layers=2)

    vision_transformer_config = vlm.HFCLIPVisionConfig(
        pretrained_model_name_or_path="openai/clip-vit-large-patch14-336"
    )
    vision_projection_config = vlm.MultimodalProjectorConfig(
        projector_type="mcore_mlp",
        input_size=vision_transformer_config.hidden_size,
        hidden_size=language_transformer_config.hidden_size,
        ffn_hidden_size=language_transformer_config.hidden_size,
    )

    # NEVA model configuration
    neva_config = vlm.NevaConfig(
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        freeze_language_model=True,
        freeze_vision_model=True,
    )

    dataset_config = MultimodalDatasetConfig(
        random_seed=1234,
        sequence_length=decoder_seq_length,
        split="9999,8,2",
    )

    cfg = ConfigContainer(
        model_config=neva_config,
        train_config=TrainingConfig(
            train_iters=1000,
            eval_interval=50,
            eval_iters=8,
            global_batch_size=gbs,
            micro_batch_size=gbs,
            exit_signal_handler=True,
        ),
        optimizer_config=OptimizerConfig(
            optimizer="adam",
            lr=6e-4,
            min_lr=6e-5,
            use_distributed_optimizer=False,
            bf16=True,
        ),
        scheduler_config=SchedulerConfig(
            start_weight_decay=0.033,
            end_weight_decay=0.033,
            weight_decay_incr_style="constant",
            lr_decay_style="cosine",
            lr_warmup_iters=10,
            lr_warmup_init=0.0,
            lr_decay_iters=1000,
            override_opt_param_scheduler=True,
        ),
        dataset_config=dataset_config,
        logger_config=LoggerConfig(
            tensorboard_dir="/nemo_run/tensorboard",
            log_timers_to_tensorboard=True,
            log_validation_ppl_to_tensorboard=True,
            tensorboard_log_interval=10,
            timing_log_level=2,
            log_progress=True,
            log_interval=10,
            logging_level="INFO",
        ),
        tokenizer_config=None,  # TODO: ???
        checkpoint_config=CheckpointConfig(
            save_interval=100,
            save="/nemo_run/checkpoints",
            load="/nemo_run/checkpoints",
            async_save=True,
            fully_parallel_save=True,
        ),
    )

    megatron_pretrain(cfg, forward_step, neva_dataset_provider)
