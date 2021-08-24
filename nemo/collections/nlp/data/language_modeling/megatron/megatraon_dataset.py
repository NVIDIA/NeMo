# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""Megatron dataset class that handles initialization and other megatron specific things common to all datasets."""

import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_megatron_for_nemo
from nemo.utils import AppState, logging


class MegatronDataset(torch.utils.data.Dataset):
    """
    Megatron GPT pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        app_state = AppState()

        if not hasattr(app_state, "_megatron_init"):
            logging.info(
                f"Initializing megatron since it hasn't been initialized by the model. This is normal if you are using a NeMo model with Megatron dataloaders."
            )
            app_state.global_rank = trainer.global_rank
            app_state.world_size = trainer.world_size
            app_state.model_parallel_size = 1
            app_state.model_parallel_rank = trainer.global_rank

            initialize_megatron_for_nemo(
                world_size=app_state.world_size,
                global_rank=app_state.global_rank,
                micro_batch_size=cfg.get('micro_batch_size', 1),
                tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
                tensor_model_parallel_rank=app_state.model_parallel_rank,
                encoder_seq_length=cfg.get('encoder_seq_length', 512),
                num_layers=cfg.get('num_layers', 1),
                hidden_size=cfg.get('hidden_size', 16),
                num_attention_heads=cfg.get('num_attention_heads', 1),
                max_position_embeddings=cfg.get('max_position_embeddings', 512),
                tokenizer_type='GPT2BPETokenizer',
                vocab_file=cfg.vocab_file,
                merge_file=cfg.merge_file,
            )
