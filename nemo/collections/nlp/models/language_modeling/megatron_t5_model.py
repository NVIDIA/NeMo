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

import re
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from apex.transformer import parallel_state, tensor_parallel
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron.lm_encoder_decoder_model import MegatronLMEncoderDecoderModel
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.clip_grads import clip_grad_norm_fp32
from nemo.collections.nlp.modules.common.megatron.megatron_init import (
    initialize_model_parallel_for_nemo,
    set_jit_fusion_options,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    make_inference_attention_mask_3d,
    make_inference_history_mask_3d,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import AppState, logging


class MegatronT5Model(MegatronLMEncoderDecoderModel):
    """
    Megatron T5 pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        # T5-related construction
        self.num_sentinel_tokens = self.cfg.tokenizer.num_sentinel_tokens
        self._add_special_tokens_to_tokenizer()

    def _add_special_tokens_to_tokenizer(self):
        if self.cfg.tokenizer.library == 'huggingface' or self.cfg.tokenizer.library == 'megatron':
            additional_tokens = {
                'additional_special_tokens': [f'<extra_id_{i}>' for i in range(self.num_sentinel_tokens)]
            }
            self.tokenizer.add_special_tokens(additional_tokens)

        if self.cfg.tokenizer.library == 'sentencepiece':
            # Need to add cls, sep, mask tokens to the tokenizer if they don't exist.
            # If cls, sep and mask are not attributes of the tokenizer, add it.
            if not hasattr(self.tokenizer, 'cls_token'):
                self.tokenizer.add_special_tokens({'cls_token': '<cls>'})
            if not hasattr(self.tokenizer.tokenizer, 'sep_id'):
                self.tokenizer.add_special_tokens({'sep_token': '<sep>'})
            if not hasattr(self.tokenizer.tokenizer, 'mask_id'):
                self.tokenizer.add_special_tokens({'mask_token': '<mask>'})

            # bos, eos, pad and unk may be present in the provided spm .model file, if they are, use it.
            if not hasattr(self.tokenizer, 'pad_token'):
                if hasattr(self.tokenizer.tokenizer, 'pad_id'):
                    self.tokenizer.pad_token = self.tokenizer.tokenizer.id_to_piece(self.tokenizer.tokenizer.pad_id())
            else:
                self.tokenizer.add_special_tokens({'pad_token': '<pad>'})

            if not hasattr(self.tokenizer, 'bos_token'):
                if hasattr(self.tokenizer.tokenizer, 'bos_id'):
                    self.tokenizer.bos_token = self.tokenizer.tokenizer.id_to_piece(self.tokenizer.tokenizer.bos_id())
            else:
                self.tokenizer.add_special_tokens({'bos_token': '<s>'})

            if not hasattr(self.tokenizer, 'eos_token'):
                if hasattr(self.tokenizer.tokenizer, 'eos_id'):
                    self.tokenizer.eos_token = self.tokenizer.tokenizer.id_to_piece(self.tokenizer.tokenizer.eos_id())
            else:
                self.tokenizer.add_special_tokens({'eos_token': '</s>'})

            additional_tokens = [f'<extra_id_{i}>' for i in range(self.num_sentinel_tokens)]
            self.tokenizer.add_special_tokens(additional_tokens)

    def list_available_models():
        pass
