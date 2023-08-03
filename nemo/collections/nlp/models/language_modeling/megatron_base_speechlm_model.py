# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
import re
from collections import OrderedDict
from typing import Any, Optional

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.trainer.trainer import Trainer
from torch import Tensor

from nemo.collections.nlp.metrics.prompt_learning_metrics import AccuracyScore, BLEUScore, ROUGEScores
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.transformer.text_generation import TextGeneration
from nemo.collections.nlp.parts.nlp_overrides import GradScaler
from nemo.utils import AppState, logging

try:
    from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


__all__ = ['MegatronSpeechLMBaseModel']


class MegatronSpeechLMBaseModel(MegatronBaseModel, TextGeneration):
    """
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.init_model(cfg, trainer)

    def init_model(self, cfg: DictConfig, trainer: Trainer):
        self.cfg = cfg

        self.load_frozen_model(cfg, trainer)
        self.prompt_encoder = None
        self.tokenizer = self.frozen_model.tokenizer

        if hasattr(self.frozen_model.cfg, "encoder") and hasattr(self.frozen_model.cfg, "decoder"):
            self.hidden_size = (
                self.frozen_model.cfg.encoder.hidden_size
            )  # Encoder and decoder need to have the same hidden size and we check for this in the frozen enc-dec model.
        else:
            self.hidden_size = self.frozen_model.cfg.hidden_size

        if self.first_stage_of_pipeline():
            # TODO: Handle this when moving GPT prompt learning to the base class.
            self.word_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.word_embeddings

        
        self._reduced_loss_buffer = []
        self._inference_config = None

        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.unk_id
        self.decoder_seq_length = cfg.get('decoder_seq_length', 40)

        if self.trainer.precision == 'bf16':
            self.autocast_dtype = torch.bfloat16
        elif int(self.trainer.precision) == 32:
            self.autocast_dtype = torch.float
        elif int(self.trainer.precision) == 16:
            self.autocast_dtype = torch.half
        else:
            raise ValueError('precision must be in [32, 16, "bf16"]')
        # make sure the default pytorch lightning gradient clipping in the basemodel
        self.grad_clip_pl_default = True
        self.lowest_val_loss = None
        self.prompt_encoder = None

        self.enable_autocast = (
            True if (not self.megatron_amp_o2) and (self.autocast_dtype in [torch.float16, torch.bfloat16]) else False
        )

        # define validation metric
        if self.cfg.get('report_validation_metric', False):
            validation_metric = self.cfg.get('validation_metric', 'accuracy')
            if validation_metric == 'accuracy':
                self.validation_metric = AccuracyScore()
            elif validation_metric == 'bleu':
                self.validation_metric = BLEUScore()
            elif validation_metric == 'rouge':
                self.validation_metric = ROUGEScores()

    def state_dict(self):
        """
        Custom state dict that only contains prompt table and prompt encoder parameters.
        No frozen model parameters are stored in the state dict. Prompt encoder parameters
        are only in state dict for intermediate checkpoints saved during training. Final
        nemo checkpoints at the end of training will contain prompt table parameters only.
        """
        state_dict_ = {}
        state_dict_["frozen_model_enc_dec_model"] = self.frozen_model.enc_dec_model.state_dict()
        state_dict_["word_embeddings"] = self.word_embeddings.state_dict()
        
        return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Custom load state dict method that only loads prompt table and prompt encoder
        parameters. Matching load method for this class' custom state dict method.
        """
        self.frozen_model.enc_dec_model.load_state_dict(state_dict["frozen_model_enc_dec_model"], strict)
        self.word_embeddings.load_state_dict(state_dict["word_embeddings"], strict)
    
    def on_train_end(self):
        # Save p-tuned prompts to prompt table for inference or future task training
        self.save_to(save_path=self.cfg.nemo_path)

    
    def _reconfigure_and_process_inference_batch(self, global_batch_size_per_gpu, gbs):
        # This should happen only on the last batch of the dataset.
        if global_batch_size_per_gpu != gbs // parallel_state.get_data_parallel_world_size():
            # NOTE: This is reconfiguring to make sure there is no grad-acc for validation batches.
            app_state = AppState()
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=global_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
                micro_batch_size=global_batch_size_per_gpu,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

    def _reconfigure_batch_sizes(self, gbs: int, mbs: int):
        app_state = AppState()
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=gbs,
            micro_batch_size=mbs,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

    def set_inference_config(self, inference_config):
        self._inference_config = inference_config

    def get_inference_config(self):
        return self._inference_config

    def set_input_tensor(self, input_tensor):
        pass

    def first_stage_of_pipeline(self):
        pass

    @classmethod
    def list_available_models(cls):
        pass

    def load_frozen_model(self, cfg, trainer):
        pass