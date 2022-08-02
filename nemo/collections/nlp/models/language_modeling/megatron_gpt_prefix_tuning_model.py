# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# This code has been adapted from the following private repo: https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/tree/prompt-learning/prefix_tuning_v2
# Adapted by: @adithyare


import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from torch import nn

from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.modules.common import VirtualPromptStyle
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.transformer import ColumnLinear
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group

try:
    from apex.transformer.parallel_state import get_tensor_model_parallel_world_size

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


class Prefix(MegatronModule):
    """
    Class that contains prefix parameters for prefix tuning.
    """

    def __init__(
        self, hidden_size, num_layers, num_heads, prefix_len, prefix_projection_size, prefix_dropout,
    ):

        super(Prefix, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_emb_per_head = hidden_size // num_heads
        self.prefix_len = prefix_len
        self.prefix_projection_size = prefix_projection_size
        self.prefix_dropout = prefix_dropout
        world_size = get_tensor_model_parallel_world_size()
        self.num_attention_heads_per_partition = (
            num_heads // world_size
        )  # TODO (@adithyare): megatron-LM code used mpu.divide for this, not sure why...
        self.prefix_idx = torch.arange(0, self.prefix_len).long().unsqueeze(0)
        self.prefix_embeddings = nn.Embedding(prefix_len, hidden_size)
        self.prefix_projection = nn.Linear(hidden_size, prefix_projection_size)
        self.prefix_nonlinearity = nn.Tanh()
        self.prefix_layer_projection = ColumnLinear(prefix_projection_size, num_layers * self.num_emb_per_head * self.num_attention_heads_per_partition * 2)
        nn.init.normal_(self.prefix_embeddings.weight)
        nn.init.xavier_uniform_(self.prefix_projection.weight)
        nn.init.constant_(self.prefix_projection.bias, 0)
        self.dropout = nn.Dropout(prefix_dropout)


    def forward(self, bsz):
        """
        Args:
            bsz: an integer representing the batch size
        """
        prefix_reps = self.prefix_embeddings(self.prefix_idx.expand(bsz, -1))  # (bsz, seqlen, hidden_size)
        prefix_reps = self.prefix_projection(prefix_reps)  # (bsz, seqlen, prefix_projection_size)
        prefix_reps = self.prefix_nonlinearity(prefix_reps)
        prefix_reps, _ = self.prefix_layer_projection(prefix_reps)  # (bsz, seqlen, num_layers * hidden_size * 2)
        prefix_reps = self.dropout(prefix_reps)
        effective_bsz, effective_prefix_len, _ = prefix_reps.shape
        prefix_reps = prefix_reps.view(
            effective_bsz, effective_prefix_len, 2, self.num_layers, self.num_attention_heads_per_partition, self.num_emb_per_head,
        )
        prefix_reps = prefix_reps.permute([2, 1, 3, 0, 4, 5])
        return prefix_reps

    def to(self, device):
        self.prefix_idx = self.prefix_idx.to(device)
        super().to(device)
        return True


class PrefixTuningModel(MegatronGPTPromptLearningModel):
    """
    PrefixTuningModel is a model that combines a backbone model with a prefix-module.
    The backbone model is expected to accept the prefix-representations as input in the forward pass.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.prefix_generator = Prefix(
            self.frozen_model.cfg.hidden_size,
            self.frozen_model.cfg.num_layers,
            self.frozen_model.cfg.num_attention_heads,
            cfg.prefix_tuning.prefix_len,
            cfg.prefix_tuning.prefix_projection_size,
            cfg.prefix_tuning.prefix_dropout,
        )
        self.prefix_generator_key = "prefix_tuning_generator"
        self.prefix_generator_learning_rate = cfg.prefix_tuning.prefix_learning_rate
        self.prefix_generator.to(self.frozen_model.device)

        for name, layer in self.frozen_model.named_modules():
            if hasattr(layer, 'activations_checkpoint_method'):
                layer.activations_checkpoint_method = None  # (@adithyare) prefix tuning does not support activations checkpointing atm.
            if hasattr(layer, 'scale_mask_softmax'):
                layer.scale_mask_softmax.scaled_masked_softmax_fusion = False

    @classmethod
    def list_available_models(cls):
        pass

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        taskname_ids,
        labels=None,
        inference=True,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
    ):
        # Call forward on GPT model with preprocessed embeddings
        if self.autocast_dtype == torch.float32:
            bsz = len(input_ids)
            prefix_tuning_key_values = self.prefix_generator(bsz)
            output = self.frozen_model.model(
                input_ids=input_ids,
                position_ids=position_ids,
                encoder_input=None,
                attention_mask=attention_mask,
                labels=labels,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=inference_max_sequence_len,
                prefix_tuning_key_values=prefix_tuning_key_values,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                bsz = len(input_ids)
                prefix_tuning_key_values = self.prefix_generator(bsz)
                output = self.frozen_model.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    encoder_input=None,
                    attention_mask=attention_mask,
                    labels=labels,
                    set_inference_key_value_memory=set_inference_key_value_memory,
                    inference_max_sequence_len=inference_max_sequence_len,
                    prefix_tuning_key_values=prefix_tuning_key_values,
                )

        return output

    def setup(self, stage=None):
        if (
            stage == 'predict' or self.virtual_prompt_style == VirtualPromptStyle.INFERENCE
        ) and self.frozen_model.model.pre_process:
            self.freeze_existing_virtual_prompt_params()
            return

        self.setup_test_data()
        if stage == 'test':
            return

        self.setup_training_data()
        self.setup_validation_data()

    def state_dict(self):
        return {self.prefix_generator_key: self.prefix_generator.state_dict()}

    def load_state_dict(self, state_dict, strict=True):
        assert (
            self.prefix_generator_key in state_dict
        ), f"Prefix generator key {self.prefix_generator_key} not found in state dict: {state_dict}"
        self.prefix_generator.load_state_dict(state_dict[self.prefix_generator_key], strict=strict)

    def on_train_end(self):
        # Save the best nemo model
        self.save_to(save_path=self.cfg.nemo_path)

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(batch, model):
            batch = [x.cuda(non_blocking=True) for x in batch]
            input_ids, labels, loss_mask, position_ids, attention_mask, taskname_ids = batch
            output_tensor = model(input_ids, position_ids, attention_mask, taskname_ids, labels, inference=False)

            def loss_func(output_tensor):
                loss = self.frozen_model.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func