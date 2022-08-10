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
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.utils import logging
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.modules.common import VirtualPromptStyle
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.common.parts import adapter_modules
from nemo.core.classes.mixins import adapter_mixins

try:
    
    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

class MegatronGPTAdapterLearningModel(MegatronGPTPromptLearningModel):
    """
    MegatronGPTAdapterLearningModel is a model that combines a backbone model (GPTModel) with a adapters.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        
        self.adapter_name_keys = ['adapter_1', 'adapter_2']
        frozen_model_cfg = MegatronGPTModel.restore_from(
            cfg.get('language_model_path'), trainer=trainer, return_config=True
        ) 
        for _, layer in self.frozen_model.named_modules():
            if hasattr(layer, 'activations_checkpoint_method'):
                layer.activations_checkpoint_method = None  # (@adithyare) adapter learning does not support activations checkpointing atm.
            if hasattr(layer, 'scale_mask_softmax'):
                layer.scale_mask_softmax.scaled_masked_softmax_fusion = False
        
        logging.info(f'Before adding adapters:\n{self.frozen_model.summarize()}')
        self.frozen_model.freeze()
        for _, module in self.frozen_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    module.add_adapter(name=adapter_key, cfg=adapter_modules.LinearAdapterConfig(in_features=frozen_model_cfg.hidden_size, dim=cfg.adapter_tuning.adapter_dim))
                module.set_enabled_adapters(enabled=True)
                module.unfreeze_enabled_adapters()
        
        logging.info(f'After adding adapters:\n{self.frozen_model.summarize()}')

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
            output = self.frozen_model.model(
                input_ids=input_ids,
                position_ids=position_ids,
                encoder_input=None,
                attention_mask=attention_mask,
                labels=labels,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=inference_max_sequence_len,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                output = self.frozen_model.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    encoder_input=None,
                    attention_mask=attention_mask,
                    labels=labels,
                    set_inference_key_value_memory=set_inference_key_value_memory,
                    inference_max_sequence_len=inference_max_sequence_len,
                )

        return output


    def setup(self, stage=None):
        if (
            stage == 'predict' or self.virtual_prompt_style == VirtualPromptStyle.INFERENCE
        ):
            self.frozen_model.freeze()
            return

        self.setup_test_data()
        if stage == 'test':
            return

        self.setup_training_data()
        self.setup_validation_data()
        logging.info(f'setup completed:\n{self.frozen_model.summarize()}') 

    def on_train_end(self):
        # Save the best nemo model
        self.save_to(save_path=self.cfg.nemo_path)
    
    def on_validation_end(self):
        # Save the best nemo model
        self.save_to(save_path=self.cfg.nemo_path)


    def get_forward_output_only_func(self):
        """
        Used for generate method only for now.
        """

        def fwd_output_only_func(batch, model):
            extra_arg = {}
            (
                tokens,
                attention_mask,
                position_ids,
                task_ids,
                set_inference_key_value_memory,
                inference_max_sequence_len,
            ) = batch

            tokens = tokens.cuda()
            attention_mask = attention_mask.cuda()
            position_ids = position_ids.cuda()
            task_ids = task_ids.cuda()
            extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
            extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()

            output_tensor = model(tokens, position_ids, attention_mask, task_ids, **extra_arg)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        state_dict_ = {}
        for name, module in self.frozen_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.adapter_layer[adapter_key]
                    state_adapter_key = ':'.join([name, adapter_key])
                    state_dict_[state_adapter_key] = adapter_module.state_dict()

                module.set_enabled_adapters(enabled=True) 
        return state_dict_
    
    def load_state_dict(self, state_dict, strict: bool = True):
        for name, module in self.frozen_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.adapter_layer[adapter_key]
                    state_adapter_key = ':'.join([name, adapter_key])
                    adapter_module.load_state_dict(state_dict[state_adapter_key], strict)
                module.set_enabled_adapters(enabled=True)
        
    def setup_optimizer_param_groups(self):
        """
        ModelPT override. Optimizer will get self._optimizer_param_groups. 
        Makes two optimizer param groups, one for the frozen model params
        and one for the prompt-table/prompt-encoder params. The learning 
        rate for the frozen model's params will always be zero effectively
        freezing the model's params but still allowing for the needed gradients
        to be passed around in pipeline parallel models. The prompt-encoder 
        and/or prompt table will use the learning rate set by the user. 
        """
        # Freeze frozen model
        self.frozen_model.freeze()
        for _, module in self.frozen_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                module.set_enabled_adapters(enabled=True) 
                module.unfreeze_enabled_adapters()
        print(self.frozen_model.summarize())
        self._optimizer_param_groups = {'params': [p for p in self.frozen_model.parameters()]},

        # Need to handle frozen model freezing differently when pp > 1
        if self.pipeline_parallel:
            raise NotImplementedError('Pipeline parallel not implemented yet')

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