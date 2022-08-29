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

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.machine_translation.megatron_nmt_model import MegatronNMTModel
from nemo.collections.nlp.modules.common.megatron.parallel_adapters import ParallelLinearAdapterConfig
from nemo.collections.common.parts import adapter_modules
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import logging

__all__ = ["MegatronAdapterNMTModel"]


class MegatronAdapterNMTModel(MegatronNMTModel):
    """
    Megatron Adapter NMT training
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)

        with open_dict(cfg):
            cfg.megatron_amp_O2 = False
            cfg.micro_batch_size = self.cfg.micro_batch_size
            cfg.global_batch_size = self.cfg.global_batch_size
            cfg.precision = trainer.precision

        self.adapter_name_keys = ['adapter_1', 'adapter_2']
        frozen_model = MegatronNMTModel.restore_from(
            cfg.get('pretrained_model_path'),
            trainer=trainer,
            override_config_path=cfg,
            save_restore_connector=NLPSaveRestoreConnector(),
        )

        # set the base model and enc_dec_model module
        self.enc_dec_model = frozen_model.enc_dec_model
        logging.info(f'Before adding adapters:\n{self.summarize()}')
        self.freeze()
        for _, module in self.enc_dec_model.enc_dec_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    module.add_adapter(
                        name=adapter_key,
                        # cfg=ParallelLinearAdapterConfig(
                        cfg =adapter_modules.LinearAdapterConfig(
                            in_features=cfg.hidden_size,
                            dim=cfg.adapter_tuning.adapter_dim,
                            norm_position='post',
                            dropout=cfg.adapter_tuning.adapter_dropout,
                        ),
                    )

        logging.info(f'After adding adapters:\n{self.summarize()}')

    @classmethod
    def list_available_models(cls):
        pass

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        state_dict_ = {}
        for name, module in self.enc_dec_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.adapter_layer[adapter_key]
                    state_adapter_key = ':'.join([name, adapter_key])
                    state_dict_[state_adapter_key] = adapter_module.state_dict()
        return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        for name, module in self.enc_dec_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.adapter_layer[adapter_key]
                    state_adapter_key = ':'.join([name, adapter_key])
                    # only load the adapters if they are in the state_dict
                    if state_adapter_key in state_dict:
                        adapter_module.load_state_dict(state_dict[state_adapter_key], strict)

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
        # self freeze and unfreeze enabled adapters
        self.freeze()
        param_groups = {'params': [p for p in self.enc_dec_model.parameters()]}
        for _, module in self.enc_dec_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                module.set_enabled_adapters(enabled=True)
                module.unfreeze_enabled_adapters()
        self._optimizer_param_groups = [param_groups]
        logging.info(f'Optimizer groups set:\n{self.summarize()}')
