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

from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.collections.nlp.models.language_modeling.megatron_lm_encoder_decoder_model import (
    MegatronLMEncoderDecoderModel,
)
from nemo.collections.nlp.modules.common.megatron.parallel_adapters import ParallelLinearAdapterConfig
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import logging

__all__ = ["MegatronLMAdapterEncoderDecoderModel"]


class MegatronLMAdapterEncoderDecoderModel(MegatronLMEncoderDecoderModel):
    """
    Megatron Adapter NMT training
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)

        if hasattr(cfg, 'adapter_tuning'):
            logging.info('Using Adapters')

            # validate and add adapters
            self._validate_adapters_cfg(cfg.adapter_tuning)

            with open_dict(cfg):
                cfg.tokenizer = self.cfg.encoder_tokenizer

            self.adapter_name_keys = ['adapter_1', 'adapter_2']

            # set the base model and enc_dec_model module
            logging.info(f'Before adding adapters:\n{self.summarize()}')
            self.freeze()
            if cfg.adapter_tuning.type == "parallel_adapter":
                adapter_cfg = ParallelLinearAdapterConfig(
                    in_features=cfg.hidden_size,
                    dim=cfg.adapter_tuning.adapter_dim,
                    norm_position=cfg.adapter_tuning.get('norm_position', 'pre'),
                    norm_type=cfg.adapter_tuning.get('norm_type', 'mixedfusedlayernorm'),
                    column_init_method=cfg.adapter_tuning.get('column_init_method', 'xavier'),
                    row_init_method=cfg.adapter_tuning.get('row_init_method', 'zero'),
                    dropout=cfg.adapter_tuning.adapter_dropout,
                )
            else:
                adapter_cfg = LinearAdapterConfig(
                    in_features=cfg.hidden_size,
                    dim=cfg.adapter_tuning.adapter_dim,
                    norm_position=cfg.adapter_tuning.get('norm_position', 'pre'),
                    dropout=cfg.adapter_tuning.adapter_dropout,
                )

            for _, module in self.enc_dec_model.named_modules():
                if isinstance(module, adapter_mixins.AdapterModuleMixin):
                    for adapter_key in self.adapter_name_keys:
                        module.add_adapter(name=adapter_key, cfg=adapter_cfg)

            logging.info(f'After adding adapters:\n{self.summarize()}')

    @classmethod
    def list_available_models(cls):
        pass

    def _validate_adapters_cfg(self, cfg):
        assert cfg.type in ['parallel_adapter', 'linear_adapter']
        assert hasattr(cfg, 'adapter_dim')
        assert cfg.norm_position in ['pre', 'post']

        if hasattr(cfg, 'norm_type'):
            assert cfg.norm_type in ['mixedfusedlayernorm', 'layernorm']

        for val in ['row_init_method', 'column_init_method']:
            if hasattr(cfg, val):
                assert cfg.get(val) in ['xavier', 'zero', 'normal']

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        state_dict_ = {}

        if hasattr(self.cfg, 'adapter_tuning'):
            for name, module in self.enc_dec_model.named_modules():
                if isinstance(module, adapter_mixins.AdapterModuleMixin):
                    for adapter_key in self.adapter_name_keys:
                        adapter_module = module.adapter_layer[adapter_key]
                        state_adapter_key = ':'.join([name, adapter_key])
                        state_dict_[state_adapter_key] = adapter_module.state_dict()
        else:
            for name, module in self.enc_dec_model.named_modules():
                state_dict_[name] = module.state_dict()
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
        if hasattr(self.cfg, 'adapter_tuning'):
            self.freeze()
            param_groups = {'params': [p for p in self.enc_dec_model.parameters()]}
            for _, module in self.enc_dec_model.named_modules():
                if isinstance(module, adapter_mixins.AdapterModuleMixin):
                    module.set_enabled_adapters(enabled=True)
                    module.unfreeze_enabled_adapters()
        else:
            param_groups = {'params': [p for p in self.enc_dec_model.parameters()]}

        self._optimizer_param_groups = [param_groups]
        logging.info(f'Optimizer groups set:\n{self.summarize()}')
