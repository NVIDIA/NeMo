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


import os
from typing import List
import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.modules.common import VirtualPromptStyle
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    InfusedAdapterConfig,
    MLPInfusedAdapterConfig,
    ParallelLinearAdapterConfig,
    PromptEncoderAdapterConfig,
)
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import AppState, logging, model_utils
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.core.classes import ModelPT


class AdapterWrapper(ModelPT):
    def __init__(self, cfg, dummy_params: torch.nn.Linear):
        super().__init__(cfg, trainer=None)
        self.cfg = cfg
        self.dummy_params = dummy_params

    @classmethod
    def list_available_models(cls):
        pass

    @classmethod
    def setup_training_data(cls):
        pass

    @classmethod
    def setup_validation_data(cls):
        pass

    def state_dict(self, base_model, destination=None, prefix=None, keep_vars=False):
        """
        Creates a state_dict using only the adapter parameters.
        This ensures that this wrapper class will only checkpoint the adapter
        weights and not the rest of the base GPT Model.
        """
        state_dict_ = {}
        for name, module in base_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.get_adapter_module(adapter_key)
                    if adapter_module:
                        state_adapter_key = ':'.join([name, adapter_key])
                        state_dict_[state_adapter_key] = adapter_module.state_dict()

                module.set_enabled_adapters(enabled=True)
        return state_dict_

    def load_state_dict(self, base_model, state_dict, strict: bool = True):
        """
        Loads a state_dict expecting the state_dict to contain key,values 
        only for the adapter parameters.
        """
        for name, module in base_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.get_adapter_module(adapter_key)
                    if adapter_module:
                        state_adapter_key = ':'.join([name, adapter_key])
                        adapter_module.load_state_dict(state_dict[state_adapter_key], strict)
                module.set_enabled_adapters(enabled=True)

class MegatronGPTBaseAdapterModel(MegatronGPTSFTModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.setup_complete = False
        self.adapter_name_keys = [AdapterName.PRE_ATTN_ADAPTER, AdapterName.POST_ATTN_ADAPTER]
        adapter_tuning_cfg = cfg.adapter_tuning

        if adapter_tuning_cfg.type == "parallel_adapter":
            adapter_cfg = ParallelLinearAdapterConfig(
                in_features=self.cfg.hidden_size,
                dim=adapter_tuning_cfg.adapter_dim,
                norm_position=adapter_tuning_cfg.get('norm_position', 'pre'),
                norm_type=adapter_tuning_cfg.get('norm_type', 'mixedfusedlayernorm'),
                column_init_method=adapter_tuning_cfg.get('column_init_method', 'xavier'),
                row_init_method=adapter_tuning_cfg.get('row_init_method', 'zero'),
                dropout=adapter_tuning_cfg.adapter_dropout,
            )
        else:
            adapter_cfg = LinearAdapterConfig(
                in_features=self.cfg.hidden_size,
                dim=adapter_tuning_cfg.adapter_dim,
                norm_position=adapter_tuning_cfg.get('norm_position', 'pre'),
                dropout=adapter_tuning_cfg.adapter_dropout,
            )
        self.adapter_cfg = adapter_cfg
        self.base_keys = self.get_all_keys()
        self.init_adapters()
        self.adapter_keys = self.get_all_keys() - self.base_keys
    
    def setup(self, stage=None):
        super().setup(stage)
        self.setup_complete = True
    
    def get_all_keys(self,):
        k = [n for n, p in self.named_parameters()]
        return set(k)
    
    def get_adapter_state_dict(self,):
        """ 
        Gets the keys associated with the adapters only.
        """
        state_dict = self.model.state_dict(prefix="model.")
        adapter_state_dict = {}
        for k in self.adapter_keys:
            adapter_state_dict[k] = state_dict[k]
        return adapter_state_dict 

    def init_adapters(self):
        """ 
        Randomly initialize the adapter params and add them to the appropriate modules.
        """
        logging.info(f'Before adding adapters:\n{self.summarize()}')
        for _, module in self.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    if model_utils.import_class_by_path(self.adapter_cfg._target_) in module.get_accepted_adapter_types():
                        module.add_adapter(
                            name=adapter_key, cfg=self.adapter_cfg,
                        )
        logging.info(f'After adding adapters:\n{self.summarize()}')
        return True
    
    def state_dict(self, destination=None, prefix=None, keep_vars=False): 
        if self.setup_complete:
            # Once setup is complete we no longer need to track the frozen part of the model. Only there adapter state dict keeps changing so state_dict only track these.
            return self.get_adapter_state_dict()
        else:
            # we want all the params with the same keys as calling self.state_dict()
            # but we can't call self.state_dict() here as it would be a recursive call.
            # so we call self.model.state_dict(prefix="model.") which will return all the keys and params same as calling self.state_dict()
            return self.model.state_dict(prefix="model.")
    
    def load_state_dict(self, state_dict, strict: bool = True):
        if self.setup_complete:
            # at this stage only adapter params will appear in the state_dict arg, so we only update those while the rest of the model is frozen.
            # setting strict=False will ignore the missing keys (which are not being updated anyway)
            assert set(state_dict.keys()) == self.adapter_keys
            super().load_state_dict(state_dict, strict=False)
        else:
            super().load_state_dict(state_dict, strict=True)
                
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
        self.freeze()  # Freeze the entire model
        opt_params = []
        for _, module in self.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                module.set_enabled_adapters(enabled=True)
                module.unfreeze_enabled_adapters()  # selectively unfreeze the adapter modules.
                opt_params += [p for p in module.parameters()]

        self._optimizer_param_groups = ({'params': opt_params},)
        logging.info(f'Optimizer groups set:\n{self.summarize()}')
        
        
class MegatronNLPAdapter(ModelPT):
    """ 
    Wrapper class that handles loading and saving of adapters via load_state_dict and save_dict. Also handles initialization of restore_from for adapters.
    """
    def __init__(self, cfg: DictConfig, trainer: Trainer, adapter_name_keys: List[str], base_model: MegatronGPTModel):
        super().__init__(cfg, trainer)
        self.base_model = base_model
        self.adapter_name_keys = adapter_name_keys
    
    @classmethod
    def list_available_models(cls):
        pass

    @classmethod
    def setup_training_data(cls):
        pass

    @classmethod
    def setup_validation_data(cls):
        pass

    def init_adapters(self,):
        """
        TODO: 
        """
        raise NotImplementedError("not implemented")

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        Creates a state_dict using only the adapter parameters.
        This ensures that this wrapper class will only checkpoint the adapter
        weights and not the rest of the base GPT Model.
        """
        state_dict_ = {}
        for name, module in self.base_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.get_adapter_module(adapter_key)
                    if adapter_module:
                        state_adapter_key = ':'.join([name, adapter_key])
                        state_dict_[state_adapter_key] = adapter_module.state_dict()

                module.set_enabled_adapters(enabled=True)
        return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Loads a state_dict expecting the state_dict to contain key,values 
        only for the adapter parameters.
        """
        for name, module in self.base_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.get_adapter_module(adapter_key)
                    if adapter_module:
                        state_adapter_key = ':'.join([name, adapter_key])
                        adapter_module.load_state_dict(state_dict[state_adapter_key], strict)
                module.set_enabled_adapters(enabled=True)

class ___MegatronGPTAdapterLearningModel(MegatronNLPAdapter):
    """
    MegatronGPTAdapterLearningModel is a model that combines a base model (GPTModel) with a adapters.
    This class only supports the canonical Adapter training described in Houlsby et al. (https://arxiv.org/pdf/1902.00751.pdf)

    Two adapter's are inserted into each Transformer layer in the base GPT Model.

    It is assumed that these set of adapters will then be trained for a specific task.
    Once trained, the adapter weights will be saved and can be re-loaded 
    and infused into the same GPT Model for inference. 
    """

    def __init__(self, adapter_tuning_cfg: DictConfig, trainer: Trainer, base_model: MegatronGPTModel):
        adapter_name_keys = [AdapterName.PRE_ATTN_ADAPTER, AdapterName.POST_ATTN_ADAPTER]
        super().__init__(adapter_tuning_cfg, trainer, adapter_name_keys, base_model)
        assert adapter_tuning_cfg.get('adapter_dim', 0) > 0, "adapter_dim has not been set."
        assert (
            adapter_tuning_cfg.adapter_dim % self.base_model.cfg.tensor_model_parallel_size == 0
        ), "The adapter dim should be divisible by tensor_model_parallel_size."
        assert adapter_tuning_cfg.type in [
            'linear_adapter',
            'parallel_adapter',
        ], "Adapter type should be 'linear_adapter' or 'parallel_adapter'"


        if adapter_tuning_cfg.type == "parallel_adapter":
            adapter_cfg = ParallelLinearAdapterConfig(
                in_features=self.base_model.cfg.hidden_size,
                dim=adapter_tuning_cfg.adapter_dim,
                norm_position=adapter_tuning_cfg.get('norm_position', 'pre'),
                norm_type=adapter_tuning_cfg.get('norm_type', 'mixedfusedlayernorm'),
                column_init_method=adapter_tuning_cfg.get('column_init_method', 'xavier'),
                row_init_method=adapter_tuning_cfg.get('row_init_method', 'zero'),
                dropout=adapter_tuning_cfg.adapter_dropout,
            )
        else:
            adapter_cfg = LinearAdapterConfig(
                in_features=self.base_model.cfg.hidden_size,
                dim=adapter_tuning_cfg.adapter_dim,
                norm_position=adapter_tuning_cfg.get('norm_position', 'pre'),
                dropout=adapter_tuning_cfg.adapter_dropout,
            )
        self.adapter_cfg = adapter_cfg
        self.init_adapters()

    


class MegatronGPTInfusedAdapterModel(MegatronGPTBaseAdapterModel):
    """
    MegatronGPTInfusedAdapterModel is a model that combines a base model (GPTModel) with a "Infused Adapter that can Inhibiting and Amplify Inner Activations", known as IA3.
    This class supports the addition of IA3 into a transformer based LM as described in Liu et al. (https://arxiv.org/pdf/2205.05638.pdf)

    Three adapter's are inserted into each Transformer layer in the base GPT Model. Each adapter is basically a vector that simply scales the key, value or ffn hidden representations.

    It is assumed that these set of adapters will then be trained for a specific task.
    Once trained, the adapter weights will be saved and can be re-loaded 
    and infused into the same GPT Model for inference. 
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.adapter_name_keys = [AdapterName.KEY_INFUSED, AdapterName.VALUE_INFUSED, AdapterName.MLP_INFUSED]

        logging.info(f'Before adding adapters:\n{self.model.summarize()}')

        for _, module in self.model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    if adapter_key == AdapterName.MLP_INFUSED:
                        cfg = MLPInfusedAdapterConfig(
                            in_features=cfg.ffn_hidden_size
                            // cfg.tensor_model_parallel_size
                        )
                    elif adapter_key in [AdapterName.KEY_INFUSED, AdapterName.VALUE_INFUSED]:
                        cfg = InfusedAdapterConfig(
                            in_features=cfg.hidden_size
                            // cfg.tensor_model_parallel_size
                        )
                    else:
                        raise ValueError(f"Adapter Key {adapter_key} is unknown.")
                    if model_utils.import_class_by_path(cfg._target_) in module.get_accepted_adapter_types():
                        module.add_adapter(name=adapter_key, cfg=cfg)

        logging.info(f'After adding adapters:\n{self.model.summarize()}')

    @classmethod
    def list_available_models(cls):
        pass


class MegatronPTuningAdapterLearningModel(MegatronGPTBaseAdapterModel):
    """
    TODO: 
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.adapter_name_keys = [AdapterName.PTUNING_ADAPTER]

        logging.info(f'Before adding adapters:\n{self.model.summarize()}')

        self.adapter_cfg = PromptEncoderAdapterConfig(
            cfg.prompt_encoder_adapter.virtual_tokens,
            cfg.prompt_encoder_adapter.bottleneck_dim,
            cfg.prompt_encoder_adapter.embedding_dim,
            cfg.prompt_encoder_adapter.init_std,
            cfg.hidden_size,
        )

    def _init_adapters(self,):
        for module_name, module in self.model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    if model_utils.import_class_by_path(self.adapter_cfg._target_) in module.get_accepted_adapter_types():
                        module.add_adapter(
                            name=adapter_key, cfg=self.adapter_cfg,
                        )

        logging.info(f'After adding adapters:\n{self.summarize()}')


    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        state_dict_ = {}
        if self.first_stage_of_pipeline():
            state_dict_ = super().state_dict(destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.first_stage_of_pipeline():
            super().load_state_dict(state_dict, strict)

    def setup_optimizer_param_groups(self):
        if self.first_stage_of_pipeline():
            super().setup_optimizer_param_groups()
        else:
            self.model.freeze()  # Freeze the entire model
            self._optimizer_param_groups = ({'params': []},)
        logging.info(f'Optimizer groups set:\n{self.model.summarize()}')

    @classmethod
    def list_available_models(cls):
        pass
