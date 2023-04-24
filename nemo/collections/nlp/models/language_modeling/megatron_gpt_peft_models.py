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


import abc
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    InfusedAdapterConfig,
    MLPInfusedAdapterConfig,
    ParallelLinearAdapterConfig,
    PromptEncoderAdapterConfig,
)
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import logging, model_utils


class MegatronGPTPEFTModel(MegatronGPTSFTModel):
    """
    base class for all mixin based adapter models
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.setup_complete = False
        self.base_keys = self.get_all_keys()
        self.init_peft_modules()
        self.adapter_keys = self.get_all_keys() - self.base_keys

    def first_stage_of_pipeline(self):
        return self.model.pre_process

    @abc.abstractmethod
    def init_peft_modules(self,):
        return

    def setup(self, stage=None):
        super().setup(stage)
        self.setup_complete = True

    def get_all_keys(self,):
        """ 
        Returns all the keys in the model
        """
        k = [n for n, p in self.named_parameters()]
        return set(k)

    def get_peft_state_dict(self,):
        """ 
        Gets the keys associated with the adapters only.
        """
        state_dict = self.model.state_dict(prefix="model.")
        peft_state_dict = {}
        for k in self.adapter_keys:
            peft_state_dict[k] = state_dict[k]
        return peft_state_dict

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        if self.setup_complete:
            # Once setup is complete we no longer need to track the frozen part of the model. Only there adapter state dict keeps changing so state_dict only track these.
            return self.get_peft_state_dict()
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

        self._optimizer_param_groups = ({"params": opt_params},)
        logging.info(f"Optimizer groups set:\n{self.summarize()}")


class MegatronGPTAdapterModel(MegatronGPTPEFTModel):
    """
    MegatronGPTAdapterLearningModel is a model that combines a base model (GPTModel) with a adapters.
    This class only supports the canonical Adapter training described in Houlsby et al. (https://arxiv.org/pdf/1902.00751.pdf)

    Two adapter's are inserted into each Transformer layer in the base GPT Model.

    It is assumed that these set of adapters will then be trained for a specific task.
    Once trained, the adapter weights will be saved and can be re-loaded 
    and infused into the same GPT Model for inference. 
    """

    def __init__(
        self, cfg: DictConfig, trainer: Trainer,
    ):
        self.adapter_name_keys = [
            AdapterName.PRE_ATTN_ADAPTER,
            AdapterName.POST_ATTN_ADAPTER,
        ]
        adapter_tuning_cfg = cfg.peft.adapter_tuning

        if adapter_tuning_cfg.type == "parallel_adapter":
            adapter_cfg = ParallelLinearAdapterConfig(
                in_features=cfg.hidden_size,
                dim=adapter_tuning_cfg.adapter_dim,
                norm_position=adapter_tuning_cfg.get("norm_position", "pre"),
                norm_type=adapter_tuning_cfg.get("norm_type", "mixedfusedlayernorm"),
                column_init_method=adapter_tuning_cfg.get("column_init_method", "xavier"),
                row_init_method=adapter_tuning_cfg.get("row_init_method", "zero"),
                dropout=adapter_tuning_cfg.adapter_dropout,
            )
        else:
            adapter_cfg = LinearAdapterConfig(
                in_features=cfg.hidden_size,
                dim=adapter_tuning_cfg.adapter_dim,
                norm_position=adapter_tuning_cfg.get("norm_position", "pre"),
                dropout=adapter_tuning_cfg.adapter_dropout,
            )

        self.name_key_to_cfg = {}
        for k in self.adapter_name_keys:
            self.name_key_to_cfg[k] = adapter_cfg

        super().__init__(cfg, trainer)

    def init_peft_modules(self):
        """ 
        Randomly initialize the adapter params and add them to the appropriate modules.
        """
        logging.info(f"Before adding adapters:\n{self.summarize()}")
        for _, module in self.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    adapter_cfg = self.name_key_to_cfg[adapter_key]
                    if model_utils.import_class_by_path(adapter_cfg._target_) in module.get_accepted_adapter_types():
                        module.add_adapter(
                            name=adapter_key, cfg=adapter_cfg,
                        )
        logging.info(f"After adding adapters:\n{self.summarize()}")
        return True


class MegatronGPTIA3Model(MegatronGPTPEFTModel):
    """
    MegatronGPTInfusedAdapterModel is a model that combines a base model (GPTModel) with a "Infused Adapter that can Inhibiting and Amplify Inner Activations", known as IA3.
    This class supports the addition of IA3 into a transformer based LM as described in Liu et al. (https://arxiv.org/pdf/2205.05638.pdf)

    Three adapter's are inserted into each Transformer layer in the base GPT Model. Each adapter is basically a vector that simply scales the key, value or ffn hidden representations.

    It is assumed that these set of adapters will then be trained for a specific task.
    Once trained, the adapter weights will be saved and can be re-loaded 
    and infused into the same GPT Model for inference. 
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self.adapter_name_keys = [AdapterName.KEY_INFUSED, AdapterName.VALUE_INFUSED, AdapterName.MLP_INFUSED]

        mlp_infused_adapter_cfg = MLPInfusedAdapterConfig(
            in_features=cfg.ffn_hidden_size // cfg.tensor_model_parallel_size
        )
        infused_adapter_cfg = InfusedAdapterConfig(in_features=cfg.hidden_size // cfg.tensor_model_parallel_size)

        self.name_key_to_cfg = {}
        for k in self.adapter_name_keys:
            if k == AdapterName.MLP_INFUSED:
                self.name_key_to_cfg[k] = mlp_infused_adapter_cfg
            elif k in [
                AdapterName.KEY_INFUSED,
                AdapterName.VALUE_INFUSED,
            ]:
                self.name_key_to_cfg[k] = infused_adapter_cfg
            else:
                raise ValueError(f"Adapter Key {k} is unknown.")
        super().__init__(cfg, trainer)

    def init_peft_modules(self):
        logging.info(f"Before adding adapters:\n{self.summarize()}")

        for _, module in self.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    cfg = self.name_key_to_cfg[adapter_key]
                    if model_utils.import_class_by_path(cfg._target_) in module.get_accepted_adapter_types():
                        module.add_adapter(name=adapter_key, cfg=cfg)

        logging.info(f"After adding adapters:\n{self.summarize()}")
        return True


class MegatronGPTPTuningModel(MegatronGPTPEFTModel):
    """
    Mixin based implementation of p-tuning model for PEFT tuning.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self.adapter_name_keys = [AdapterName.PTUNING_ADAPTER]

        adapter_cfg = PromptEncoderAdapterConfig(
            cfg.peft.p_tuning.virtual_tokens,
            cfg.peft.p_tuning.bottleneck_dim,
            cfg.peft.p_tuning.embedding_dim,
            cfg.peft.p_tuning.init_std,
            cfg.hidden_size,
        )
        self.name_key_to_cfg = {AdapterName.PTUNING_ADAPTER: adapter_cfg}
        super().__init__(cfg, trainer)
        self.virtual_tokens = cfg.peft.p_tuning.virtual_tokens

    def init_peft_modules(self,):
        if not self.first_stage_of_pipeline():
            # There are no params to add if we are not in the first state of the pipeline
            return True

        logging.info(f"Before adding adapters:\n{self.summarize()}")

        for _, module in self.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    cfg = self.name_key_to_cfg[adapter_key]
                    if model_utils.import_class_by_path(cfg._target_) in module.get_accepted_adapter_types():
                        module.add_adapter(
                            name=adapter_key, cfg=cfg,
                        )

        logging.info(f"After adding adapters:\n{self.summarize()}")
        return True

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        if self.setup_complete:
            if self.first_stage_of_pipeline():
                return self.get_peft_state_dict()
            else:
                # if we are not in the first state of pipeline after setup is done
                # there should be no params in the state_dict
                return {}
        else:
            return self.model.state_dict(prefix="model.")

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.setup_complete:
            if self.first_stage_of_pipeline():
                assert set(state_dict.keys()) == self.adapter_keys
                super().load_state_dict(state_dict, strict=False)
            else:
                # if we are not in the first state of pipeline after setup is done
                # there should be no params to load...
                pass
        else:
            super().load_state_dict(state_dict, strict=True)

    def setup_optimizer_param_groups(self):
        if self.first_stage_of_pipeline():
            super().setup_optimizer_param_groups()
        else:
            self.freeze()  # Freeze the entire model
            self._optimizer_param_groups = ({"params": []},)
        logging.info(f"Optimizer groups set:\n{self.summarize()}")
