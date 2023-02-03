import logging
import re
from functools import partial
from typing import List, Optional
import os

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.trainer.trainer import Trainer
from torch import masked_select
from torch.nn.utils.rnn import pad_sequence

from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning.utilities import model_summary
from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_dataset import GPTPromptLearningDataset
from nemo.collections.nlp.data.language_modeling.megatron.retro_dataset import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
from nemo.collections.nlp.modules.common import VirtualPromptPlaceholderToken, VirtualPromptSource, VirtualPromptStyle
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    build_position_ids,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core import adapter_mixins

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    InfusedAdapterConfig,
    MLPInfusedAdapterConfig,
    ParallelLinearAdapterConfig,
)
from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.utils import logging, model_utils

try:
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
        forward_backward_pipelining_without_interleaving,
    )
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


# Fuse adapters and prompt learning with retro model

# Model initalizes both adapter and retro mdoel classes
# The forward function here calls the adapter forward function gets the output and uses that as the input to some retro subclass


class MegatronFusedRetrievalAdapterModel(MegatronRetrievalModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        # if adapter_mixins.get_registered_adapter(MegatronRetrievalModel) is None:
        #     adapter_mixins.register_adapter(MegatronRetrievalModel, MegatronFusedRetrievalAdapterModel)

        assert cfg.adapter_tuning.get('adapter_dim', 0) > 0, "adapter_dim has not been set."
        assert (
            cfg.adapter_tuning.adapter_dim % cfg.tensor_model_parallel_size == 0
        ), "The adapter dim should be divisible by tensor_model_parallel_size."
        assert cfg.adapter_tuning.type in [
            'linear_adapter',
            'parallel_adapter',
        ], "Adapter type should be 'linear_adapter' or 'parallel_adapter'"

        self.adapter_name_keys = [AdapterName.PRE_ATTN_ADAPTER, AdapterName.POST_ATTN_ADAPTER]
        # frozen_model_cfg = MegatronGPTModel.restore_from(
        #     cfg.get('restore_from_path'), trainer=trainer, return_config=True
        # )



        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.get('restore_from_path')):
            save_restore_connector.model_extracted_dir = cfg.get('restore_from_path')
        frozen_model_cfg = MegatronGPTModel.restore_from(
            cfg.get('restore_from_path'),
            trainer=trainer,
            return_config=True,
            # save_restore_connector=save_restore_connector,
        )
        # Need to overwrite some params in frozen model's config before restoring
        with open_dict(frozen_model_cfg):
            frozen_model_cfg.megatron_amp_O2 = False
            frozen_model_cfg.optim.name = "fused_adam"
            frozen_model_cfg.micro_batch_size = self.cfg.micro_batch_size
            # frozen_model_cfg.global_batch_size = self.cfg.global_batch_size
            frozen_model_cfg.precision = trainer.precision
            frozen_model_cfg.sequence_parallel = self.cfg.get("sequence_parallel", False)
            frozen_model_cfg.activations_checkpoint_granularity = self.cfg.get(
                "activations_checkpoint_granularity", None
            )
            frozen_model_cfg.activations_checkpoint_num_layers = self.cfg.get(
                "activations_checkpoint_num_layers", None
            )
            frozen_model_cfg.activations_checkpoint_method = self.cfg.get("activations_checkpoint_method", None)

        if self.trainer.precision == 'bf16':
            self.autocast_dtype = torch.bfloat16
        elif int(self.trainer.precision) == 32:
            self.autocast_dtype = torch.float
        elif int(self.trainer.precision) == 16:
            self.autocast_dtype = torch.half
        else:
            raise ValueError('precision must be in [32, 16, "bf16"]')

        if cfg.get('restore_from_path', None):
            self.frozen_model = MegatronGPTModel.restore_from(
                cfg.get('restore_from_path'),
                trainer=trainer,
                save_restore_connector=save_restore_connector,
                override_config_path=frozen_model_cfg,
            ).to(dtype=self.autocast_dtype)


        for _, layer in self.frozen_model.named_modules():
            if hasattr(layer, 'activations_checkpoint_method'):
                layer.activations_checkpoint_method = (
                    None  # (@adithyare) adapter learning does not support activations checkpointing atm.
                )
        
        logging.info(f'Before adding adapters:\n{self.frozen_model.summarize()}')

        if cfg.adapter_tuning.type == "parallel_adapter":
            adapter_cfg = ParallelLinearAdapterConfig(
                in_features=frozen_model_cfg.hidden_size,
                dim=cfg.adapter_tuning.adapter_dim,
                norm_position=cfg.adapter_tuning.get('norm_position', 'pre'),
                norm_type=cfg.adapter_tuning.get('norm_type', 'mixedfusedlayernorm'),
                column_init_method=cfg.adapter_tuning.get('column_init_method', 'xavier'),
                row_init_method=cfg.adapter_tuning.get('row_init_method', 'zero'),
                dropout=cfg.adapter_tuning.adapter_dropout,
            )
        else:
            adapter_cfg = LinearAdapterConfig(
                in_features=frozen_model_cfg.hidden_size,
                dim=cfg.adapter_tuning.adapter_dim,
                norm_position=cfg.adapter_tuning.get('norm_position', 'pre'),
                dropout=cfg.adapter_tuning.adapter_dropout,
            )

        self.frozen_model.freeze()
        for _, module in self.frozen_model.named_modules():
            logging.info(f'Named modules:\n{module}')
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    if model_utils.import_class_by_path(adapter_cfg._target_) in module.get_accepted_adapter_types():
                        module.add_adapter(
                            name=adapter_key, cfg=adapter_cfg,
                        )

        logging.info(f'After adding adapters:\n{self.frozen_model.summarize()}')

        for name, module in self.frozen_model.named_modules():
            logging.info(f'Module name:\n{name}{module}')

        logging.info("Done")

    def load_model_state_dict(self, checkpoint) -> None:
        self.load_state_dict(state_dict())


    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        Creates a state_dict using only the adapter parameters.
        This ensures that this wrapper class will only checkpoint the adapter
        weights and not the rest of the base GPT Model.
        """
        state_dict_ = {}
        for name, module in self.frozen_model.named_modules():
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
        for name, module in self.frozen_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.get_adapter_module(adapter_key)
                    if adapter_module:
                        state_adapter_key = ':'.join([name, adapter_key])
                        stact_dict_output = state_dict[state_adapter_key]
                        adapter_module.load_state_dict(stact_dict_output, strict)
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
        self.frozen_model.freeze()  # Freeze the entire model
        opt_params = []
        for _, module in self.frozen_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                module.set_enabled_adapters(enabled=True)
                module.unfreeze_enabled_adapters()  # selectively unfreeze the adapter modules.
                opt_params += [p for p in module.parameters()]

        self._optimizer_param_groups = [{'params': opt_params}]
        logging.info(f'Optimizer groups set:\n{self.frozen_model.summarize()}')

    @classmethod
    def list_available_models(cls):
        pass