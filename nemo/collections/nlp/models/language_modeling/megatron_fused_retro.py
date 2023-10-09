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
from nemo.collections.nlp.parts.utils_funcs import get_last_rank


from omegaconf.omegaconf import OmegaConf, open_dict
from nemo.collections.nlp.modules.common import VirtualPromptSource
from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_dataset import GPTPromptLearningDataset
from nemo.collections.nlp.data.language_modeling.megatron.retro_dataset import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
from nemo.collections.nlp.data.language_modeling.megatron.retro_prompt_learning_dataset import RetroPromptLearningDataset
from nemo.collections.nlp.data.language_modeling.megatron.retro_fine_tune_dataset import RetroQAFineTuneDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)

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
    LoraKQVAdapterConfig,
    LoraKVAdapterConfig,
    LoraQAdapterConfig,
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

__all__ = ['MegatronFusedRetrievalAdapterModel', 'MegatronFusedRetrievalLoraModel']


class MegatronFusedRetrievalAdapterModel(MegatronRetrievalModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)

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
        self.megatron_amp_o2 = cfg.get('megatron_amp_O2', False)
        
        if self._trainer.precision == 'bf16':
            self.autocast_dtype = torch.bfloat16
        elif int(self._trainer.precision) == 32:
            self.autocast_dtype = torch.float
        elif int(self._trainer.precision) == 16:
            self.autocast_dtype = torch.half
        else:
            raise ValueError('precision must be in [32, 16, "bf16"]')

        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.get('restore_from_path')):
            save_restore_connector.model_extracted_dir = cfg.get('restore_from_path')

        frozen_model_cfg = MegatronRetrievalModel.restore_from(
            cfg.get('restore_from_path'), trainer=trainer, return_config=True, save_restore_connector=save_restore_connector,
        )

        with open_dict(frozen_model_cfg):
            # work around for the fused softmax bug
            frozen_model_cfg.masked_softmax_fusion = False
            frozen_model_cfg.precision = trainer.precision


        # Need to overwrite some params in frozen model's config before restoring
        with open_dict(frozen_model_cfg):
            frozen_model_cfg.megatron_amp_O2 = False
            frozen_model_cfg.optim.name = "fused_adam"
            frozen_model_cfg.micro_batch_size = cfg.micro_batch_size
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
            frozen_model_cfg.task_templates = cfg["task_templates"]

        self.model = MegatronRetrievalModel.restore_from(
            cfg.get('restore_from_path'),
            trainer=trainer,
            save_restore_connector=save_restore_connector,
            override_config_path=frozen_model_cfg,
        ).to(dtype=self.autocast_dtype)

        for _, layer in self.model.named_modules():
            if hasattr(layer, 'activations_checkpoint_method'):
                layer.activations_checkpoint_method = (
                    None  # (@adithyare) adapter learning does not support activations checkpointing atm.
                )
        
        logging.info(f'Before adding adapters:\n{self.model.summarize()}')

        if cfg.adapter_tuning.type == "parallel_adapter":
            adapter_cfg = ParallelLinearAdapterConfig(
                in_features=frozen_model_cfg.hidden_size,
                out_features=frozen_model_cfg.hidden_size,
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

        self.enable_autocast = (
            True
        )

        self.model.freeze()
        if cfg.adapter_tuning.pre_decoder is True:
            logging.info(f'Adding pre decoder adapters')
            adapter_cfg = ParallelLinearAdapterConfig(
                in_features=frozen_model_cfg.hidden_size,
                out_features=frozen_model_cfg.hidden_size,
                dim=cfg.adapter_tuning.pre_decoder_size,
                norm_position=cfg.adapter_tuning.get('norm_position', 'pre'),
                norm_type=cfg.adapter_tuning.get('norm_type', 'mixedfusedlayernorm'),
                column_init_method=cfg.adapter_tuning.get('column_init_method', 'xavier'),
                row_init_method=cfg.adapter_tuning.get('row_init_method', 'zero'),
                dropout=cfg.adapter_tuning.adapter_dropout,
            )
            for _, module in self.model.model.pre_decoder.named_modules():
                if isinstance(module, adapter_mixins.AdapterModuleMixin):
                    self.add_adapters_init(module, adapter_cfg)
        if cfg.adapter_tuning.post_decoder is True:
            adapter_cfg = ParallelLinearAdapterConfig(
                in_features=frozen_model_cfg.hidden_size,
                out_features=frozen_model_cfg.hidden_size,
                dim=cfg.adapter_tuning.post_decoder_size,
                norm_position=cfg.adapter_tuning.get('norm_position', 'pre'),
                norm_type=cfg.adapter_tuning.get('norm_type', 'mixedfusedlayernorm'),
                column_init_method=cfg.adapter_tuning.get('column_init_method', 'xavier'),
                row_init_method=cfg.adapter_tuning.get('row_init_method', 'zero'),
                dropout=cfg.adapter_tuning.adapter_dropout,
            )
            logging.info(f'Adding post decoder adapters')
            for _, module in self.model.model.post_decoder.named_modules():
                if isinstance(module, adapter_mixins.AdapterModuleMixin):
                    self.add_adapters_init(module, adapter_cfg)
        if cfg.adapter_tuning.encoder is True:
            adapter_cfg = ParallelLinearAdapterConfig(
                in_features=frozen_model_cfg.hidden_size,
                out_features=frozen_model_cfg.hidden_size,
                dim=cfg.adapter_tuning.encoder_size,
                norm_position=cfg.adapter_tuning.get('norm_position', 'pre'),
                norm_type=cfg.adapter_tuning.get('norm_type', 'mixedfusedlayernorm'),
                column_init_method=cfg.adapter_tuning.get('column_init_method', 'xavier'),
                row_init_method=cfg.adapter_tuning.get('row_init_method', 'zero'),
                dropout=cfg.adapter_tuning.adapter_dropout,
            )
            logging.info(f'Adding encoder adapters')
            for _, module in self.model.model.encoder.named_modules():
                if isinstance(module, adapter_mixins.AdapterModuleMixin):
                    self.add_adapters_init(module, adapter_cfg)

        logging.info(f'After adding adapters:\n{self.model.summarize()}')

        # for name, module in self.model.named_modules():
        #     logging.info(f'Module name:\n{name}{module}')

        logging.info("Done")
        # self.model = self.frozen_model
        # if cfg.eval == True:
        #     self.load_adapters(strict=False)

        # self.model.freeze()

    def add_adapters_init(self, module, adapter_cfg):
        for adapter_key in self.adapter_name_keys:
            if model_utils.import_class_by_path(adapter_cfg._target_) in module.get_accepted_adapter_types():
                module.add_adapter(
                    name=adapter_key, cfg=adapter_cfg,
                )

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        Creates a state_dict using only the adapter parameters.
        This ensures that this wrapper class will only checkpoint the adapter
        weights and not the rest of the base GPT Model.
        """
        state_dict_ = {}
        for name, module in self.model.named_modules():
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
        for name, module in self.model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.get_adapter_module(adapter_key)
                    if adapter_module:
                        # state_adapter_key = '.'.join(["frozen_model", name, "adapter_layer", adapter_key])
                        # temp_state_dict = {
                        #     "layer_norm.weight": state_dict[state_adapter_key + '.layer_norm.weight'],
                        #     "layer_norm.bias": state_dict[state_adapter_key + '.layer_norm.bias'],
                        #     "linear_in.weight": state_dict[state_adapter_key + '.linear_in.weight'],
                        #     "linear_out.weight": state_dict[state_adapter_key + '.linear_out.weight']
                        # }
                        temp_state_dict = state_dict[name + ":" + adapter_key]
                        adapter_module.load_state_dict(temp_state_dict, strict)
                        # adapter_module.load_state_dict(state_dict[state_adapter_key], strict)
                module.set_enabled_adapters(enabled=True)

    def forward(
        self,
        input_ids,
        input_attn_mask,
        retrieved_ids,
        retrieved_attn_mask,
        token_type_ids=None,
        labels=None,
        input_emb=None,
        position_ids=None,
    ):
        output_tensor = self.model(
            input_ids=input_ids,
            input_attn_mask=input_attn_mask,
            retrieved_ids=retrieved_ids,
            retrieved_attn_mask=retrieved_attn_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            input_emb=input_emb,
            position_ids=position_ids,
        )
        return output_tensor

    def create_state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        Creates a state_dict using only the adapter parameters.
        This ensures that this wrapper class will only checkpoint the adapter
        weights and not the rest of the base GPT Model.
        """
        state_dict_ = {}
        for name, module in self.model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.get_adapter_module(adapter_key)
                    if adapter_module:
                        state_adapter_key = ':'.join([name, adapter_key])
                        state_dict_[state_adapter_key] = adapter_module.state_dict()

                module.set_enabled_adapters(enabled=True)
        return state_dict_

    def load_adapters(self, strict: bool = True):
        """
        Loads a state_dict expecting the state_dict to contain key,values 
        only for the adapter parameters.
        """
        state_dict = self.create_state_dict()
        for name, module in self.model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.get_adapter_module(adapter_key)
                    if adapter_module:
                        state_adapter_key = ':'.join([name, adapter_key])
                        adapter_module.load_state_dict(state_dict[state_adapter_key], strict)
                module.set_enabled_adapters(enabled=True)


    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    @classmethod
    def list_available_models(cls):
        pass


class MegatronFusedRetrievalLoraModel(MegatronRetrievalModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)

        encoder_adapter_name_keys = [AdapterName.LORA_KQV_ADAPTER]
        decoder_adapter_name_keys = [
            AdapterName.LORA_KQV_ADAPTER,
            AdapterName.LORA_KV_ADAPTER,
            AdapterName.LORA_Q_ADAPTER,
        ]
        # add adapter keys to the list -> to update state dict
        self.adapter_name_keys = encoder_adapter_name_keys
        # self.adapter_name_keys = encoder_adapter_name_keys + decoder_adapter_name_keys
        
        if self._trainer.precision == 'bf16':
            self.autocast_dtype = torch.bfloat16
        elif int(self._trainer.precision) == 32:
            self.autocast_dtype = torch.float
        elif int(self._trainer.precision) == 16:
            self.autocast_dtype = torch.half
        else:
            raise ValueError('precision must be in [32, 16, "bf16"]')

        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.get('restore_from_path')):
            save_restore_connector.model_extracted_dir = cfg.get('restore_from_path')

        frozen_model_cfg = MegatronRetrievalModel.restore_from(
            cfg.get('restore_from_path'), trainer=trainer, return_config=True, save_restore_connector=save_restore_connector,
        )

        with open_dict(frozen_model_cfg):
            # work around for the fused softmax bug
            frozen_model_cfg.masked_softmax_fusion = False
            frozen_model_cfg.precision = trainer.precision


        # Need to overwrite some params in frozen model's config before restoring
        with open_dict(frozen_model_cfg):
            frozen_model_cfg.megatron_amp_O2 = False
            frozen_model_cfg.optim.name = "fused_adam"
            frozen_model_cfg.micro_batch_size = cfg.micro_batch_size
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
            frozen_model_cfg.task_templates = cfg["task_templates"]

        self.model = MegatronRetrievalModel.restore_from(
            cfg.get('restore_from_path'),
            trainer=trainer,
            save_restore_connector=save_restore_connector,
            override_config_path=frozen_model_cfg,
        ).to(dtype=self.autocast_dtype)

        for _, layer in self.model.named_modules():
            if hasattr(layer, 'activations_checkpoint_method'):
                layer.activations_checkpoint_method = (
                    None  # (@adithyare) adapter learning does not support activations checkpointing atm.
                )
        
        logging.info(f'Before adding adapters:\n{self.model.summarize()}')

        self.model.freeze()
        # if cfg.adapter_tuning.pre_decoder is True:
        #     logging.info(f'Adding pre decoder adapters')
        #     adapter_cfg = self._get_adapter_cfg(cfg.adapter_tuning.adapter_key, frozen_model_cfg.hidden_size, frozen_model_cfg.num_attention_heads, cfg.adapter_tuning.adapter_dim)
        #     for _, module in self.model.model.pre_decoder.named_modules():
        #         if isinstance(module, adapter_mixins.AdapterModuleMixin):
        #             self.add_adapters_init(module, adapter_cfg)
        # if cfg.adapter_tuning.post_decoder is True:
        #     adapter_cfg = self._get_adapter_cfg(cfg.adapter_tuning.adapter_key, frozen_model_cfg.hidden_size, frozen_model_cfg.num_attention_heads, cfg.adapter_tuning.adapter_dim)
        #     logging.info(f'Adding post decoder adapters')
        #     for _, module in self.model.model.post_decoder.named_modules():
        #         if isinstance(module, adapter_mixins.AdapterModuleMixin):
        #             self.add_adapters_init(module, adapter_cfg)
        # if cfg.adapter_tuning.encoder is True:
        #     adapter_cfg = self._get_adapter_cfg(cfg.adapter_tuning.adapter_key, frozen_model_cfg.hidden_size, frozen_model_cfg.num_attention_heads, cfg.adapter_tuning.adapter_dim)
        #     logging.info(f'Adding encoder adapters')
        #     for _, module in self.model.model.encoder.named_modules():
        #         if isinstance(module, adapter_mixins.AdapterModuleMixin):
        #             self.add_adapters_init(module, adapter_cfg)

        logging.info(f'Adding pre decoder adapters')
        adapter_cfg = self._get_adapter_cfg(cfg.adapter_tuning.adapter_key, frozen_model_cfg.hidden_size, frozen_model_cfg.num_attention_heads, cfg.adapter_tuning.adapter_dim)
        for _, module in self.model.model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                self.add_adapters_init(module, adapter_cfg)

        logging.info(f'After adding adapters:\n{self.model.summarize()}')

        # for name, module in self.model.named_modules():
        #     logging.info(f'Module name:\n{name}{module}')

        logging.info("Done")
        # self.model = self.frozen_model
        # if cfg.eval == True:
        #     self.load_adapters(strict=False)

        self.enable_autocast = (
            True
        )

        # self.model.freeze()

    def add_adapters_init(self, module, adapter_cfg):
        for adapter_key in self.adapter_name_keys:
            if model_utils.import_class_by_path(adapter_cfg._target_) in module.get_accepted_adapter_types():
                module.add_adapter(
                    name=adapter_key, cfg=adapter_cfg,
                )

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        Creates a state_dict using only the adapter parameters.
        This ensures that this wrapper class will only checkpoint the adapter
        weights and not the rest of the base GPT Model.
        """
        state_dict_ = {}
        for name, module in self.model.named_modules():
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
        for name, module in self.model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.get_adapter_module(adapter_key)
                    if adapter_module:
                        # state_adapter_key = '.'.join(["frozen_model", name, "adapter_layer", adapter_key])
                        # temp_state_dict = {
                        #     "layer_norm.weight": state_dict[state_adapter_key + '.layer_norm.weight'],
                        #     "layer_norm.bias": state_dict[state_adapter_key + '.layer_norm.bias'],
                        #     "linear_in.weight": state_dict[state_adapter_key + '.linear_in.weight'],
                        #     "linear_out.weight": state_dict[state_adapter_key + '.linear_out.weight']
                        # }
                        temp_state_dict = state_dict[name + ":" + adapter_key]
                        adapter_module.load_state_dict(temp_state_dict, strict)
                        # adapter_module.load_state_dict(state_dict[state_adapter_key], strict)
                module.set_enabled_adapters(enabled=True)

    def _get_adapter_cfg(self, adapter_key, hidden_size, num_attention_heads, adapter_dim):
        assert (
            hidden_size % num_attention_heads == 0
        ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
        kv_channels = hidden_size // num_attention_heads
        projection_size = kv_channels * num_attention_heads

        if adapter_key == AdapterName.LORA_KQV_ADAPTER:
            adapter_cfg = LoraKQVAdapterConfig(
                in_features=hidden_size,
                out_features=3 * projection_size,
                dim=adapter_dim,
                norm_position="none",
                norm_type="none",
                activation="identity",
                column_init_method="normal",
                row_init_method="zero",
                gather_output=False,
                dropout=0.0,
            )
        elif adapter_key == AdapterName.LORA_KV_ADAPTER:
            adapter_cfg = LoraKVAdapterConfig(
                in_features=hidden_size,
                out_features=2 * projection_size,
                dim=adapter_dim,
                norm_position="none",
                norm_type="none",
                activation="identity",
                column_init_method="normal",
                row_init_method="zero",
                gather_output=False,
                dropout=0.0,
            )
        elif adapter_key == AdapterName.LORA_Q_ADAPTER:
            adapter_cfg = LoraQAdapterConfig(
                in_features=hidden_size,
                out_features=1 * projection_size,
                dim=adapter_dim,
                norm_position="none",
                norm_type="none",
                activation="identity",
                column_init_method="normal",
                row_init_method="zero",
                gather_output=False,
                dropout=0.0,
            )
        else:
            raise RuntimeError("Unexpected adapter key name..")

        return adapter_cfg

    def forward(
        self,
        input_ids,
        input_attn_mask,
        retrieved_ids,
        retrieved_attn_mask,
        token_type_ids=None,
        labels=None,
        input_emb=None,
        position_ids=None,
    ):
        output_tensor = self.model(
            input_ids=input_ids,
            input_attn_mask=input_attn_mask,
            retrieved_ids=retrieved_ids,
            retrieved_attn_mask=retrieved_attn_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            input_emb=input_emb,
            position_ids=position_ids,
        )
        return output_tensor

    def create_state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        Creates a state_dict using only the adapter parameters.
        This ensures that this wrapper class will only checkpoint the adapter
        weights and not the rest of the base GPT Model.
        """
        state_dict_ = {}
        for name, module in self.model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.get_adapter_module(adapter_key)
                    if adapter_module:
                        state_adapter_key = ':'.join([name, adapter_key])
                        state_dict_[state_adapter_key] = adapter_module.state_dict()

                module.set_enabled_adapters(enabled=True)
        return state_dict_

    def load_adapters(self, strict: bool = True):
        """
        Loads a state_dict expecting the state_dict to contain key,values 
        only for the adapter parameters.
        """
        state_dict = self.create_state_dict()
        for name, module in self.model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.get_adapter_module(adapter_key)
                    if adapter_module:
                        state_adapter_key = ':'.join([name, adapter_key])
                        adapter_module.load_state_dict(state_dict[state_adapter_key], strict)
                module.set_enabled_adapters(enabled=True)


    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    @classmethod
    def list_available_models(cls):
        pass


def build_all_datasets(
    cfg, tokenizer, train_valid_test_num_samples,
):
    """Build train, valid, and test RETRO datasets.
       There is one to one mapping between data_prefix and knn_map_path.
       Currently only supports one retrieval dataset.
    """
    train_dataset = RetroQAFineTuneDataset(
        cfg.train_ds.get('file_name'),
        tokenizer,
        cfg.train_ds.get('answer_only_loss'),
        tokenizer.pad_id,
        cfg.train_ds.get('seq_length'),
        cfg.train_ds.get('add_bos'),
        cfg.train_ds.get('add_eos'),
        train_valid_test_num_samples[0],
        cfg.train_ds.get('seed'),
        cfg.train_ds.get('neighbors'),
    )
    val_dataset = RetroQAFineTuneDataset(
        cfg.val_ds.get('file_name'),
        tokenizer,
        cfg.val_ds.get('answer_only_loss'),
        tokenizer.pad_id,
        cfg.val_ds.get('seq_length'),
        cfg.val_ds.get('add_bos'),
        cfg.val_ds.get('add_eos'),
        train_valid_test_num_samples[1],
        cfg.val_ds.get('seed'),
        cfg.val_ds.get('neighbors'),
    )
    test_dataset = RetroQAFineTuneDataset(
        cfg.test_ds.get('file_name'),
        tokenizer,
        cfg.test_ds.get('answer_only_loss'),
        tokenizer.pad_id,
        cfg.test_ds.get('seq_length'),
        cfg.test_ds.get('add_bos'),
        cfg.test_ds.get('add_eos'),
        train_valid_test_num_samples[2],
        cfg.test_ds.get('seed'),
        cfg.test_ds.get('neighbors'),
    )

    return train_dataset, val_dataset, test_dataset