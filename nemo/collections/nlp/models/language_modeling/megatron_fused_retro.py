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

        self.model.freeze()
        if cfg.adapter_tuning.pre_decoder is True:
            logging.info(f'Adding pre decoder adapters')
            adapter_cfg = ParallelLinearAdapterConfig(
                in_features=frozen_model_cfg.hidden_size,
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

    # def build_train_valid_test_datasets(self):
    #     logging.info('Building RETRO datasets.')
    #     global_batch_size = self.trainer.world_size * self.cfg.micro_batch_size // self.cfg.tensor_model_parallel_size
    #     # Compute trianing micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
    #     max_train_steps = self.trainer.max_steps * self.trainer.accumulate_grad_batches
    #     eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
    #     test_iters = int(self.trainer.limit_test_batches)

    #     train_valid_test_num_samples = [
    #         max_train_steps * global_batch_size,
    #         eval_iters * global_batch_size,
    #         test_iters * global_batch_size,
    #     ]

    #     self._train_ds, self._validation_ds, self._test_ds = build_all_datasets(
    #         cfg=self.cfg.data, tokenizer=self.tokenizer, train_valid_test_num_samples=train_valid_test_num_samples,
    #     )
    #     if self._train_ds is not None:
    #         logging.info(f'Length of train dataset: {len(self._train_ds)}')
    #     if self._validation_ds is not None:
    #         logging.info(f'Length of val dataset: {len(self._validation_ds)}')
    #     if self._test_ds is not None:
    #         logging.info(f'Length of test dataset: {len(self._test_ds)}')
    #     logging.info(f'Finished building RETRO datasets.')
    #     return self._train_ds, self._validation_ds, self._test_ds

    # def build_pretraining_data_loader(self, dataset, consumed_samples):
    #     if isinstance(dataset, BlendableDataset):
    #         collate_fun = dataset.datasets[0].collate_fn
    #     else:
    #         collate_fun = dataset.collate_fn

    #     collate_fn = partial(collate_fun, tp_workers=0)
    #     global_batch_size = self.trainer.world_size * self.cfg.micro_batch_size // self.cfg.tensor_model_parallel_size
    #     batch_sampler = MegatronPretrainingBatchSampler(
    #         total_samples=len(dataset),
    #         consumed_samples=consumed_samples,
    #         micro_batch_size=self.cfg.micro_batch_size,
    #         global_batch_size=global_batch_size,
    #         data_parallel_rank=parallel_state.get_data_parallel_rank(),
    #         data_parallel_size=parallel_state.get_data_parallel_world_size(),
    #         drop_last=True,
    #     )
    #     return torch.utils.data.DataLoader(
    #         dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=0, pin_memory=True,
    #     )

    # def build_virtual_prompt_dataset(
    #     self,
    #     data,
    #     batch_size,
    #     max_seq_length=2048,
    #     min_seq_length=1,
    #     add_bos=False,
    #     add_eos=False,
    #     for_train=True,
    #     drop_last=False,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=False,
    #     tokens_to_generate=None,
    #     get_dataset_only=False,
    #     cache_data_path=None,
    #     load_cache=False,
    #     num_neighbors=2,
    #     retrieved_doc_len = 128 
    # ):
    #     task_template = {
    #         "car": {
    #             "prompt_template": " \nQuestion: {question} \nAnswer: {answer}",
    #             "prompt_template_fields": ["question", "answer"],
    #             "total_virtual_tokens": 0,
    #             "virtual_token_splits": [],
    #             "truncate_field": "question",
    #             "answer_only_loss": True,
    #             "answer_field": "answer"
    #     }}
       
    #     dataset = RetroPromptLearningDataset(
    #         data=data,
    #         tokenizer=self.tokenizer,
    #         virtual_prompt_source= VirtualPromptSource.NO_PROMPT,
    #         task_templates=task_template,
    #         pseudo_tokens=[],
    #         pad_token_id=self.model.tokenizer.eos_id,
    #         max_seq_length=max_seq_length,
    #         min_seq_length=min_seq_length,
    #         add_bos=add_bos,
    #         add_eos=add_eos,
    #         for_train=for_train,
    #         tokens_to_generate=tokens_to_generate,
    #         cache_data_path=cache_data_path,  # the cache file
    #         load_cache=load_cache, # whether to load from the cache if it is available
    #         seed=1234,
    #         neighbors=num_neighbors,
    #         megatron_lm_compatible=False,
    #         retrieved_doc_len = retrieved_doc_len
    #     )

    #     if get_dataset_only:
    #         return dataset

    #     # Make distributed dataloader
    #     rank = parallel_state.get_data_parallel_rank()
    #     data_parallel_size = parallel_state.get_data_parallel_world_size()
    #     sampler = torch.utils.data.distributed.DistributedSampler(
    #         dataset, num_replicas=data_parallel_size, rank=rank, shuffle=shuffle, seed=self.cfg.seed
    #     )

    #     assert batch_size % data_parallel_size == 0, "Global batch size must be evenly divisible by data parallel size"

    #     if for_train:
    #         if self.cfg.get("sequence_parallel", False):
    #             collate_fn = partial(
    #                 dataset.collate_fn, tp_workers=parallel_state.get_tensor_model_parallel_world_size()
    #             )
    #         else:
    #             collate_fn = partial(dataset.collate_fn, tp_workers=0)
    #     else:
    #         collate_fn = dataset.inference_collate_fn

    #     dataloader = torch.utils.data.DataLoader(
    #         dataset,
    #         collate_fn=collate_fn,
    #         sampler=sampler,
    #         batch_size=batch_size // data_parallel_size,
    #         drop_last=drop_last,
    #         num_workers=num_workers,
    #         pin_memory=pin_memory,
    #         persistent_workers=True,  # (@adithyare and @eharper) We need this to make spawn=True to work.
    #     )

    #     return dataset, dataloader


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
