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
from pytorch_lightning.utilities import model_summary
from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_dataset import GPTPromptLearningDataset
from nemo.collections.nlp.data.language_modeling.megatron.retro_dataset import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
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
            for _, module in self.model.model.pre_decoder.named_modules():
                logging.info(f'Named modules:\n{module}')
                if isinstance(module, adapter_mixins.AdapterModuleMixin):
                    self.add_adapters_init(module, adapter_cfg)
        if cfg.adapter_tuning.post_decoder is True:
            for _, module in self.model.model.post_decoder.named_modules():
                logging.info(f'Named modules:\n{module}')
                if isinstance(module, adapter_mixins.AdapterModuleMixin):
                    self.add_adapters_init(module, adapter_cfg)
        if cfg.adapter_tuning.encoder is True:
            for _, module in self.model.model.encoder.named_modules():
                logging.info(f'Named modules:\n{module}')
                if isinstance(module, adapter_mixins.AdapterModuleMixin):
                    self.add_adapters_init(module, adapter_cfg)

        logging.info(f'After adding adapters:\n{self.model.summarize()}')

        for name, module in self.model.named_modules():
            logging.info(f'Module name:\n{name}{module}')

        logging.info("Done")
        # self.model = self.frozen_model
        if cfg.eval == True:
            self.load_adapters(strict=False)

    def add_adapters_init(self, module, adapter_cfg):
        for adapter_key in self.adapter_name_keys:
            if model_utils.import_class_by_path(adapter_cfg._target_) in module.get_accepted_adapter_types():
                module.add_adapter(
                    name=adapter_key, cfg=adapter_cfg,
                )

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


    def build_train_valid_test_datasets(self):
        logging.info('Building RETRO datasets.')
        global_batch_size = self.trainer.world_size * self.cfg.micro_batch_size // self.cfg.tensor_model_parallel_size
        # Compute trianing micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
        max_train_steps = self.trainer.max_steps * self.trainer.accumulate_grad_batches
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = int(self.trainer.limit_test_batches)

        train_valid_test_num_samples = [
            max_train_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]

        self._train_ds, self._validation_ds, self._test_ds = build_all_datasets(
            cfg=self.cfg.data, tokenizer=self.tokenizer, train_valid_test_num_samples=train_valid_test_num_samples,
        )
        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building RETRO datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        if isinstance(dataset, BlendableDataset):
            collate_fun = dataset.datasets[0].collate_fn
        else:
            collate_fun = dataset.collate_fn

        collate_fn = partial(collate_fun, tp_workers=0)
        global_batch_size = self.trainer.world_size * self.cfg.micro_batch_size // self.cfg.tensor_model_parallel_size
        batch_sampler = MegatronPretrainingBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=self.cfg.micro_batch_size,
            global_batch_size=global_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=True,
        )
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=0, pin_memory=True,
        )

    # def load_model_state_dict(self, checkpoint) -> None:
    #     self.load_state_dict(state_dict())


    # def state_dict(self, destination=None, prefix=None, keep_vars=False):
    #     """
    #     Creates a state_dict using only the adapter parameters.
    #     This ensures that this wrapper class will only checkpoint the adapter
    #     weights and not the rest of the base GPT Model.
    #     """
    #     state_dict_ = {}
    #     for name, module in self.frozen_model.named_modules():
    #         if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
    #             for adapter_key in self.adapter_name_keys:
    #                 adapter_module = module.get_adapter_module(adapter_key)
    #                 if adapter_module:
    #                     state_adapter_key = ':'.join([name, adapter_key])
    #                     state_dict_[state_adapter_key] = adapter_module.state_dict()

    #             module.set_enabled_adapters(enabled=True)
    #     return state_dict_

    # def load_state_dict(self, state_dict, strict: bool = True):
    #     """
    #     Loads a state_dict expecting the state_dict to contain key,values 
    #     only for the adapter parameters.
    #     """
    #     # state_dict = self.frozen_model.state_dict()
    #     for name, module in self.frozen_model.named_modules():
    #         if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
    #             for adapter_key in self.adapter_name_keys:
    #                 adapter_module = module.get_adapter_module(adapter_key)
    #                 if adapter_module:
    #                     state_adapter_key = ':'.join([name, adapter_key])
    #                     stact_dict_output = state_dict[state_adapter_key]
    #                     adapter_module.load_state_dict(stact_dict_output, strict)
    #             module.set_enabled_adapters(enabled=True)

    #     # self.frozen_model.load_state_dict(state_dict)
    #     print("Loaded state dict")

    # def get_forward_output_only_func(self):
    #     """
    #     Used for generate method only.
    #     """

    #     def fwd_output_only_func(batch, model):
    #         extra_arg = {}
    #         (
    #             tokens,
    #             attention_mask,
    #             retrieved,
    #             retrieved_mask,
    #             set_inference_key_value_memory,
    #             inference_max_sequence_len,
    #             neighbors,
    #             position_ids,
    #         ) = batch

    #         if len(retrieved.shape) == 1:
    #             retrieved = None
    #             retrieved_mask = None
    #         else:
    #             retrieved = retrieved.cuda()
    #             retrieved_mask = retrieved_mask.cuda()

    #         extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
    #         extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()
    #         extra_arg['neighbors'] = neighbors[0].item()
    #         # extra_arg['position_ids'] = position_ids

    #         output_tensor = model.model(tokens, attention_mask, retrieved, retrieved_mask, **extra_arg)

    #         def id_func(output_tensor):
    #             return output_tensor, {'logits': output_tensor}

    #         return output_tensor, id_func

    #     return fwd_output_only_func

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    # def setup_optimizer_param_groups(self):
    #     """
    #     ModelPT override. Optimizer will get self._optimizer_param_groups. 
    #     Makes two optimizer param groups, one for the frozen model params
    #     and one for the prompt-table/prompt-encoder params. The learning 
    #     rate for the frozen model's params will always be zero effectively
    #     freezing the model's params but still allowing for the needed gradients
    #     to be passed around in pipeline parallel models. The prompt-encoder 
    #     and/or prompt table will use the learning rate set by the user. 
    #     """
    #     self.frozen_model.freeze()  # Freeze the entire model
    #     opt_params = []
    #     for _, module in self.frozen_model.named_modules():
    #         if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
    #             module.set_enabled_adapters(enabled=True)
    #             module.unfreeze_enabled_adapters()  # selectively unfreeze the adapter modules.
    #             opt_params += [p for p in module.parameters()]

    #     self._optimizer_param_groups = [{'params': opt_params}]
    #     logging.info(f'Optimizer groups set:\n{self.frozen_model.summarize()}')

    # def training_step(self, batch, batch_idx):
    #     # we zero grads here because we also call backward in the apex fwd/bwd functions
    #     self._optimizer.zero_grad()
    #     loss_mean = self.fwd_bwd_step(batch, batch_idx, forward_only=False)
    #     self.allreduce_gradients()

    #     ## logging
    #     # we can only log on one rank if it is rank zero so we broadcast from last rank
    #     # we can avoid this broadcast by updating the PTL log function to accept specific ranks
    #     torch.distributed.broadcast(loss_mean, get_last_rank())

    #     if self.cfg.precision == 16:
    #         loss_scale = self.trainer.precision_plugin.scaler._scale
    #         if loss_scale is not None:
    #             self.log('loss_scale', loss_scale)

    #     self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True)
    #     lr = self._optimizer.param_groups[0]['lr']
    #     self.log('lr', lr, rank_zero_only=True)
    #     self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True)

    #     # Need to make sure the frozen model param learning rate stays 0.0
    #     # so forceing lr to be 0.0 for gpt layers before param update
    #     return loss_mean

    # def training_step(self, batch, batch_idx):
    #     # we zero grads here because we also call backward in the apex fwd/bwd functions
    #     self._optimizer.zero_grad()
    #     loss_mean = self.fwd_bwd_step(batch, batch_idx, forward_only=False)
    #     self.allreduce_gradients()

    #     ## logging
    #     # we can only log on one rank if it is rank zero so we broadcast from last rank
    #     # we can avoid this broadcast by updating the PTL log function to accept specific ranks
    #     torch.distributed.broadcast(loss_mean, get_last_rank())

    #     if self.cfg.precision == 16:
    #         loss_scale = self.trainer.precision_plugin.scaler._scale
    #         if loss_scale is not None:
    #             self.log('loss_scale', loss_scale)

    #     self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True)
    #     lr = self._optimizer.param_groups[0]['lr']
    #     self.log('lr', lr, rank_zero_only=True)
    #     self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True)

    #     # Need to make sure the frozen model param learning rate stays 0.0
    #     # so forceing lr to be 0.0 for gpt layers before param update
    #     return loss_mean


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
