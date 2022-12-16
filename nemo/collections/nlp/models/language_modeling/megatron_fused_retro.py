import logging
import re
from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_dataset import GPTPromptLearningDataset
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
from nemo.collections.nlp.modules.common import VirtualPromptPlaceholderToken, VirtualPromptSource, VirtualPromptStyle
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core import adapter_mixins

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


class MegatronFusedRetrievalAdapterModel(MegatronRetrievalModel, adapter_mixins.AdapterModelPTMixin):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        if adapter_mixins.get_registered_adapter(MegatronRetrievalModel) is None:
            adapter_mixins.register_adapter(MegatronRetrievalModel, MegatronFusedRetrievalAdapterModel)

        self.existing_tasks = list(self.cfg.get('existing_tasks', []))
        self.new_tasks = list(self.cfg.get('new_tasks', []))
        self.virtual_prompt_style = VirtualPromptStyle(cfg.virtual_prompt_style)
        save_restore_connector = NLPSaveRestoreConnector()
        frozen_model_cfg = MegatronGPTModel.restore_from(
            cfg.get('language_model_path'),
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        if cfg.get('language_model_path', None):
            self.frozen_model = MegatronGPTModel.restore_from(
                cfg.get('language_model_path'),
                trainer=trainer,
                save_restore_connector=save_restore_connector,
                override_config_path=frozen_model_cfg,
            ).to(dtype=self.autocast_dtype)

        self.layers = nn.ModuleList()

        # Prepare pseudo token ids for virtual/virtual prompt tokens
        self.load_task_templates(self.cfg.task_templates)
        self.pseudo_tokens = get_pseudo_tokens(self.max_virtual_tokens)
        self.tokenizer.add_special_tokens({'additional_special_tokens': self.pseudo_tokens})
        self.pseudo_token_ids = self.tokenizer.tokens_to_ids(self.pseudo_tokens)
        self.pseudo_token_ids_start = self.pseudo_token_ids[0] if self.pseudo_token_ids else None
        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.unk_id

        # Prompt tuning stores virtual prompts in the prompt table and tunes their weight directly
        if self.virtual_prompt_style in [VirtualPromptStyle.PROMPT_TUNING, VirtualPromptStyle.INFERENCE]:
            self.virtual_prompt_source = VirtualPromptSource.PROMPT_TABLE

        # P-Tuning uses an LSTM Encoder to produce virtual token embeddings
        elif self.virtual_prompt_style == VirtualPromptStyle.P_TUNING:
            self.virtual_prompt_source = VirtualPromptSource.PROMPT_ENCODER
        elif self.virtual_prompt_style == VirtualPromptStyle.NO_PROMPT:
            self.virtual_prompt_source = VirtualPromptSource.NO_PROMPT
        else:
            raise ValueError(
                f"\nvirtual prompt style '{cfg.virtual_prompt_style}' not recognized, please use one of 'prompt-tuning' or 'p-tuning'"
            )

    def forward(
        self,
        input_ids,
        input_attn_mask,
        position_ids,
        retrieved_ids,
        retrieved_attn_mask=None,
        token_type_ids=None,
        labels=None,
        input_emb=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        neighbors=None,
        encoder_input=None,
    ):

        self.forward_enabled_adapters(input_ids)

    def setup(self, stage=None):
        if stage == 'predict' or self.virtual_prompt_style == VirtualPromptStyle.INFERENCE:
            self.frozen_model.freeze()
            return

        self.setup_test_data()
        if stage == 'test':
            return

        self.setup_training_data()
        self.setup_validation_data()
        logging.info(f'setup completed:\n{self.frozen_model.summarize()}')

    def setup_training_data(self, training_data_config=None):
        if self.cfg.data.get('train_ds', None):
            self._train_ds, self._train_dl = self.build_virtual_prompt_dataset(
                data=self.cfg.data.train_ds,
                batch_size=self.cfg.global_batch_size,
                max_seq_length=self.frozen_model.cfg.encoder_seq_length,
                min_seq_length=self.cfg.data.get('min_seq_length', 1),
                add_bos=self.cfg.data.get('add_bos', False),
                add_eos=self.cfg.data.get('add_eos', True),
                for_train=True,
                drop_last=True,
                shuffle=True,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
                cache_data_path=self.cfg.data.get('train_cache_data_path', None),
                load_cache=self.cfg.data.get('load_cache', False),
            )

    def setup_validation_data(self, validation_data_config=None):
        if self.cfg.data.get('validation_ds', None):
            self._validation_ds, self._validation_dl = self.build_virtual_prompt_dataset(
                data=self.cfg.data.validation_ds,
                batch_size=self.cfg.global_batch_size,
                max_seq_length=self.frozen_model.cfg.encoder_seq_length,
                min_seq_length=self.cfg.data.get('min_seq_length', 1),
                add_bos=self.cfg.data.get('add_bos', False),
                add_eos=self.cfg.data.get('add_eos', True),
                for_train=True,
                drop_last=True,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
                cache_data_path=self.cfg.data.get('validation_cache_data_path', None),
                load_cache=self.cfg.data.get('load_cache', False),
            )

    def setup_test_data(self, test_data_config=None):
        if self.cfg.data.get('test_ds', None):
            self._test_ds, self._test_dl = self.build_virtual_prompt_dataset(
                data=self.cfg.data.test_ds,
                batch_size=self.cfg.global_batch_size,
                max_seq_length=self.frozen_model.cfg.encoder_seq_length,
                min_seq_length=self.cfg.data.get('min_seq_length', 1),
                add_bos=self.cfg.data.get('add_bos', False),
                add_eos=self.cfg.data.get('add_eos', True),
                for_train=False,
                drop_last=False,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
                cache_data_path=self.cfg.data.get('test_cache_data_path', None),
                load_cache=self.cfg.data.get('load_cache', False),
            )

    def build_virtual_prompt_dataset(
        self,
        data,
        batch_size=None,
        max_seq_length=2048,
        min_seq_length=1,
        add_bos=False,
        add_eos=False,
        for_train=True,
        drop_last=False,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        tokens_to_generate=None,
        get_dataset_only=False,
        cache_data_path=None,
        load_cache=False,
    ):
        dataset = GPTPromptLearningDataset(
            data=data,
            tokenizer=self.tokenizer,
            virtual_prompt_source=self.virtual_prompt_source,
            task_templates=self.task_templates,
            pseudo_tokens=self.pseudo_tokens,
            pad_token_id=self.pad_token_id,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            add_bos=add_bos,
            add_eos=add_eos,
            for_train=for_train,
            tokens_to_generate=tokens_to_generate,
            cache_data_path=cache_data_path,
            load_cache=load_cache,
        )

        if get_dataset_only:
            return dataset

        # Make distributed dataloader
        rank = parallel_state.get_data_parallel_rank()
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=data_parallel_size, rank=rank, shuffle=shuffle
        )

        assert batch_size % data_parallel_size == 0, "Global batch size must be evenly divisible by data parallel size"

        if for_train:
            if self.cfg.get("sequence_parallel", False):
                collate_fn = partial(
                    dataset.collate_fn, tp_workers=parallel_state.get_tensor_model_parallel_world_size()
                )
            else:
                collate_fn = partial(dataset.collate_fn, tp_workers=0)
        else:
            collate_fn = dataset.inference_collate_fn

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            sampler=sampler,
            batch_size=batch_size // data_parallel_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return dataset, dataloader

    def freeze_existing_virtual_prompt_params(self):
        """Freeze params of existing virtual prompts that should not be tuned further
            """
        # Only want new prompt tags to be tunable, leave existing prompt tags alone
        for taskname in self.prompt_table.prompt_table.keys():
            if taskname in set(self.new_tasks):
                for params in self.prompt_table.prompt_table[taskname].parameters():
                    params.requires_grad = True
            else:
                for params in self.prompt_table.prompt_table[taskname].parameters():
                    params.requires_grad = False

        # Make sure word embeddings are frozen
        for params in self.word_embeddings.parameters():
            params.requires_grad = False

    def add_adapter(self, name: str, cfg: DictConfig):
        # call the same method on each `MLP` layer, collecting results
        for layer in self.layers:
            layer.add_adapter(name, cfg)

    def get_enabled_adapters(self) -> List[str]:
        # call the same method on each `MLP` layer, collecting results
        enabled_adapters = set([])
        for layer in self.layers:
            names = layer.get_enabled_adapters()
            enabled_adapters.update(names)
        return list(enabled_adapters)

    def set_enabled_adapters(self, name: Optional[str], enabled: bool):
        # call the same method on each `MLP` layer, collecting results
        for layer in self.layers:
            layer.set_enabled_adapters(name, enabled)

    def is_adapter_available(self) -> bool:
        # call the same method on each `MLP` layer, collecting results
        is_available = any([layer.is_adapter_available() for layer in self.layers])
        return is_available

    # First we call forward on Adapter learning

    # Take outputs from forward adapter learning function

    # Feed inputs with retrieved_ids to retro model

    def load_task_templates(self, task_templates):
        """
        Takes in the task template portion of the config and turns  
        it into a table where each task's prompt template and 
        the number of virtual tokens to insert in a given part of 
        the prompt template are specified. 
        """
        self.task_templates = {}
        self.task_id_num_to_name = {}
        self.max_virtual_tokens = 0

        task_id_num = 0
        for task in task_templates:
            self.task_templates[task.taskname] = {
                "prompt_template": task.prompt_template,
                "prompt_template_fields": re.findall("\{(.*?)\}", task.prompt_template),
                "answer_only_loss": task.get("answer_only_loss", False),
                "answer_field": task.get("answer_field", None),
                "truncate_field": task.truncate_field,
                "total_virtual_tokens": task.total_virtual_tokens,
                "virtual_token_splits": task.virtual_token_splits,
                "task_id_num": task_id_num,
            }

            self.max_virtual_tokens = max(self.max_virtual_tokens, task.total_virtual_tokens)
            self.task_id_num_to_name[task_id_num] = task.taskname
            task_id_num += 1

        # Check that all new tasks have the same total num virtual tokens
        # Num virtual tokens for new tasks don't need to match num used for previously tuned tasks
        if self.new_tasks:
            new_task_name = self.new_tasks[0]
            self.total_new_task_virtual_tokens = self.task_templates[new_task_name]["total_virtual_tokens"]

            assert all(
                self.task_templates[taskname]["total_virtual_tokens"] == self.total_new_task_virtual_tokens
                for taskname in self.new_tasks
            ), "Total virtual tokens for each task tuned simultaneously must match. If you want to use a different number of virtual tokens for different tasks, tune them separately."


def get_pseudo_tokens(num_virtual_tokens):
    """
    Takes in an integer and returns a list of strings where each string
    is a numbered virtual token placeholder. If 
    num_virtual_tokens = 3, then this function returns:

    ["<prompt_0>", "<prompt_1>", "<prompt_2>"]

    Args:
        num_virtual_tokens: (int) Number of virtual token strings you want to make

    returns a list of string. 

    """
    pseudo_tokens = [
        VirtualPromptPlaceholderToken.BASE.value + str(i) + VirtualPromptPlaceholderToken.END.value
        for i in range(num_virtual_tokens)
    ]

    return pseudo_tokens
