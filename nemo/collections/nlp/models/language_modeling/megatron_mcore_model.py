# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from dataclasses import fields
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_lm_encoder_decoder_model import MegatronLMEncoderDecoderModel
from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.parts.utils_funcs import activation_to_func
from nemo.utils import logging

try:
    from megatron.core import parallel_state
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import init_method_normal, scaled_init_method_normal
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, get_gpt_layer_local_spec
    from megatron.core.models.T5 import T5Model
    from megatron.core.models.T5.t5_spec import encoder_model_with_transformer_engine_default_spec, decoder_model_with_transformer_engine_default_spec
    from megatron.core.models.bert.bert_model import BertModel
    from megatron.core.models.bert.bert_layer_specs import bert_layer_with_transformer_engine_spec, bert_layer_local_spec
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.core.datasets.bert_dataset import BERTMaskedWordPieceDataset, BERTMaskedWordPieceDatasetConfig
    from megatron.core.datasets.t5_dataset import T5MaskedWordPieceDataset, T5MaskedWordPieceDatasetConfig
    
    HAVE_MEGATRON_CORE = True
    
except (ImportError, ModuleNotFoundError):
    
    HAVE_MEGATRON_CORE = False


class MegatronMcoreModel(MegatronGPTModel, MegatronLMEncoderDecoderModel, MegatronBertModel):
    """
    Megatron models pretraining.
    """
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        
        MegatronBaseModel.__init__(self, cfg, trainer)
        self.model_parent = cfg.get('model_parent')
        self.model_name = cfg.get('model_name')
        self.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)
        self.architecture_config = self.build_architecture_config()
        self.model_config = self.build_model_config()
        
        match self.model_parent:
            case 'gpt':
                MegatronGPTModel.__init__(self, cfg, trainer)
                self.mcore_gpt = True
            case 't5':
                MegatronLMEncoderDecoderModel.__init__(self, cfg, trainer)
            case 'bert':
                MegatronBertModel.__init__(self, cfg, trainer)
                self.mcore_bert = True
            case _:
                raise NotImplementedError(f'{self.model_parent} trainer is not supported')

    def setup(self, stage=None):
        match self.model_parent:
            case 'gpt':
                MegatronGPTModel.setup(self, stage)
            case 't5':
                MegatronLMEncoderDecoderModel.setup(self, stage)
            case 'bert':
                MegatronBertModel.setup(self, stage)
                
    def list_available_models(self):
        return None
        
    def model_provider_func(self, pre_process, post_process, add_encoder=False, add_decoder=False):
        match self.model_name:
            case 'gpt':
                model = GPTModel(
                    config=self.architecture_config,
                    pre_process=pre_process, 
                    post_process=post_process,
                    **self.model_config,
                )
            case 't5':
                model = T5Model(
                    config=self.architecture_config,
                    pre_process=pre_process, 
                    post_process=post_process,
                    **self.model_config,
                )
            case 'bert':
                model = BertModel(
                    config=self.architecture_config,
                    pre_process=pre_process, 
                    post_process=post_process,
                    **self.model_config,
                )
            case _:
                raise NotImplementedError(f'{self.model_name} model architecture is not supported')
        return model

    def build_architecture_config(self):
        # create a dictionary copy of the model config
        cfg_cli = OmegaConf.to_container(self.cfg, resolve=True)
        cfg_cli_convert = {
            # general config
            'fp16': self.torch_dtype == torch.float16 and self.megatron_amp_O2,
            'bf16': self.torch_dtype == torch.bfloat16 and self.megatron_amp_O2,
            'params_dtype': self.torch_dtype if self.torch_dtype in [torch.bfloat16, torch.float16] and self.megatron_amp_O2 else torch.float32,
            'timers': self.megatron_timers,
            'async_tensor_model_parallel_allreduce': cfg_cli.get('tensor_model_parallel_world_size', 1) > 1 and not cfg_cli.get('sequence_parallel', False),
            'pipeline_dtype': self.torch_dtype,
            'grad_scale_func': self.trainer.precision_plugin.scaler.scale if self.trainer.precision in ["16", "16-mixed"] else None,
            'enable_autocast': self.torch_dtype in [torch.bfloat16, torch.float16] and not self.megatron_amp_O2,
            'autocast_dtype': self.torch_dtype,
            # transformer config
            'activation_func': activation_to_func(cfg_cli.get('activation', 'gelu')),
            'gated_linear_unit': cfg_cli.get('activation', 'gelu').endswith('glu'),
            'init_method': init_method_normal(cfg_cli.get('init_method_std', 0.02)),
            'output_layer_init_method': scaled_init_method_normal(cfg_cli.get('init_method_std', 0.02), num_layers=cfg_cli.get('num_layers', 1)) \
            if cfg_cli.get('use_scaled_init_method', True) else init_method_normal(cfg_cli.get('init_method_std', 0.02)),
            'apply_query_key_layer_scaling': cfg_cli.get('apply_query_key_layer_scaling', False) and self.trainer.precision in [16, '16', '16-mixed'],
            'attention_softmax_in_fp32': cfg_cli.get('attention_softmax_in_fp32', True)or cfg_cli.get('apply_query_key_layer_scaling', False) ,
        }
        
        # create a dict to store the architecture config arguments
        architecture_config_dict = {}
        for field in fields(TransformerConfig):
            if field.name in cfg_cli:
                architecture_config_dict[field.name] = cfg_cli[field.name]
            elif field.name in cfg_cli_convert:
                architecture_config_dict[field.name] = cfg_cli_convert[field.name]
            else:
                logging.warning(
                    f"The model: {self} does not have the argument: {field.name} in its cfg. "
                    f"Add this key to cfg to make to make it configurable."
                )

        architecture_config = TransformerConfig(**architecture_config_dict)
        return architecture_config

    def build_model_config(self):
        # create a dictionary copy of the model config
        cfg = OmegaConf.to_container(self.cfg, resolve=True)
        match self.model_name:
            case 'gpt':
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec if cfg.get('transformer_engine') else get_gpt_layer_local_spec
                model_config = {
                    'transformer_layer_spec': transformer_layer_spec(
                        self.architecture_config.num_moe_experts,
                        self.architecture_config.moe_grouped_gemm,
                        self.architecture_config.qk_layernorm,
                    ),
                    'vocab_size': self.padded_vocab_size,
                    'max_sequence_length': 512,
                    'share_embeddings_and_output_weights': True,
                    'position_embedding_type': 'learned_absolute',
                    'rotary_percent': 1.0,
                    'seq_len_interpolation_factor': None,
                    'fp16_lm_cross_entropy': False,
                    'parallel_output': True,
                    'rotary_base': 10000,
                }
            case 't5':
                model_config = {
                    'transformer_encoder_layer_spec': encoder_model_with_transformer_engine_default_spec(),
                    'transformer_decoder_layer_spec': decoder_model_with_transformer_engine_default_spec(),
                    'vocab_size': self.padded_vocab_size,
                    'max_sequence_length': 512,
                    'share_embeddings_and_output_weights': True,
                    'position_embedding_type': 'learned_absolute',
                    'rotary_percent': 1.0,
                    'seq_len_interpolation_factor': None,
                    'fp16_lm_cross_entropy': False,
                    'parallel_output': True,
                }
            case 'bert':
                transformer_layer_spec = bert_layer_with_transformer_engine_spec if cfg.get('transformer_engine') else bert_layer_local_spec
                model_config = {
                    'transformer_layer_spec': transformer_layer_spec,
                    'vocab_size': self.padded_vocab_size,
                    'max_sequence_length': 512,
                    'share_embeddings_and_output_weights': True,
                    'position_embedding_type': 'learned_absolute',
                    'rotary_percent': 1.0,
                    'seq_len_interpolation_factor': None,
                    'fp16_lm_cross_entropy': False,
                    'parallel_output': True,
                    'add_binary_head': True,
                    'num_tokentypes': 0 if cfg.get('add_binary_head') == False else 2,
                }
        
        for key in model_config:
            if key in cfg:
                model_config[key] = cfg[key]
            else:
                logging.warning(
                    f"The model: {self} does not have the argument: {key} in its cfg. "
                    f"Add this key to cfg to make to make it configurable."
                )
        return model_config
        
    def build_transformer_config(self):
        return 
        
    def build_model_parallel_config(self):
        return

    def setup_optimizer_param_groups(self):
        match self.model_parent:
            case 'gpt':
                MegatronGPTModel.setup_optimizer_param_groups(self)
            case 't5':
                MegatronLMEncoderDecoderModel.setup_optimizer_param_groups(self)
            case 'bert':
                MegatronBertModel.setup_optimizer_param_groups(self)

    def configure_optimizers(self):
        match self.model_parent:
            case 'gpt':
                return MegatronGPTModel.configure_optimizers(self)
            case 't5':
                return MegatronLMEncoderDecoderModel.configure_optimizers(self)
            case 'bert':
                return MegatronBertModel.configure_optimizers(self)
    
    def forward(self):
        raise NotImplementedError(f'Foward function for trainer is not supported.')

    def training_step(self, dataloader_iter):
        match self.model_parent:
            case 'gpt':
                return MegatronGPTModel.training_step(self, dataloader_iter)
            case 't5':
                return MegatronLMEncoderDecoderModel.training_step(self, dataloader_iter)
            case 'bert':
                return MegatronBertModel.training_step(self, dataloader_iter)

    def validation_step(self, dataloader_iter, dataloader_idx=0):
        match self.model_parent:
            case 'gpt':
                return MegatronGPTModel.validation_step(self, dataloader_iter, dataloader_idx)
            case 't5':
                return MegatronLMEncoderDecoderModel.validation_step(self, dataloader_iter)
            case 'bert':
                return MegatronBertModel.validation_step(self, dataloader_iter)
                
    def test_step(self, dataloader_iter):
        match self.model_parent:
            case 'gpt':
                return MegatronGPTModel.test_step(self, dataloader_iter)
            case 't5':
                return MegatronLMEncoderDecoderModel.test_step(self, dataloader_iter)
            case 'bert':
                return MegatronBertModel.test_step(self, dataloader_iter)
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        match self.model_parent:
            case 'gpt':
                return MegatronGPTModel.predict_step(self, batch, batch_idx, dataloader_idx)
            case 't5':
                return MegatronLMEncoderDecoderModel.predict_step(self, batch, batch_idx, dataloader_idx)
            case 'bert':
                raise NotImplementedError(f'{self.model_parent} does not support this function.')
                
    def on_validation_epoch_end(self):
        match self.model_parent:
            case 'gpt':
                return MegatronGPTModel.on_validation_epoch_end(self)
            case 't5':
                return MegatronLMEncoderDecoderModel.on_validation_epoch_end(self)
            case 'bert':
                return MegatronBertModel.on_validation_epoch_end(self)
    
    def on_validation_model_zero_grad(self):
        match self.model_parent:
            case 'gpt':
                MegatronGPTModel.on_validation_model_zero_grad(self)
            case _:
                MegatronBaseModel.on_validation_model_zero_grad(self) 
                
    def on_test_epoch_end(self):
        match self.model_parent:
            case 'gpt':
                return MegatronGPTModel.on_test_epoch_end(self)
            case 't5':
                return MegatronLMEncoderDecoderModel.on_test_epoch_end(self)
            case 'bert':
                return MegatronBertModel.on_test_epoch_end(self)

    def loss_func(self, *args):
        match self.model_parent:
            case 'gpt':
                return MegatronGPTModel.loss_func(self, *args)
            case 't5':
                return MegatronLMEncoderDecoderModel.loss_func(self, *args)
            case 'bert':
                return MegatronBertModel.loss_func(self, *args)

    def fwd_bwd_step(self, dataloader_iter, forward_only, first_val_step=None):
        match self.model_parent:
            case 'gpt':
                return MegatronGPTModel.fwd_bwd_step(self, dataloader_iter, forward_only, first_val_step)
            case 't5':
                return MegatronLMEncoderDecoderModel.fwd_bwd_step(self, dataloader_iter, forward_only)
            case _:
                raise NotImplementedError(f'{self.model_parent} does not support this function.')

    def get_forward_output_and_loss_func(self, validation_step=False, tuning=False):
        match self.model_parent:
            case 'gpt':
                return MegatronGPTModel.get_forward_output_and_loss_func(self, validation_step, tuning)
            case 't5':
                return MegatronLMEncoderDecoderModel.get_forward_output_and_loss_func(self)
            case 'bert':
                return MegatronBertModel.get_forward_output_and_loss_func(self)

    def get_forward_output_only_func(self):
        match self.model_parent:
            case 'gpt': # for generation
                return MegatronGPTModel.get_forward_output_only_func(self)
            case _:
                raise NotImplementedError(f'{self.model_parent} does not support this function.')
        
    def _make_data_iterator_list(self, data_iterator: Iterator) -> List[Iterator]:
        match self.model_parent:
            case 'gpt':
                return MegatronGPTModel._make_data_iterator_list(self, data_iterator)
            case 'bert':
                return MegatronBertModel._make_data_iterator_list(self, data_iterator)
            case _:
                raise NotImplementedError(f'{self.model_parent} does not support this function.')

    def on_save_checkpoint(self, checkpoint) -> None:
        match self.model_parent:
            case 'gpt':
                MegatronGPTModel.on_save_checkpoint(self, checkpoint)
            case 'bert':
                MegatronBertModel.on_save_checkpoint(self, checkpoint)
            case _:
                raise NotImplementedError(f'{self.model_parent} does not support this function.')
                
    def on_load_checkpoint(self, checkpoint) -> None:
        match self.model_parent:
            case 'gpt':
                MegatronGPTModel.on_load_checkpoint(self, checkpoint)
            case 'bert':
                MegatronBertModel.on_load_checkpoint(self, checkpoint)
            case _:
                raise NotImplementedError(f'{self.model_parent} does not support this function.')
    
    def sharded_state_dict(self, prefix: str = '') -> Dict[str, Any]:
        match self.model_parent:
            case 'gpt':
                return MegatronGPTModel.sharded_state_dict(self, prefix)
            case 'bert':
                return MegatronBertModel.sharded_state_dict(self, prefix)
            case _:
                raise NotImplementedError(f'{self.model_parent} does not support this function.')
                
    def parameters(self):
        match self.model_parent:
            case 'gpt':
                return MegatronGPTModel.parameters(self)
            case 'bert':
                return MegatronBertModel.parameters(self)
            case _:
                return self.parameters()
    
    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds') and self._train_ds:
            self._train_dl = self.build_pretraining_data_loader(
                type='train', dataset=self._train_ds, consumed_samples=self.compute_consumed_samples(0)
            )
            
    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds') and self._validation_ds:
            self._validation_dl = self.build_pretraining_data_loader(
                type='validation', dataset=self._validation_ds, consumed_samples=0,
            )

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds') and self._test_ds:
            self._test_dl = self.build_pretraining_data_loader(
                type='test', dataset=self._test_ds, consumed_samples=0,
            )

    def build_train_valid_test_datasets(self):
        logging.info(f'Building {self.model_parent} datasets.')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")
        global_batch_size = self._cfg.global_batch_size
        eval_iters = (self.trainer.max_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches
        train_valid_test_num_samples = [
            self.trainer.max_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]

        if self.trainer.limit_val_batches <= 1.0 and isinstance(self.trainer.limit_val_batches, float):
            # This is to make sure we only have one epoch on every validation iteration
            train_valid_test_num_samples[1] = 1

        default_config = {
            "random_seed": self.cfg.seed,
            "blend_per_split": [self.cfg.data.data_prefix.get('train'), self.cfg.data.data_prefix.get('validation')] \
                                if isinstance(self.cfg.data.data_prefix, DictConfig) else None,
            "blend": self.cfg.data.get('data_prefix'),
            "split": self.cfg.data.get('splits_string'),
            "sequence_length": self.cfg.data.seq_length,
            "path_to_cache": self.cfg.data.index_mapping_dir,
            "mmap_bin_files": self.cfg.data.get("mmap_bin_files", True),
            "tokenizer": self.tokenizer,
        }
        
        match self.model_parent:
            case 'gpt':
                dataset_type = GPTDataset
                dataset_config = GPTDatasetConfig(
                    reset_position_ids=self.cfg.data.get('gpt_reset_position_ids', False),
                    reset_attention_mask=self.cfg.data.get('gpt_reset_attention_mask', False),
                    eod_mask_loss=self.cfg.data.get('gpt_eod_mask_loss', False),
                    create_attention_mask=self.cfg.data.get('gpt_create_attention_mask', False),
                    **default_config
                )
            case 't5':
                dataset_type = T5MaskedWordPieceDataset
                dataset_config = T5MaskedWordPieceDatasetConfig(
                    masking_probability=self.cfg.data.get('t5_masked_lm_prob', 0.15),
                    short_sequence_probability=self.cfg.data.get('t5_short_seq_prob', 0.0),
                    masking_max_ngram=self.cfg.data.get('t5_max_ngram_size', 10),
                    masking_do_full_word=self.cfg.data.get('t5_whole_word_masking', True),
                    masking_do_permutation=self.cfg.data.get('t5_permutation', False),
                    masking_use_longer_ngrams=self.cfg.data.get('t5_favor_longer_ngrams', False),
                    masking_use_geometric_distribution=self.cfg.data.get('t5_geometric_dist', True),
                    **default_config
                )
            case 'bert':
                dataset_type = BERTMaskedWordPieceDataset
                dataset_config = BERTMaskedWordPieceDatasetConfig(
                    masking_probability=self.cfg.data.get('bert_masked_lm_prob', 0.15),
                    short_sequence_probability=self.cfg.data.get('bert_short_seq_prob', 0.1),
                    masking_max_ngram=self.cfg.data.get('bert_max_ngram_size', 10),
                    masking_do_full_word=self.cfg.data.get('bert_whole_word_masking', True),
                    masking_do_permutation=self.cfg.data.get('bert_permutation', False),
                    masking_use_longer_ngrams=self.cfg.data.get('bert_favor_longer_ngrams', False),
                    masking_use_geometric_distribution=self.cfg.data.get('bert_geometric_dist', True),
                    classification_head=self.cfg.data.get('bert_classification_head', False),
                    **default_config
                )
                
        self._train_ds, self._validation_ds, self._test_ds = BlendedMegatronDatasetBuilder(
            dataset_type, 
            sizes=train_valid_test_num_samples, 
            is_built_on_rank=lambda: True, 
            config=dataset_config,
        ).build()
        
        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
            
        logging.info(f'Finished building {self.model_parent} datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(self, type, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""
        logging.info(f'Setting up {type} dataloader with length: {len(dataset)} and consumed samples: {consumed_samples}')
        
        if self.cfg.data.get('dataloader_type') == 'single':
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=self.cfg.micro_batch_size,
                global_batch_size=self.cfg.global_batch_size,
                rampup_batch_size=self.cfg.data.get('rampup_batch_size'),
                data_parallel_rank=parallel_state.get_data_parallel_rank(),
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
                pad_samples_to_global_batch_size=False,
                drop_last=self.cfg.data.get('drop_last', True),
            )
        elif self.cfg.data.get('dataloader_type') == 'cyclic':
            batch_sampler = MegatronPretrainingRandomSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=self.cfg.micro_batch_size,
                global_batch_size=None,
                data_parallel_rank=parallel_state.get_data_parallel_rank(),
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
                pad_samples_to_global_batch_size=False,
                drop_last=self.cfg.data.get('drop_last', True),
            )
        else:
            raise NotImplementedError('cfg.data.dataloader_type must be "single" or "cyclic"')

        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.data.num_workers > 0 else False,
        )