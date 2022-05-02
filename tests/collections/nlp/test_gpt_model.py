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

import pytest
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin


@pytest.fixture()
def model_cfg():

    model_cfg = {
        'micro_batch_size': 4,
        'global_batch_size': 8,
        'tensor_model_parallel_size': 1,
        'pipeline_model_parallel_size': 1,
        'resume_from_checkpoint': None,
        'encoder_seq_length': 512,
        'max_position_embeddings': 512,
        'num_layers': 1,
        'hidden_size': 128,
        'ffn_hidden_size': 128 * 4,
        'num_attention_heads': 2,
        'init_method_std': 0.02,
        'hidden_dropout': 0.1,
        'kv_channels': None,
        'apply_query_key_layer_scaling': True,
        'layernorm_epsilon': '1e-5',
        'make_vocab_size_divisible_by': 128,
        'pre_process': True,
        'post_process': True,
        'persist_layer_norm': True,
        'gradient_as_bucket_view': True,
        'tokenizer': {
            'library': 'megatron',
            'type': 'GPT2BPETokenizer',
            'model': None,
            'vocab_file': None,
            'merge_file': None,
            'delimiter': None,
        },
        'native_amp_init_scale': 4294967296,
        'native_amp_growth_interval': 1000,
        'hysteresis': 2,
        'fp32_residual_connection': False,
        'fp16_lm_cross_entropy': False,
        'megatron_amp_O2': False,
        'seed': 1234,
        'use_cpu_initialization': False,
        'onnx_safe': False,
        'apex_transformer_log_level': 30,
        'activations_checkpoint_method': None,
        'activations_checkpoint_num_layers': 1,
        'data': {
            'data_prefix': '???',
            'index_mapping_dir': None,
            'data_impl': 'mmap',
            'splits_string': '900,50,50',
            'seq_length': 512,
            'skip_warmup': True,
            'num_workers': 2,
            'dataloader_type': 'single',
            'reset_position_ids': False,
            'reset_attention_mask': False,
            'eod_mask_loss': False,
        },
        'optim': {
            'name': 'fused_adam',
            'lr': '2e-4',
            'weight_decay': 0.01,
            'betas': [0.9, 0.98],
            'sched': {'name': 'CosineAnnealing', 'warmup_steps': 500, 'constant_steps': 50000, 'min_lr': '2e-5'},
        },
    }
    return model_cfg


@pytest.fixture()
def trainer_cfg():

    trainer_cfg = {
        'devices': 1,
        'num_nodes': 1,
        'accelerator': 'gpu',
        'precision': 16,
        'logger': False,
        'enable_checkpointing': False,
        'replace_sampler_ddp': False,
        'max_epochs': 1000,
        'max_steps': 100000,
        'log_every_n_steps': 10,
        'val_check_interval': 100,
        'limit_val_batches': 50,
        'limit_test_batches': 500,
        'accumulate_grad_batches': 1,
        'gradient_clip_val': 1.0,
    }

    return trainer_cfg


@pytest.fixture()
def gpt_model(model_cfg, trainer_cfg):

    plugins = [NLPDDPPlugin()]

    trainer = Trainer(plugins=plugins, **trainer_cfg)

    cfg = DictConfig(model_cfg)

    model = MegatronGPTModel(cfg=cfg, trainer=trainer)

    return model


class TestGPTModel:
    @pytest.mark.unit
    def test_tokenizer(self, model_cfg):

        tokenizer_cfg = model_cfg['tokenizer']
        # TODO: add vocab file to test tar
        tokenizer_cfg['vocab_file'] = '/raid/data/gpt_vocab_merges/vocab.json'
        tokenizer_cfg['merge_file'] = '/raid/data/gpt_vocab_merges/merges.txt'
        tokenizer = get_nmt_tokenizer(
            library=tokenizer_cfg['library'],
            model_name=tokenizer_cfg['type'],
            vocab_file=tokenizer_cfg['vocab_file'],
            merges_file=tokenizer_cfg['merge_file'],
        )
        assert isinstance(tokenizer, AutoTokenizer)
        assert tokenizer.name == 'GPT2Tokenizer'
        assert tokenizer.vocab_size == 50257

    @pytest.mark.unit
    def test_constructor(self, gpt_model):
        assert isinstance(gpt_model, MegatronGPTModel)

        num_weights = gpt_model.num_weights
        assert num_weights == 6702976
