import os
import time

import pytest
import torch
from megatron.core import parallel_state
from omegaconf import DictConfig, open_dict
from pytorch_lightning import seed_everything

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder


@pytest.mark.run_only_on('GPU')
class TestGPTModelParallel:
    DEFAULT_SEED = 1234
    EXPECTED_LOSS = torch.tensor(0.5541142225265503)

    @pytest.mark.unit
    def test_tensor_parallel(self, gpt_cfg, trainer_cfg):
        cfg = DictConfig({"model": gpt_cfg, "trainer": trainer_cfg,})
        with open_dict(cfg):
            cfg.model.tensor_model_parallel_size = 2
            cfg.trainer.devices = 2

        with TrainingBenchmark(self.DEFAULT_SEED):
            trainer = MegatronTrainerBuilder(cfg).create_trainer()
            gpt = MegatronGPTModel(cfg.model, trainer)

            trainer.fit(gpt)

        torch.testing.assert_close(gpt._loss, self.EXPECTED_LOSS, check_device=False)

    @pytest.mark.unit
    def test_sequence_parallel(self, gpt_cfg, trainer_cfg):
        cfg = DictConfig({"model": gpt_cfg, "trainer": trainer_cfg,})
        with open_dict(cfg):
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.sequence_parallel = True
            cfg.trainer.devices = 2

        with TrainingBenchmark(self.DEFAULT_SEED):
            trainer = MegatronTrainerBuilder(cfg).create_trainer()
            gpt = MegatronGPTModel(cfg.model, trainer)

            trainer.fit(gpt)

        torch.testing.assert_close(
            gpt._loss, self.EXPECTED_LOSS, check_device=False, atol=0.01, rtol=0.01,
        )

    def teardown_torch_dist_group(self):
        torch.distributed.destroy_process_group()
        # sleep for a bit to avoid race conditions with new process group.
        time.sleep(3)


class TrainingBenchmark:
    def __init__(self, seed):
        seed_everything(seed)

    def __enter__(self):
        torch.backends.cudnn.deterministic = True

    def __exit__(self, *args):
        torch.distributed.barrier()
        parallel_state.destroy_model_parallel()


@pytest.fixture()
def gpt_cfg(test_data_dir):

    model_cfg = {
        'precision': 32,
        'micro_batch_size': 4,
        'global_batch_size': 8,
        'tensor_model_parallel_size': 1,
        'pipeline_model_parallel_size': 1,
        'resume_from_checkpoint': None,
        'encoder_seq_length': 512,
        'max_position_embeddings': 512,
        'num_layers': 1,
        'hidden_size': 128,
        'ffn_hidden_size': 512,
        'num_attention_heads': 2,
        'init_method_std': 0.02,
        'hidden_dropout': 0.1,
        'kv_channels': None,
        'apply_query_key_layer_scaling': True,
        'layernorm_epsilon': 1e-5,
        'make_vocab_size_divisible_by': 128,
        'pre_process': True,
        'post_process': True,
        'persist_layer_norm': True,
        'gradient_as_bucket_view': True,
        'tokenizer': {
            'library': 'megatron',
            'type': 'GPT2BPETokenizer',
            'model': None,
            'vocab_file': os.path.join(test_data_dir, 'nlp/gpt_vocab_merges/vocab.json'),
            'merge_file': os.path.join(test_data_dir, 'nlp/gpt_vocab_merges/merges.txt'),
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
            'data_impl': 'hf_wikitext wikitext-2-v1',
            "data_prefix": [],
            'splits_string': '8,1,1',
            'seq_length': 512,
            'skip_warmup': True,
            'num_workers': 0,
            'dataloader_type': 'single',
            'reset_position_ids': False,
            'reset_attention_mask': False,
            'eod_mask_loss': False,
        },
        'optim': {
            'name': 'fused_adam',
            'lr': 2e-3,
            'weight_decay': 0.01,
            'betas': [0.9, 0.98],
            'sched': {'name': 'CosineAnnealing', 'warmup_steps': 10, 'constant_steps': 200, 'min_lr': 2e-4},
        },
    }
    return model_cfg


@pytest.fixture()
def trainer_cfg():

    trainer_cfg = {
        'devices': 2,
        'num_nodes': 1,
        'accelerator': 'gpu',
        'precision': 32,
        'logger': False,
        'enable_checkpointing': False,
        'use_distributed_sampler': False,
        'max_epochs': 1,
        'max_steps': 500,
        'val_check_interval': 100,
        'limit_val_batches': 1,
        'accumulate_grad_batches': 1,
        'gradient_clip_val': 1.0,
    }

    return trainer_cfg
