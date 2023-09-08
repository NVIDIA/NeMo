import os
import pytest
import torch

from omegaconf import DictConfig, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder


@pytest.mark.run_only_on('GPU')
class TestDistributedFusedAdam:

    def test_auto_config_bucket_cap_mb(gpt_1b_cfg, trainer_cfg):
        cfg = DictConfig({"model": gpt_1b_cfg, "trainer": trainer_cfg})

        def get_memory_allocated(cfg):
            trainer = MegatronTrainerBuilder(cfg).create_trainer()
            gpt = MegatronGPTModel(cfg.model, trainer)
            trainer.fit(gpt)
            max_memory_allocated = torch.cuda.max_memory_allocated()
            del gpt, trainer
            cuda_clean()

            return max_memory_allocated

        # bucket_cap_mb=125M
        with open_dict(cfg):
            cfg.model.optim.bucket_cap_mb = 125
        memory_with_cap_125m = get_memory_allocated(cfg)

        # bucket_cap_mb=10M
        with open_dict(cfg):
            cfg.model.optim.bucket_cap_mb = 20
        memory_with_cap_10m = get_memory_allocated(cfg)

        # bucket_cap_mb=auto
        with open_dict(cfg):
            cfg.model.optim.bucket_cap_mb = "auto"
        memory_with_cap_auto = get_memory_allocated(cfg)

        print("memory_with_cap_125m", memory_with_cap_125m)
        print("memory_with_cap_10m", memory_with_cap_10m)
        print("memory_with_cap_auto", memory_with_cap_auto)


def cuda_clean():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@pytest.fixture()
def gpt_1b_cfg(test_data_dir):

    model_cfg = {
        'precision': 16,
        'micro_batch_size': 1,
        'global_batch_size': 2,
        'tensor_model_parallel_size': 1,
        'pipeline_model_parallel_size': 1,
        'resume_from_checkpoint': None,
        'encoder_seq_length': 2048,
        'max_position_embeddings': 2048,
        'num_layers': 24,
        'hidden_size': 2048,
        'ffn_hidden_size': 8192,
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
            'data_prefix': 'mock',
            'index_mapping_dir': None,
            'data_impl': 'mmap',
            'splits_string': '900,50,50',
            'seq_length': 2048,
            'skip_warmup': True,
            'num_workers': 2,
            'dataloader_type': 'single',
            'reset_position_ids': False,
            'reset_attention_mask': False,
            'eod_mask_loss': False,
        },
        'optim': {
            'name': 'distributed_fused_adam',
            'lr': 2e-4,
            'weight_decay': 0.01,
            'betas': [0.9, 0.98],
            'bucket_cap_mb': 125,
            'contiguous_grad_buffer': True,
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
        'use_distributed_sampler': False,
        'max_epochs': 1000,
        'max_steps': 1,
        'log_every_n_steps': 10,
        'val_check_interval': 100,
        'limit_val_batches': 0,
        'limit_test_batches': 500,
        'accumulate_grad_batches': 1,
        'gradient_clip_val': 1.0,
    }

    return trainer_cfg
