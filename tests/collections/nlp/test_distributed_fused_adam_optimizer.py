import gc
import os
import time

import pytest
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder


@pytest.mark.run_only_on('GPU')
class TestDistributedFusedAdam:
    @pytest.mark.unit
    def test_auto_config_bucket_cap_mb(self, gpt_127m_cfg, trainer_cfg):
        cfg = DictConfig({"model": gpt_127m_cfg, "trainer": trainer_cfg})

        # bucket_cap_mb=125
        with open_dict(cfg):
            cfg.model.optim.bucket_cap_mb = 125
        step_time_with_cap_125m, memory_with_cap_125m = benchmark(cfg)
        print("memory_with_cap_125m", memory_with_cap_125m, step_time_with_cap_125m)

        # bucket_cap_mb=10
        with open_dict(cfg):
            cfg.model.optim.bucket_cap_mb = 10
        step_time_with_cap_10m, memory_with_cap_10m = benchmark(cfg)
        print("memory_with_cap_10m", memory_with_cap_10m, step_time_with_cap_10m)

        # bucket_cap_mb=auto
        with open_dict(cfg):
            cfg.model.optim.bucket_cap_mb = "auto"
        step_time_with_cap_auto, memory_with_cap_auto = benchmark(cfg)
        print("memory_with_cap_auto", memory_with_cap_auto, step_time_with_cap_auto)

        # bucket_cap_mb=auto saves far more memory than 125mb bucket cap
        assert memory_with_cap_125m / memory_with_cap_auto > 1.5

        # bucket_cap_mb=auto saves more memory than 10mb bucket cap
        assert memory_with_cap_10m / memory_with_cap_auto > 1.0

        # bucket_cap_mb=auto is more efficient than 10mb bucket cap
        assert step_time_with_cap_auto / step_time_with_cap_10m < 1.0


def benchmark(cfg):
    gc.collect()
    cuda_clean()

    builder = MegatronTrainerBuilder(cfg)
    strategy = builder._training_strategy()
    plugins = builder._plugins()
    metrics = MetricsCallback()
    trainer = Trainer(**cfg.trainer, strategy=strategy, plugins=plugins, callbacks=[metrics,],)
    gpt = MegatronGPTModel(cfg.model, trainer)
    trainer.fit(gpt)
    torch.cuda.synchronize()
    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(torch.cuda.memory_summary())

    avg_step_time = sum(metrics.step_time) / len(metrics.step_time)

    return avg_step_time, max_memory_allocated


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()

        self.step_time = []
        self.last_train_ts = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.last_train_ts is None:
            self.last_train_ts = time.time()
            return

        self.step_time.append(time.time() - self.last_train_ts)
        self.last_train_ts = time.time()


def cuda_clean():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    time.sleep(1)


@pytest.fixture()
def gpt_127m_cfg(test_data_dir):

    model_cfg = {
        'precision': 16,
        'micro_batch_size': 1,
        'global_batch_size': 2,
        'tensor_model_parallel_size': 1,
        'pipeline_model_parallel_size': 1,
        'resume_from_checkpoint': None,
        'encoder_seq_length': 2048,
        'max_position_embeddings': 2048,
        'num_layers': 12,
        'hidden_size': 768,
        'ffn_hidden_size': 3072,
        'num_attention_heads': 12,
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
            'data_impl': 'mock',
            'data_prefix': [],
            'index_mapping_dir': None,
            'splits_string': '1,1,1',
            'seq_length': 2048,
            'skip_warmup': True,
            'num_workers': 0,
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
            'sched': {'name': 'CosineAnnealing', 'warmup_steps': 500, 'constant_steps': 50000, 'min_lr': 2e-5},
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
        'max_steps': 10,
        'log_every_n_steps': 10,
        'val_check_interval': 2,
        'limit_val_batches': 1,
        'limit_test_batches': 1,
        'accumulate_grad_batches': 1,
        'gradient_clip_val': 1.0,
    }

    return trainer_cfg
