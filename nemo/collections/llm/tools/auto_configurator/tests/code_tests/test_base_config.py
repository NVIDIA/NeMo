import os

import autoconfig.base_config as bc
import pytest
from omegaconf import OmegaConf


class TestEstimateModelSize:

    margin = 0.05

    @pytest.mark.parametrize(
        "training_days,gpus,tflops,tokens,model_name,expected",
        [
            # GPT-3 tests
            # T5 tests
            (10, 4 * 8, 140, 1000, "t5", 0.48),
            (10, 8 * 8, 140, 1000, "t5", 0.97),
            (15, 8 * 8, 140, 1000, "t5", 1.45),
            (15, 16 * 8, 140, 1000, "t5", 2.9),
            (15, 20 * 8, 140, 1000, "t5", 3.6),
            (20, 20 * 8, 140, 1000, "t5", 4.8),
            (20, 32 * 8, 140, 1000, "t5", 7.7),
            (30, 32 * 8, 140, 1000, "t5", 11.6),
            (30, 40 * 8, 140, 1000, "t5", 14.5),
            (30, 48 * 8, 140, 1000, "t5", 17.4),
            (30, 60 * 8, 140, 1000, "t5", 21.8),
            (30, 80 * 8, 140, 1000, "t5", 29.0),
            (50, 80 * 8, 140, 1000, "t5", 48.4),
            # mT5 tests
        ],
    )
    def test_estimate_model_size(
        self, training_days, gpus, tflops, tokens, model_name, expected
    ):
        params = {
            "max_training_days": training_days,
            "gpu_count": gpus,
            "tflops_per_gpu": tflops,
            "num_tokens_in_b": tokens,
            "model_name": model_name,
        }
        output_size = bc._estimate_model_size(**params)
        assert output_size == pytest.approx(
            expected=expected, rel=self.margin
        ), f"Output of _estimate_model_size should be approximately {expected}, "
        f"but it is {output_size}. Inputs: max_training_days={training_days}, gpu_count={gpus}, "
        f"tflops_per_gpu={tflops}, num_tokens_in_b={tokens}, model_name={model_name}."

    def test_estimate_training_time_not_implemented_error(self):
        params = {
            "max_training_days": 1,
            "gpu_count": 8,
            "tflops_per_gpu": 140,
            "num_tokens_in_b": 300,
            "model_name": "invalid_name",
        }
        output_size = bc._estimate_model_size(**params)
        assert output_size == None


class TestEstimateTrainingTime:

    margin = 0.05

    @pytest.mark.parametrize(
        "model_size,gpus,tflops,tokens,model_name,expected",
        [
            # GPT-3 tests
            (0.126, 8 * 8, 140, 300, "gpt3", 0.4),
            (0.843, 8 * 8, 140, 300, "gpt3", 2.6),
            (2, 8 * 8, 140, 300, "gpt3", 6),
            (5, 20 * 8, 140, 300, "gpt3", 6),
            (8, 20 * 8, 140, 300, "gpt3", 9.9),
            (20, 80 * 8, 140, 300, "gpt3", 6),
            (43, 80 * 8, 140, 300, "gpt3", 13),
            (175, 128 * 8, 140, 300, "gpt3", 35),
            # T5 tests
            (0.22, 4 * 8, 140, 1000, "t5", 4.5),
            (2.8, 20 * 8, 140, 1000, "t5", 11.6),
            (11, 20 * 8, 140, 1000, "t5", 45.5),
            (23.5, 40 * 8, 140, 1000, "t5", 48.6),
            (41.2, 40 * 8, 140, 1000, "t5", 85.1),
            # mT5 tests
            (0.17, 4 * 8, 140, 1000, "mt5", 4.0),
            (0.39, 8 * 8, 140, 1000, "mt5", 4.6),
            (3.2, 20 * 8, 140, 1000, "mt5", 15.2),
            (11.9, 20 * 8, 140, 1000, "mt5", 56.6),
            (24.65, 40 * 8, 140, 1000, "mt5", 58.6),
            (42.54, 40 * 8, 140, 1000, "mt5", 101.1),
            # BERT tests
            (0.11, 8 * 8, 140, 300, "bert", 0.34),
            (4, 16 * 8, 140, 300, "bert", 6.2),
            (20, 64 * 8, 140, 300, "bert", 7.75),
        ],
    )
    def test_estimate_training_time(
        self, model_size, gpus, tflops, tokens, model_name, expected
    ):
        params = {
            "model_size_in_b": model_size,
            "gpu_count": gpus,
            "tflops_per_gpu": tflops,
            "num_tokens_in_b": tokens,
            "model_name": model_name,
        }
        output_days = bc._estimate_training_time(**params)
        assert output_days == pytest.approx(
            expected=expected, rel=self.margin
        ), f"Output of _estimate_training_time should be approximately {expected}, "
        f"but it is {output_days}. Inputs: model_size_in_b={model_size}, gpu_count={gpus}, "
        f"tflops_per_gpu={tflops}, num_tokens_in_b={tokens}, model_name={model_name}."

    def test_estimate_training_time_not_implemented_error(self):
        params = {
            "model_size_in_b": 1,
            "gpu_count": 8,
            "tflops_per_gpu": 140,
            "num_tokens_in_b": 300,
            "model_name": "invalid_name",
        }
        output_days = bc._estimate_training_time(**params)
        assert output_days == None


class TestCalculateGbsTpPp:
    @pytest.mark.parametrize(
        "model_size,model_name,seq_length,expected",
        [
            # GPT-3 tests
            (0.126, "gpt3", 2048, (256, 1, 1, 1, 1)),
            (3.0, "gpt3", 2048, (1024, 1, 1, 1, 1)),
            (5.0, "gpt3", 2048, (2048, 2, 1, 1, 1)),
            (10.0, "gpt3", 2048, (2048, 4, 1, 1, 1)),
            (20.0, "gpt3", 2048, (2048, 8, 1, 1, 1)),
            (40.0, "gpt3", 2048, (2048, 8, 2, 1, 1)),
            (80.0, "gpt3", 2048, (2048, 8, 4, 1, 1)),
            (175.0, "gpt3", 2048, (2048, 8, 8, 1, 1)),
            (300.0, "gpt3", 2048, (2048, 8, 16, 1, 1)),
            (600.0, "gpt3", 2048, (2048, 8, 32, 1, 1)),
            (1000.0, "gpt3", 2048, (2048, 8, 64, 1, 1)),
            # T5 tests
            (0.5, "t5", 512, (2048, 1, 1, None, None)),
            (3.0, "t5", 512, (1920, 2, 1, None, None)),
            (6.0, "t5", 512, (1920, 4, 1, None, None)),
            (13.0, "t5", 512, (1920, 8, 1, None, None)),
            (20.0, "t5", 512, (1920, 8, 2, None, None)),
            (40.0, "t5", 512, (1920, 8, 4, None, None)),
            # mT5 tests
            (0.5, "mt5", 512, (2048, 1, 1, None, None)),
            (3.0, "mt5", 512, (1920, 2, 1, None, None)),
            (6.0, "mt5", 512, (1920, 4, 1, None, None)),
            (13.0, "mt5", 512, (1920, 8, 1, None, None)),
            (20.0, "mt5", 512, (1920, 8, 2, None, None)),
            (40.0, "mt5", 512, (1920, 8, 4, None, None)),
            # BERT tests
            (0.11, "bert", 512, (256, 1, 1, None, None)),
            (3.0, "bert", 512, (1024, 1, 1, None, None)),
            (6.0, "bert", 512, (2048, 2, 1, None, None)),
            (13.0, "bert", 512, (2048, 4, 1, None, None)),
            (20.0, "bert", 512, (2048, 8, 1, None, None)),
        ],
    )
    def test_calculate_gbs_tp_pp(self, model_size, model_name, seq_length, expected):
        params = {
            "model_size_in_b": model_size,
            "model_name": model_name,
            "seq_length": seq_length,
        }
        output = bc._calculate_gbs_tp_pp(**params)
        assert (
            expected == output
        ), f"Output of _calculate_gbs_tp_pp should be {expected} but it is {output}."


class TestGenerateBaseconfig:

    margin = 0.05

    @pytest.mark.parametrize(
        "model_size,nodes,gpus_per_node,gpu_mem,max_days,tokens,vocab,seq_length,custom_cfg,model_name,cfg,expected",
        [
            # GPT-3 tests
            (
                0.126,
                8,
                8,
                80,
                2,
                300,
                51200,
                2048,
                None,
                "gpt3",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": True, "project": "test_project"},
                },
                {
                    "name": "gpt3_0.126b",
                    "time_limit": "2-00:00:00",
                    "max_steps": 572204,
                    "max_time": "1:23:30:00",
                    "num_layers": 12,
                    "gbs": 256,
                    "hs": 768,
                    "att_heads": 12,
                    "ffn": "${multiply:4, ${.hidden_size}}",
                    "kv": "null",
                    "init_std": 0.023,
                    "lr": 6e-4,
                    "min_lr": 6e-5,
                    "warmup_steps": 858,
                    "constant_steps": 95e3,
                    "warmup_ratio": None,
                },
            ),
            (
                5.0,
                20,
                8,
                80,
                6,
                300,
                51200,
                2048,
                None,
                "gpt3",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": False},
                },
                {
                    "name": "gpt3_5.0b",
                    "time_limit": "6-00:00:00",
                    "max_steps": 71525,
                    "max_time": "5:23:30:00",
                    "num_layers": 24,
                    "gbs": 2048,
                    "hs": 4096,
                    "att_heads": 32,
                    "ffn": "${multiply:4, ${.hidden_size}}",
                    "kv": "null",
                    "init_std": 0.01,
                    "lr": 1.2e-4,
                    "min_lr": 1.2e-5,
                    "warmup_steps": 107,
                    "constant_steps": 11873,
                    "warmup_ratio": None,
                },
            ),
            (
                20.0,
                80,
                8,
                80,
                6.5,
                300,
                51200,
                2048,
                None,
                "gpt3",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": True, "project": "test_project"},
                },
                {
                    "name": "gpt3_20.0b",
                    "time_limit": "6-12:00:00",
                    "max_steps": 71525,
                    "max_time": "6:11:30:00",
                    "num_layers": 44,
                    "gbs": 2048,
                    "hs": 6144,
                    "att_heads": 48,
                    "ffn": "${multiply:4, ${.hidden_size}}",
                    "kv": "null",
                    "init_std": 0.008165,
                    "lr": 1e-4,
                    "min_lr": 1e-5,
                    "warmup_steps": 107,
                    "constant_steps": 11873,
                    "warmup_ratio": None,
                },
            ),
            (
                40.0,
                80,
                8,
                80,
                25.75,
                300,
                51200,
                2048,
                None,
                "gpt3",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": False},
                },
                {
                    "name": "gpt3_40.0b",
                    "time_limit": "25-18:00:00",
                    "max_steps": 71525,
                    "max_time": "25:17:30:00",
                    "num_layers": 48,
                    "gbs": 2048,
                    "hs": 8192,
                    "att_heads": 64,
                    "ffn": "${multiply:4, ${.hidden_size}}",
                    "kv": "null",
                    "init_std": 0.007,
                    "lr": 0.8e-4,
                    "min_lr": 0.8e-5,
                    "warmup_steps": 107,
                    "constant_steps": 11873,
                    "warmup_ratio": None,
                },
            ),
            (
                175.0,
                128,
                8,
                80,
                35,
                300,
                51200,
                2048,
                None,
                "gpt3",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": False},
                },
                {
                    "name": "gpt3_175.0b",
                    "time_limit": "35-00:00:00",
                    "max_steps": 71525,
                    "max_time": "34:23:30:00",
                    "num_layers": 96,
                    "gbs": 2048,
                    "hs": 12288,
                    "att_heads": 96,
                    "ffn": "${multiply:4, ${.hidden_size}}",
                    "kv": "null",
                    "init_std": 0.006,
                    "lr": 0.6e-4,
                    "min_lr": 0.6e-5,
                    "warmup_steps": 107,
                    "constant_steps": 11873,
                    "warmup_ratio": None,
                },
            ),
            # T5 tests
            (
                0.22,
                4,
                8,
                80,
                2,
                1000,
                29000,
                512,
                None,
                "t5",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": True, "project": "test_project"},
                },
                {
                    "name": "t5_0.22b",
                    "time_limit": "2-00:00:00",
                    "max_steps": 953675,
                    "max_time": "1:23:30:00",
                    "num_layers": 12,
                    "gbs": 2048,
                    "hs": 768,
                    "att_heads": 12,
                    "ffn": 2048,
                    "kv": 64,
                    "init_std": 0.015,
                    "lr": 1e-4,
                    "min_lr": 1e-5,
                    "warmup_steps": None,
                    "constant_steps": None,
                    "warmup_ratio": 0.01,
                },
            ),
            (
                2.8,
                20,
                8,
                80,
                15,
                1000,
                29000,
                512,
                None,
                "t5",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": False},
                },
                {
                    "name": "t5_2.8b",
                    "time_limit": "15-00:00:00",
                    "max_steps": 1017250,
                    "max_time": "14:23:30:00",
                    "num_layers": 24,
                    "gbs": 1920,
                    "hs": 2048,
                    "att_heads": 32,
                    "ffn": 5120,
                    "kv": 64,
                    "init_std": 0.015,
                    "lr": 1e-4,
                    "min_lr": 1e-5,
                    "warmup_steps": None,
                    "constant_steps": None,
                    "warmup_ratio": 0.01,
                },
            ),
            (
                11.0,
                20,
                8,
                80,
                45,
                1000,
                29000,
                512,
                None,
                "t5",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": False},
                },
                {
                    "name": "t5_11.0b",
                    "time_limit": "45-00:00:00",
                    "max_steps": 1017250,
                    "max_time": "44:23:30:00",
                    "num_layers": 24,
                    "gbs": 1920,
                    "hs": 4096,
                    "att_heads": 64,
                    "ffn": 10240,
                    "kv": 64,
                    "init_std": 0.015,
                    "lr": 1e-4,
                    "min_lr": 1e-5,
                    "warmup_steps": None,
                    "constant_steps": None,
                    "warmup_ratio": 0.01,
                },
            ),
            (
                41.2,
                40,
                8,
                80,
                85,
                1000,
                29000,
                512,
                None,
                "t5",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": False},
                },
                {
                    "name": "t5_41.2b",
                    "time_limit": "85-00:00:00",
                    "max_steps": 1017250,
                    "max_time": "84:23:30:00",
                    "num_layers": 48,
                    "gbs": 1920,
                    "hs": 6144,
                    "att_heads": 96,
                    "ffn": 10880,
                    "kv": 64,
                    "init_std": 0.015,
                    "lr": 1e-4,
                    "min_lr": 1e-5,
                    "warmup_steps": None,
                    "constant_steps": None,
                    "warmup_ratio": 0.01,
                },
            ),
            # mT5 tests
            (
                0.17,
                4,
                8,
                80,
                4,
                1000,
                250000,
                512,
                None,
                "mt5",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": True, "project": "test_project"},
                },
                {
                    "name": "mt5_0.17b",
                    "time_limit": "4-00:00:00",
                    "max_steps": 953675,
                    "max_time": "3:23:30:00",
                    "num_layers": 8,
                    "gbs": 2048,
                    "hs": 512,
                    "att_heads": 6,
                    "ffn": 1024,
                    "kv": 64,
                    "init_std": 0.015,
                    "lr": 1e-4,
                    "min_lr": 1e-5,
                    "warmup_steps": None,
                    "constant_steps": None,
                    "warmup_ratio": 0.01,
                },
            ),
            (
                0.39,
                8,
                8,
                80,
                5,
                1000,
                250000,
                512,
                None,
                "mt5",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": True, "project": "test_project"},
                },
                {
                    "name": "mt5_0.39b",
                    "time_limit": "5-00:00:00",
                    "max_steps": 953675,
                    "max_time": "4:23:30:00",
                    "num_layers": 12,
                    "gbs": 2048,
                    "hs": 768,
                    "att_heads": 12,
                    "ffn": 2048,
                    "kv": 64,
                    "init_std": 0.015,
                    "lr": 1e-4,
                    "min_lr": 1e-5,
                    "warmup_steps": None,
                    "constant_steps": None,
                    "warmup_ratio": 0.01,
                },
            ),
            (
                3.2,
                20,
                8,
                80,
                14,
                1000,
                250000,
                512,
                None,
                "mt5",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": True, "project": "test_project"},
                },
                {
                    "name": "mt5_3.2b",
                    "time_limit": "14-00:00:00",
                    "max_steps": 1017250,
                    "max_time": "13:23:30:00",
                    "num_layers": 24,
                    "gbs": 1920,
                    "hs": 2048,
                    "att_heads": 32,
                    "ffn": 5120,
                    "kv": 64,
                    "init_std": 0.015,
                    "lr": 1e-4,
                    "min_lr": 1e-5,
                    "warmup_steps": None,
                    "constant_steps": None,
                    "warmup_ratio": 0.01,
                },
            ),
            (
                11.9,
                20,
                8,
                80,
                50,
                1000,
                250000,
                512,
                None,
                "mt5",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": True, "project": "test_project"},
                },
                {
                    "name": "mt5_11.9b",
                    "time_limit": "50-00:00:00",
                    "max_steps": 1017250,
                    "max_time": "49:23:30:00",
                    "num_layers": 24,
                    "gbs": 1920,
                    "hs": 4096,
                    "att_heads": 64,
                    "ffn": 10240,
                    "kv": 64,
                    "init_std": 0.015,
                    "lr": 1e-4,
                    "min_lr": 1e-5,
                    "warmup_steps": None,
                    "constant_steps": None,
                    "warmup_ratio": 0.01,
                },
            ),
            (
                24.65,
                40,
                8,
                80,
                55,
                1000,
                250000,
                512,
                None,
                "mt5",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": True, "project": "test_project"},
                },
                {
                    "name": "mt5_24.65b",
                    "time_limit": "55-00:00:00",
                    "max_steps": 1017250,
                    "max_time": "54:23:30:00",
                    "num_layers": 36,
                    "gbs": 1920,
                    "hs": 5120,
                    "att_heads": 80,
                    "ffn": 10880,
                    "kv": 64,
                    "init_std": 0.015,
                    "lr": 1e-4,
                    "min_lr": 1e-5,
                    "warmup_steps": None,
                    "constant_steps": None,
                    "warmup_ratio": 0.01,
                },
            ),
            (
                42.54,
                40,
                8,
                80,
                90.25,
                1000,
                250000,
                512,
                None,
                "mt5",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": True, "project": "test_project"},
                },
                {
                    "name": "mt5_42.54b",
                    "time_limit": "90-06:00:00",
                    "max_steps": 1017250,
                    "max_time": "90:05:30:00",
                    "num_layers": 48,
                    "gbs": 1920,
                    "hs": 6144,
                    "att_heads": 96,
                    "ffn": 10880,
                    "kv": 64,
                    "init_std": 0.015,
                    "lr": 1e-4,
                    "min_lr": 1e-5,
                    "warmup_steps": None,
                    "constant_steps": None,
                    "warmup_ratio": 0.01,
                },
            ),
            # BERT tests
            (
                0.11,
                8,
                8,
                80,
                2,
                1800,
                30522,
                512,
                None,
                "bert",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": True, "project": "test_project"},
                },
                {
                    "name": "bert_0.11b",
                    "time_limit": "2-00:00:00",
                    "max_steps": 13800000,
                    "max_time": "1:23:30:00",
                    "num_layers": 12,
                    "gbs": 256,
                    "hs": 768,
                    "att_heads": 12,
                    "ffn": 768 * 4,
                    "kv": "null",
                    "init_std": 0.023094,
                    "lr": 2e-4,
                    "min_lr": 2e-5,
                    "warmup_steps": 20000,
                    "constant_steps": 2300000,
                    "warmup_ratio": None,
                },
            ),
            (
                4.0,
                16,
                8,
                80,
                7,
                1800,
                30522,
                512,
                None,
                "bert",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": True, "project": "test_project"},
                },
                {
                    "name": "bert_4.0b",
                    "time_limit": "7-00:00:00",
                    "max_steps": 1720000,
                    "max_time": "6:23:30:00",
                    "num_layers": 48,
                    "gbs": 2048,
                    "hs": 2560,
                    "att_heads": 32,
                    "ffn": 2560 * 4,
                    "kv": "null",
                    "init_std": 0.012649,
                    "lr": 1e-4,
                    "min_lr": 1e-5,
                    "warmup_steps": 2600,
                    "constant_steps": 285000,
                    "warmup_ratio": None,
                },
            ),
            (
                20.0,
                64,
                8,
                80,
                12,
                1800,
                30522,
                512,
                None,
                "bert",
                {
                    "search_config": {"train_settings": {"logs": "."}},
                    "auto_configurator_path": ".",
                    "wandb": {"enable": True, "project": "test_project"},
                },
                {
                    "name": "bert_20.0b",
                    "time_limit": "12-00:00:00",
                    "max_steps": 1716613,
                    "max_time": "11:23:30:00",
                    "num_layers": 44,
                    "gbs": 2048,
                    "hs": 6144,
                    "att_heads": 48,
                    "ffn": 6144 * 4,
                    "kv": "null",
                    "init_std": 0.008165,
                    "lr": 1e-4,
                    "min_lr": 1e-5,
                    "warmup_steps": 2500,
                    "constant_steps": 285000,
                    "warmup_ratio": None,
                },
            ),
        ],
    )
    def test_generate_base_config(
        self,
        model_size,
        nodes,
        gpus_per_node,
        gpu_mem,
        max_days,
        tokens,
        vocab,
        seq_length,
        custom_cfg,
        model_name,
        cfg,
        expected,
    ):
        cfg = OmegaConf.create(cfg)
        params = {
            "model_size_in_b": model_size,
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
            "gpu_memory_gb": gpu_mem,
            "max_training_days": max_days,
            "num_tokens_in_b": tokens,
            "vocab_size": vocab,
            "seq_length": seq_length,
            "custom_cfg": custom_cfg,
            "model_name": model_name,
            "cfg": cfg,
        }
        out_cfg = bc.generate_base_config(**params)

        # Run parameters
        assert (
            out_cfg["run"]["name"] == expected["name"]
        ), "run.name doesn't match the expected value."
        assert (
            out_cfg["run"]["results_dir"] == "${base_results_dir}/${.name}"
        ), "run.results_dir must be set to ${base_results_dir}/${.name}"
        assert (
            out_cfg["run"]["time_limit"] == expected["time_limit"]
        ), "run.time_limit doesn't match the expected value."

        # Trainer parameters
        assert (
            out_cfg["trainer"]["num_nodes"] == nodes
        ), "trainer.num_nodes doesn't match the expected value."
        assert (
            out_cfg["trainer"]["precision"] == "bf16"
        ), "trainer.precision doesn't match the expected value."
        assert out_cfg["trainer"]["max_steps"] == pytest.approx(
            expected=expected["max_steps"], rel=self.margin
        ), f"trainer.max_steps is {out_cfg['trainer']['max_steps']} but it should be {expected['max_steps']}."
        assert (
            out_cfg["trainer"]["max_time"] == expected["max_time"]
        ), "trainer.max_time doesn't match the expected value."

        # Exp_manager parameters
        if cfg["wandb"]["enable"]:
            assert out_cfg["exp_manager"][
                "create_wandb_logger"
            ], "exp_manager.create_wandb_logger should be True."
            assert (
                out_cfg["exp_manager"]["wandb_logger_kwargs"]["project"]
                == cfg["wandb"]["project"]
            ), "exp_manager.wandb_logger_kwargs.project doesn't match the expected value."
        else:
            assert not out_cfg["exp_manager"][
                "create_wandb_logger"
            ], "exp_manager.create_wandb_logger should be False."

        # Model parameters
        if model_name in ["gpt3", "bert"]:
            assert out_cfg["model"]["num_layers"] == expected["num_layers"]
            assert out_cfg["model"]["hidden_size"] == expected["hs"]
            assert out_cfg["model"]["num_attention_heads"] == expected["att_heads"]
            if out_cfg["model"]["ffn_hidden_size"] is not None:
                assert out_cfg["model"]["ffn_hidden_size"] == expected["ffn"]
            if out_cfg["model"]["kv_channels"] is not None:
                assert out_cfg["model"]["kv_channels"] == expected["kv"]
        else:
            assert out_cfg["model"]["encoder"]["num_layers"] == expected["num_layers"]
            assert out_cfg["model"]["encoder"]["hidden_size"] == expected["hs"]
            assert (
                out_cfg["model"]["encoder"]["num_attention_heads"]
                == expected["att_heads"]
            )
            if out_cfg["model"]["encoder"]["ffn_hidden_size"] is not None:
                assert out_cfg["model"]["encoder"]["ffn_hidden_size"] == expected["ffn"]
            if out_cfg["model"]["encoder"]["kv_channels"] is not None:
                assert out_cfg["model"]["encoder"]["kv_channels"] == expected["kv"]

        assert out_cfg["model"]["global_batch_size"] == expected["gbs"]
        assert out_cfg["model"]["init_method_std"] == pytest.approx(
            expected=expected["init_std"], rel=self.margin
        )
        assert out_cfg["model"]["optim"]["lr"] == expected["lr"]
        assert out_cfg["model"]["optim"]["sched"]["min_lr"] == pytest.approx(
            expected=expected["min_lr"], rel=self.margin
        )
        if out_cfg["model"]["optim"]["sched"].get("warmup_steps") is not None:
            assert out_cfg["model"]["optim"]["sched"]["warmup_steps"] == pytest.approx(
                expected=expected["warmup_steps"], rel=self.margin
            )
        if out_cfg["model"]["optim"]["sched"].get("constant_steps") is not None:
            assert out_cfg["model"]["optim"]["sched"][
                "constant_steps"
            ] == pytest.approx(expected=expected["constant_steps"], rel=self.margin)
        if out_cfg["model"]["optim"]["sched"].get("warmup_ratio") is not None:
            assert out_cfg["model"]["optim"]["sched"]["warmup_ratio"] == pytest.approx(
                expected=expected["warmup_ratio"], rel=self.margin
            )

        f = f"{cfg['search_config']['train_settings']['logs']}/base_cfg_{model_size}b.yaml"
        assert os.path.exists(f), "Base config file was not created correctly."
        os.remove(f)
