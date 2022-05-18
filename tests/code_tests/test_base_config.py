import os

import pytest
from omegaconf import OmegaConf

from hp_tool import base_config as bc


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
        with pytest.raises(NotImplementedError):
            output_size = bc._estimate_model_size(**params)


class TestEstimateTrainingTime:

    margin = 0.05

    @pytest.mark.parametrize(
        "model_size,gpus,tflops,tokens,model_name,expected",
        [
            # GPT-3 tests
            (0.126, 8 * 8, 140, 300, "gpt3", 0.4),
            (5, 20 * 8, 140, 300, "gpt3", 6),
            (20, 80 * 8, 140, 300, "gpt3", 6),
            (40, 80 * 8, 140, 300, "gpt3", 12.5),
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
        with pytest.raises(NotImplementedError):
            output_days = bc._estimate_training_time(**params)


class TestCalculateGbsTpPp:
    @pytest.mark.parametrize(
        "model_size,model_name,expected",
        [
            # GPT-3 tests
            (0.126, "gpt3", (256, 1, 1)),
            (3.0, "gpt3", (720, 1, 1)),
            (5.0, "gpt3", (1440, 2, 1)),
            (10.0, "gpt3", (1440, 4, 1)),
            (20.0, "gpt3", (1440, 8, 1)),
            (40.0, "gpt3", (1440, 8, 4)),
            (80.0, "gpt3", (1440, 8, 8)),
            (175.0, "gpt3", (1536, 8, 16)),
            (300.0, "gpt3", (1792, 8, 32)),
            (600.0, "gpt3", (1920, 8, 64)),
            (1000.0, "gpt3", (2048, 8, 128)),
            # T5 tests
            (0.5, "t5", (2048, 1, 1)),
            (3.0, "t5", (1920, 2, 1)),
            (6.0, "t5", (1920, 4, 1)),
            (13.0, "t5", (1920, 8, 1)),
            (20.0, "t5", (1920, 8, 2)),
            (40.0, "t5", (1920, 8, 4)),
            # mT5 tests
            (0.5, "mt5", (2048, 1, 1)),
            (3.0, "mt5", (1920, 2, 1)),
            (6.0, "mt5", (1920, 4, 1)),
            (13.0, "mt5", (1920, 8, 1)),
            (20.0, "mt5", (1920, 8, 2)),
            (40.0, "mt5", (1920, 8, 4)),
        ],
    )
    def test_calculate_gbs_tp_pp(self, model_size, model_name, expected):
        params = {"model_size_in_b": model_size, "model_name": model_name}
        output = bc._calculate_gbs_tp_pp(**params)
        assert (
            expected == output
        ), f"Output of _calculate_gbs_tp_pp should be {expected} but it is {output}."


class TestGenerateBaseconfig:
    
    margin = 0.05
    
    @pytest.mark.parametrize(
        "model_size,nodes,gpus_per_node,max_days,tokens,vocab,model_name,cfg,expected",
        [
            # GPT-3 tests
            (0.126, 8, 8, 2, 300, 51200, "gpt3", {"search_config": {"train_settings": {"logs": "."}}, "bignlp_hp_tool_path": ".", "wandb": {"enable": True, "project": "test_project"}}, {"name": "gpt3_0.126b", "time_limit": "2-0:00:00", "max_steps": 572204, "max_time": "1:23:30:00", "num_layers": 12, "gbs": 256, "hs": 768, "att_heads": 12, "ffn": "${multiply:4, ${.hidden_size}}", "kv": "null", "init_std": 0.023, "lr": 6e-4, "min_lr": 6e-5, "warmup_steps": 858, "constant_steps": 95e3, "warmup_ratio": None}),
        ],
    )
    def test_generate_base_config(
        self,
        model_size,
        nodes,
        gpus_per_node,
        max_days,
        tokens,
        vocab,
        model_name,
        cfg,
        expected,
    ):
        cfg = OmegaConf.create(cfg)
        params = {
            "model_size_in_b": model_size,
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
            "max_training_days": max_days,
            "num_tokens_in_b": tokens,
            "vocab_size": vocab,
            "model_name": model_name,
            "cfg": cfg,
        }
        out_cfg = bc.generate_base_config(**params)

        # Run parameters
        assert out_cfg["run"]["name"] == expected["name"], "run.name doesn't match the expected value."
        assert out_cfg["run"]["results_dir"] == "${base_results_dir}/${.name}", "run.results_dir must be set to ${base_results_dir}/${.name}"
        assert out_cfg["run"]["time_limit"] == expected["time_limit"], "run.time_limit doesn't match the expected value."

        # Trainer parameters
        assert out_cfg["trainer"]["num_nodes"] == nodes, "trainer.num_nodes doesn't match the expected value."
        assert out_cfg["trainer"]["precision"] == "bf16", "trainer.precision doesn't match the expected value."
        assert out_cfg["trainer"]["max_steps"] == expected["max_steps"], "trainer.max_steps doesn't match the expected value."
        assert out_cfg["trainer"]["max_time"] == expected["max_time"], "trainer.max_time doesn't match the expected value."

        # Exp_manager parameters
        if cfg["wandb"]["enable"]:
            assert out_cfg["exp_manager"]["create_wandb_logger"], "exp_manager.create_wandb_logger should be True."
            assert out_cfg["exp_manager"]["wandb_logger_kwargs"]["project"] == cfg["wandb"]["project"], "exp_manager.wandb_logger_kwargs.project doesn't match the expected value."
        else:
            assert not out_cfg["exp_manager"]["create_wandb_logger"], "exp_manager.create_wandb_logger should be False."

        # Model parameters
        assert out_cfg["model"]["num_layers"] == expected["num_layers"]
        assert out_cfg["model"]["global_batch_size"] == expected["gbs"]
        assert out_cfg["model"]["hidden_size"] == expected["hs"]
        assert out_cfg["model"]["num_attention_heads"] == expected["att_heads"]
        if out_cfg["model"]["ffn_hidden_size"] is not None:
            assert out_cfg["model"]["ffn_hidden_size"] == expected["ffn"]
        if out_cfg["model"]["kv_channels"] is not None:
            assert out_cfg["model"]["kv_channels"] == expected["kv"]
        assert out_cfg["model"]["init_method_std"] == pytest.approx(expected=expected["init_std"], rel=self.margin)
        assert out_cfg["model"]["optim"]["lr"] == expected["lr"]
        assert out_cfg["model"]["optim"]["sched"]["min_lr"] == pytest.approx(expected=expected["min_lr"], rel=self.margin)
        if out_cfg["model"]["optim"]["sched"]["warmup_steps"] is not None:
            assert out_cfg["model"]["optim"]["sched"]["warmup_steps"] == pytest.approx(expected=expected["warmup_steps"], rel=self.margin)
        if out_cfg["model"]["optim"]["sched"]["constant_steps"] is not None:
            assert out_cfg["model"]["optim"]["sched"]["constant_steps"] == pytest.approx(expected=expected["constant_steps"], rel=self.margin)
        if out_cfg["model"]["optim"]["sched"].get("warmup_ratio") is not None:
            assert out_cfg["model"]["optim"]["sched"]["warmup_ratio"] == pytest.approx(expected=expected["warmup_ratio"], rel=self.margin)

        f = f"{cfg['search_config']['train_settings']['logs']}/base_cfg_{model_size}b.yaml"
        assert os.path.exists(f), "Base config file was not created correctly."
        os.remove(f)
