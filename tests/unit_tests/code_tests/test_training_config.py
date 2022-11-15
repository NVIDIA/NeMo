import os

import pytest
from omegaconf import OmegaConf

from hp_tool import training_config as tc


class TestCalculateTpPpMbsGrid:

    margin = 0.05

    @pytest.mark.parametrize(
        "model_size,layers,model_name,train_cfg,expected",
        [
            # GPT-3 tests
            (0.126, 12, "gpt3", {"tensor_parallel_sizes": [1,2,4,5], "pipeline_parallel_sizes": [2,4,8], "micro_batch_sizes": [4,8,32], "gpu_memory_gb": 80, "min_model_parallel_size": 1, "max_model_parallel_size": 32}, {"tp": [1,2,4,5], "pp": [2,4,8], "mbs": [4,8,32], "min_par": 1, "max_par": 32}),
            (0.126, 12, "gpt3", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [1,2], "pp": [1], "mbs": [1,2,3,4,6,8], "min_par": 1, "max_par": 8}),
            (2.5, 24, "gpt3", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [1,2,4], "pp": [1], "mbs": [1,2,3,4,6,8], "min_par": 1, "max_par": 8}),
            (5.0, 24, "gpt3", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [1,2,4], "pp": [1], "mbs": [1,2,3,4,6,8], "min_par": 1, "max_par": 8}),
            (10.0, 24, "gpt3", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [1,2,4,8], "pp": [1], "mbs": [1,2,3,4,6,8], "min_par": 1, "max_par": 8}),
            (20.0, 44, "gpt3", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [1,2,4], "pp": [1,2,4], "mbs": [1,2,4], "min_par": 4, "max_par": 8}),
            (40.0, 48, "gpt3", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [2,4,8], "pp": [1,2,3,4], "mbs": [1,2,4], "min_par": 8, "max_par": 32}),
            (175.0, 96, "gpt3", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [8], "pp": [4,6,8,12,16], "mbs": [1,2,4], "min_par": 32, "max_par": 256}),
            # T5 tests
            (0.22, 12, "t5", {"tensor_parallel_sizes": [1,2,4,5], "pipeline_parallel_sizes": [2,4,8], "micro_batch_sizes": [4,8,32], "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [1,2,4,5], "pp": [2,4,8], "mbs": [4,8,32], "min_par": 1, "max_par": 8}),
            (0.22, 12, "t5", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [1,2], "pp": [1], "mbs": [16,32,64,128], "min_par": 1, "max_par": 8}),
            (3.0, 24, "t5", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [1,2,4], "pp": [1], "mbs": [4,6,8,12,16,24,32,48], "min_par": 1, "max_par": 8}),
            (11.0, 24, "t5", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [4,8], "pp": [1], "mbs": [2,4,6,8,12,16,24], "min_par": 1, "max_par": 8}),
            (23.0, 36, "t5", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [4,8], "pp": [1,2], "mbs": [1,2,4,6,8], "min_par": 4, "max_par": 16}),
            (41.0, 48, "t5", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [4,8], "pp": [1,2,4], "mbs": [1,2,4,6,8], "min_par": 8, "max_par": 32}),
            # mT5 tests
            (0.17, 6, "mt5", {"tensor_parallel_sizes": [1,2,4,5], "pipeline_parallel_sizes": [2,4,8], "micro_batch_sizes": [4,8,32], "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [1,2,4,5], "pp": [2,4,8], "mbs": [4,8,32], "min_par": 1, "max_par": 8}),
            (0.17, 6, "mt5", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [1,2], "pp": [1], "mbs": [16,32,64,128], "min_par": 1, "max_par": 8}),
            (0.39, 12, "mt5", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [1,2], "pp": [1], "mbs": [16,32,64,128], "min_par": 1, "max_par": 8}),
            (3.2, 24, "mt5", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [1,2,4], "pp": [1], "mbs": [4,6,8,12,16,24,32,48], "min_par": 1, "max_par": 8}),
            (11.9, 24, "mt5", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [4,8], "pp": [1], "mbs": [2,4,6,8,12,16,24], "min_par": 1, "max_par": 8}),
            (24.65, 36, "mt5", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [4,8], "pp": [1,2], "mbs": [1,2,4,6,8], "min_par": 4, "max_par": 16}),
            (42.54, 48, "mt5", {"tensor_parallel_sizes": "auto", "pipeline_parallel_sizes": "auto", "micro_batch_sizes": "auto", "gpu_memory_gb": 80, "min_model_parallel_size": "auto", "max_model_parallel_size": "auto"}, {"tp": [4,8], "pp": [1,2,4], "mbs": [1,2,4,6,8], "min_par": 8, "max_par": 32}),
        ],
    )
    def test_calculate_tp_pp_mbs_grid(self, model_size, layers, model_name, train_cfg, expected):
        params = {
            "model_size_in_b": model_size, 
            "num_layers": layers,
            "model_name": model_name,
            "train_cfg": train_cfg,
        }
        tp, pp, mbs, min_par, max_par = tc._calculate_tp_pp_mbs_grid(**params)

        assert tp == expected["tp"], f"TP should be {expected['tp']} but it is {tp}."
        assert pp == expected["pp"], f"PP should be {expected['pp']} but it is {pp}."
        assert mbs == expected["mbs"], f"MBS should be {expected['mbs']} but it is {mbs}."
        assert min_par == expected["min_par"], f"Minimum paralellism should be {expected['min_par']} but it is {min_par}."
        assert max_par == expected["max_par"], f"Minimum paralellism should be {expected['max_par']} but it is {max_par}."
