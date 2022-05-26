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
            (0.126, 12, "gpt3", {"tensor_parallel_sizes": [1,2,4,5], "pipeline_parallel_sizes": [2,4,8], "micro_batch_sizes": [4,8,32]}, {"tp": [1,2,4,5], "pp": [2,4,8], "mbs": [4,8,32]}),
            (0.126, 12, "gpt3", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [1,2], "pp": [1], "mbs": [1,2,4,8]}),
            (2.5, 24, "gpt3", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [1,2,4], "pp": [1], "mbs": [1,2,4,8]}),
            (5.0, 24, "gpt3", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [2,4,8], "pp": [1], "mbs": [1,2,4,8]}),
            (10.0, 24, "gpt3", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [4,8], "pp": [1], "mbs": [1,2,4,8]}),
            (20.0, 44, "gpt3", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [8], "pp": [1,2,4], "mbs": [1,2,4,8]}),
            (40.0, 48, "gpt3", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [8], "pp": [2,3,4,6], "mbs": [1,2,4,8]}),
            (175.0, 96, "gpt3", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [8], "pp": [8,12,16,24], "mbs": [1,2]}),
            # T5 tests
            (0.22, 12, "t5", {"tensor_parallel_sizes": [1,2,4,5], "pipeline_parallel_sizes": [2,4,8], "micro_batch_sizes": [4,8,32]}, {"tp": [1,2,4,5], "pp": [2,4,8], "mbs": [4,8,32]}),
            (0.22, 12, "t5", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [1,2], "pp": [1], "mbs": [16,32,64,128]}),
            (3.0, 24, "t5", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [1,2,4], "pp": [1], "mbs": [4,6,8,12,16,24,32,48]}),
            (11.0, 24, "t5", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [4,8], "pp": [1], "mbs": [2,4,6,8,12,16,24]}),
            (23.0, 36, "t5", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [4,8], "pp": [1,2], "mbs": [1,2,4,6,8]}),
            (41.0, 48, "t5", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [4,8], "pp": [1,2,4], "mbs": [1,2,4,6,8]}),
            # mT5 tests
            (0.17, 6, "mt5", {"tensor_parallel_sizes": [1,2,4,5], "pipeline_parallel_sizes": [2,4,8], "micro_batch_sizes": [4,8,32]}, {"tp": [1,2,4,5], "pp": [2,4,8], "mbs": [4,8,32]}),
            (0.17, 6, "mt5", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [1,2], "pp": [1], "mbs": [16,32,64,128]}),
            (0.39, 12, "mt5", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [1,2], "pp": [1], "mbs": [16,32,64,128]}),
            (3.2, 24, "mt5", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [1,2,4], "pp": [1], "mbs": [4,6,8,12,16,24,32,48]}),
            (11.9, 24, "mt5", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [4,8], "pp": [1], "mbs": [2,4,6,8,12,16,24]}),
            (24.65, 36, "mt5", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [4,8], "pp": [1,2], "mbs": [1,2,4,6,8]}),
            (42.54, 48, "mt5", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [4,8], "pp": [1,2,4], "mbs": [1,2,4,6,8]}),
        ],
    )
    def test_calculate_tp_pp_mbs_grid(self, model_size, layers, model_name, train_cfg, expected):
        params = {
            "model_size_in_b": model_size, 
            "num_layers": layers,
            "model_name": model_name,
            "train_cfg": train_cfg,
        }
        tp, pp, mbs = tc._calculate_tp_pp_mbs_grid(**params)

        assert tp == expected["tp"], "TP should be {expected['tp']} but it is {tp}."
        assert pp == expected["pp"], "PP should be {expected['pp']} but it is {pp}."
        assert mbs == expected["mbs"], "MBS should be {expected['mbs']} but it is {mbs}."
