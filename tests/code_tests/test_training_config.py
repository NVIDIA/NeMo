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
            (0.126, 12, "gpt3", {"tensor_parallel_sizes": None, "pipeline_parallel_sizes": None, "micro_batch_sizes": None}, {"tp": [1,2], "pp": [1], "mbs": [1,2,4,8]}),
            (0.126, 12, "gpt3", {"tensor_parallel_sizes": [1,2,4,5], "pipeline_parallel_sizes": [2,4,8], "micro_batch_sizes": [4,8,32]}, {"tp": [1,2,4,5], "pp": [2,4,8], "mbs": [4,8,32]}),
            # T5 tests
            # mT5 tests
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
