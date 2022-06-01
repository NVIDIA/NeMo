import os

import pytest


CI_JOB_RESULTS = os.environ.get("RESULTS_DIR")

class TestCIGPT126m:

    def test_ci_eval_gpt3_126m_tp1_pp1_lambada(self):
        eval_dir = os.path.join(CI_JOB_RESULTS, "eval_lambada")
        assert os.path.exists(eval_dir), f"Eval dir does not exist: {eval_dir}"
