import os
import json
from pathlib import Path

import pytest


CI_JOB_RESULTS = os.environ.get("RESULTS_DIR")

class TestCIGPT126m:

    margin = 0.05
    expected = {
        "prompt": {
            "ppl": 896469,
            "ppl_stderr": 66318,
            "acc": 0.0,
            "acc_stderr": 0.0,
        },
    }

    def test_ci_eval_gpt3_126m_tp1_pp1_prompt(self):
        p = Path(CI_JOB_RESULTS)
        files = list(p.glob('eval_gpt3_prompt_126m_tp1_pp1_squad*/metrics.json'))
        print(p)
        print(files)
        assert len(files) == 1, f"Only one metrics.json file should be present inside {CI_JOB_RESULTS}"

        metrics_file = files[0]
        assert os.path.exists(metrics_file), f"metrics.json file does not exist: {metrics_file}"

        with open(metrics_file) as json_file:
            metrics = json.load(json_file)["prompt"]
            print(metrics)
            expected_prompt = self.expected["prompt"]
            assert metrics["ppl"] == pytest.approx(expected=expected_prompt["ppl"], rel=self.margin), f"Lambada PPL should be {expected_prompt['ppl']} but it is {metrics['ppl']}"
            assert metrics["ppl_stderr"] == pytest.approx(expected=expected_prompt["ppl_stderr"], rel=self.margin), f"Lambada PPL StdErr should be {expected_prompt['ppl_stderr']} but it is {metrics['ppl_stderr']}"
            assert metrics["acc"] == pytest.approx(expected=expected_prompt["acc"], rel=self.margin), f"Lambada Accuracy should be {expected_prompt['acc']} but it is {metrics['acc']}"
            assert metrics["acc_stderr"] == pytest.approx(expected=expected_prompt["acc_stderr"], rel=self.margin), f"Lambada Accuracy StdErr should be {expected_prompt['acc_stderr']} but it is {metrics['acc_stderr']}"
