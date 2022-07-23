import os
import pathlib

import pytest

from utils import load_eval_metrics, load_export_metrics

CI_JOB_RESULTS = os.environ.get("RESULTS_DIR")


class TestExportGPT3BigNLPCI:

    MARGIN = 0.005  # 0.5% relative difference
    EVAL_JOB_NAME = "eval_gpt3_126m_tp2_pp2_lambada"

    def test_ci_export_gpt3_metrics(self):
        export_results_dir = pathlib.Path(CI_JOB_RESULTS)
        export_metrics = load_export_metrics(CI_JOB_RESULTS)

        print("export metrics")
        print(export_metrics)

        base_results_dir = export_results_dir.parent
        eval_results_dir = base_results_dir / self.EVAL_JOB_NAME
        eval_metrics = load_eval_metrics(eval_results_dir)

        print("eval metrics")
        print(eval_metrics)

        assert export_metrics["acc"] == pytest.approx(expected=eval_metrics["acc"], rel=self.MARGIN), (
            f"Lambada accuracy obtained with exported model: {export_metrics['acc']} "
            f"should be equal to evaluation: {eval_metrics['acc']} (rel={self.MARGIN})"
        )