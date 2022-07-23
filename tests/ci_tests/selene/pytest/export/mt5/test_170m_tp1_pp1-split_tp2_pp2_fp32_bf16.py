import os
import pathlib

import pytest

from utils import load_eval_metrics, load_export_metrics


CI_JOB_RESULTS = os.environ.get("RESULTS_DIR")


class TestExportmT5BigNLPCI:

    MARGIN = 0.005  # 0.5% relative difference
    EVAL_JOB_NAME = "eval_mt5_170m_tp1_pp1_xnli"

    def test_ci_export_mt5_metrics(self):
        export_results_dir = pathlib.Path(CI_JOB_RESULTS)
        export_metrics = load_export_metrics(export_results_dir)

        print("export metrics")
        print(export_metrics)

        base_results_dir = export_results_dir.parent
        eval_results_dir = base_results_dir / self.EVAL_JOB_NAME
        eval_metrics = load_eval_metrics(eval_results_dir)

        print("eval metrics")
        print(eval_metrics)

        assert export_metrics["acc"] == pytest.approx(
            expected=eval_metrics["validation_exact_string_match"], rel=self.MARGIN
        ), (
            f"XNLI accuracy obtained with exported model: {export_metrics['acc']} "
            f"should be equal to evaluation: {eval_metrics['validation_exact_string_match']} (rel={self.MARGIN})"
        )
