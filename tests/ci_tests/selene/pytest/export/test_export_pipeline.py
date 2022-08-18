import os
import pathlib
import json
import pytest


def load_export_metrics(export_results_dir):
    export_results_dir = pathlib.Path(export_results_dir)
    export_metrics_path = export_results_dir / "eval_output.json"
    assert export_metrics_path.exists(), f"Could not found {export_metrics_path} file containing export metrics"
    with export_metrics_path.open("r") as metrics_file:
        export_metrics = json.load(metrics_file)["results"]["lambada"]

    return export_metrics


def load_eval_metrics(eval_results_dir):
    eval_results_dir = pathlib.Path(eval_results_dir)
    eval_metrics_paths = list(eval_results_dir.rglob("metrics.json"))
    assert len(eval_metrics_paths) == 1, f"Only one metrics.json file should be present inside {eval_results_dir}"
    eval_metrics_path = eval_metrics_paths[0]
    with eval_metrics_path.open("r") as metrics_file:
        eval_metrics = json.load(metrics_file)["lambada"]
    return eval_metrics

CI_JOB_RESULTS = os.environ.get("RESULTS_DIR")

RUN_MODEL = os.environ.get("RUN_MODEL")
RUN_MODEL_SIZE = os.environ.get("RUN_MODEL_SIZE")
TP_SIZE = os.environ.get("TP_SIZE")
PP_SIZE = os.environ.get("PP_SIZE")
FT_EVAL_MARGIN = float(os.environ.get("FT_EVAL_MARGIN"))
FT_EVAL_TASK = float(os.environ.get("FT_EVAL_TASK"))


class TestExportBigNLPCI:

    MARGIN = FT_EVAL_MARGIN  # 0.5% relative difference
    EVAL_JOB_NAME = f"eval_{RUN_MODEL}_{RUN_MODEL_SIZE}_tp{TP_SIZE}_pp{PP_SIZE}_{FT_EVAL_TASK}"

    def test_ci_export_metrics(self):
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
