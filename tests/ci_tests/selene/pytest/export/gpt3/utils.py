import json
import pathlib


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
