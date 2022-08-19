import json
import pathlib
import re


def load_export_metrics(export_results_dir):
    export_results_dir = pathlib.Path(export_results_dir)
    export_metrics_path = export_results_dir / "eval_output.json"
    assert export_metrics_path.exists(), f"Could not found {export_metrics_path} file containing export metrics"
    with export_metrics_path.open("r") as metrics_file:
        export_metrics = json.load(metrics_file)["results"]["xnli"]

    return export_metrics


def load_eval_metrics(eval_results_dir):
    slurm_logs_paths = list(eval_results_dir.rglob("slurm*log"))
    assert len(slurm_logs_paths) == 1, f"Only one slurm log file should be present inside {eval_results_dir}"
    slurm_log_path = slurm_logs_paths[0]

    # parse eval metrics
    slurm_logs_lines = [line for line in slurm_log_path.read_text(encoding="utf-8", errors="replace").splitlines()]
    lines_numbers_with_separator = [idx for idx, line in enumerate(slurm_logs_lines) if line.startswith("──────")]
    assert (
            len(lines_numbers_with_separator) == 3
    ), "There should be section with metrics separated from other logs with 3x'──────' lines"

    start_idx = lines_numbers_with_separator[1] + 1  # row below bottom header separator
    stop_idx = lines_numbers_with_separator[2] - 1  # row above bottom table separator
    eval_metrics_lines = [slurm_logs_lines[line_no].strip() for line_no in range(start_idx, stop_idx)]
    eval_metrics = dict(tuple(re.split(r"\s+", line)) for line in eval_metrics_lines)
    eval_metrics = {k: float(v) for k, v in eval_metrics.items()}
    return eval_metrics
