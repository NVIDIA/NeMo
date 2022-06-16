import os
import sys
import json

from tensorboard.backend.event_processing import event_accumulator


CI_JOB_RESULTS = os.environ.get("RESULTS_DIR")
RUN_TASK = os.environ.get("RUN_TASK")

def _read_tb_logs_as_list(path, summary_name):
    """Reads a TensorBoard Events file from the input path, and returns the
    summary specified as input as a list.

    Arguments:
        path: str, path to the dir where the events file is located.
        summary_name: str, name of the summary to read from the TB logs.
    Output:
        summary_list: list, the values in the read summary list, formatted as a list.
    """
    files = os.listdir(path)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
    for f in files:
        if f[:6] == "events":
            event_file = os.path.join(path, f)
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            summary = ea.Scalars(summary_name)
            summary_list = [round(x.value, 5) for x in summary]
            return summary_list
    raise FileNotFoundError(f"File not found matching: {path}/events* \nFiles: {files}")

def collect_train_test_metrics(pytest_file):
    # TODO: Fetch current baseline

    # train loss
    train_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "reduced_train_loss")

    # val loss
    val_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "val_loss")

    # step timing
    train_time_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "train_step_timing")
    train_time_list = train_time_list[len(train_time_list) // 2:]  # Discard the first half.
    train_time_avg = sum(train_time_list) / len(train_time_list)

    train_metrics = {
        "reduced_train_loss": {
            "start_step": 0,
            "end_step": 100,
            "step_interval": 5,
            "values": train_loss_list[0:100:5],
        },
        "val_loss": {
            "start_step": 0,
            "end_step": 5,
            "step_interval": 1,
            "values": val_loss_list[0:5],
        },
        "train_step_timing_avg": train_time_avg,
    }
    train_metrics_file = os.path.join(CI_JOB_RESULTS, "ci_train_metrics.json")
    with open(train_metrics_file, "w") as out_file:
        json.dump(train_metrics, out_file)
    print(f" ****** CI train metrics logged in {train_metrics_file}", flush=True)

if __name__ == '__main__':
    args = sys.argv[1:]
    pytest_file = args[0]

    if RUN_TASK == "train" or RUN_TASK == "finetune":
        collect_train_test_metrics(pytest_file)

