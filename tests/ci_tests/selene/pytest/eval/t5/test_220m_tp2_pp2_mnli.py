import os

import json
import pytest

from tensorboard.backend.event_processing import event_accumulator


CI_JOB_RESULTS = os.environ.get("RESULTS_DIR")

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
            print(summary_list)
            return summary_list
    raise FileNotFoundError(f"File not found matching: {path}/events*")


class TestCIT5_220m:

    margin_loss, margin_time = 0.05, 0.1
    expected_json = \
    r"""
    {"val_loss": {"start_step": 0, "end_step": 1, "step_interval": 1, "values": [8.07252]}}
    """

    expected = json.loads(expected_json)

    def test_ci_t5_220m_val_loss_deterministic(self):
        # Expected validation loss curve at different global steps.
        expected = self.expected["val_loss"]
        expected_vals = expected["values"]
        val_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "val_loss")

        assert val_loss_list is not None, f"No TensorBoard events file was found in the logs."
        assert len(val_loss_list) == 1, f"The events file must have 1 validation loss values."
        for i, step in enumerate(range(expected["start_step"], expected["end_step"], expected["step_interval"])):
            assert val_loss_list[step] == expected_vals[i], f"The loss at step {step} should be {expected_vals[i]} but it is {val_loss_list[step]}."

    def test_ci_t5_220m_val_loss_approx(self):
        # Expected validation loss curve at different global steps.
        expected = self.expected["val_loss"]
        expected_vals = expected["values"]
        val_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "val_loss")

        assert val_loss_list is not None, f"No TensorBoard events file was found in the logs."
        assert len(val_loss_list) == 1, f"The events file must have 1 validation loss values."
        for i, step in enumerate(range(expected["start_step"], expected["end_step"], expected["step_interval"])):
            assert val_loss_list[step] == pytest.approx(expected=expected_vals[i], rel=self.margin_loss), f"The loss at step {step} should be approximately {expected_vals[i]} but it is {val_loss_list[step]}."
