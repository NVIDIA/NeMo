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


class TestBignlpCI:

    margin_loss, margin_time = 0.05, 0.1
    expected_json = \
    r"""
    {"reduced_train_loss": {"start_step": 0, "end_step": 100, "step_interval": 5, "values": [10.36845, 9.24566, 8.75846, 8.36097, 8.03214, 7.71757, 7.52091, 7.32731, 7.17058, 7.0614, 6.953, 6.86574, 6.85068, 6.79981, 6.76557, 6.74009, 6.73121, 6.71234, 6.70114, 6.65549]}, "val_loss": {"start_step": 0, "end_step": 5, "step_interval": 1, "values": [8.07252, 7.2434, 6.90797, 6.81748, 6.74248]}, "train_step_timing_avg": 1.2255}
    """

    expected = json.loads(expected_json)

    def test_ci_mt5_170m_train_loss_deterministic(self):
        # Expected training loss curve at different global steps.
        expected = self.expected["reduced_train_loss"]
        expected_vals = expected["values"]
        train_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "reduced_train_loss")

        assert train_loss_list is not None, f"No TensorBoard events file was found in the logs."
        assert len(train_loss_list) == 100, f"The events file must have 10 training loss values, one per training iteration."
        for i, step in enumerate(range(expected["start_step"], expected["end_step"], expected["step_interval"])):
            assert train_loss_list[step] == expected_vals[i], f"The loss at step {step} should be {expected_vals[i]} but it is {train_loss_list[step]}."

    def test_ci_mt5_170m_train_loss_approx(self):
        # Expected training loss curve at different global steps.
        expected = self.expected["reduced_train_loss"]
        expected_vals = expected["values"]
        train_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "reduced_train_loss")

        assert train_loss_list is not None, f"No TensorBoard events file was found in the logs."
        assert len(train_loss_list) == 100, f"The events file must have 100 training loss values, one per training iteration."
        for i, step in enumerate(range(expected["start_step"], expected["end_step"], expected["step_interval"])):
            assert train_loss_list[step] == pytest.approx(expected=expected_vals[i], rel=self.margin_loss), f"The loss at step {step} should be approximately {expected_vals[i]} but it is {train_loss_list[step]}."

    def test_ci_mt5_170m_val_loss_deterministic(self):
        # Expected validation loss curve at different global steps.
        expected = self.expected["val_loss"]
        expected_vals = expected["values"]
        val_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "val_loss")

        assert val_loss_list is not None, f"No TensorBoard events file was found in the logs."
        assert len(val_loss_list) == 5, f"The events file must have 5 validation loss values."
        for i, step in enumerate(range(expected["start_step"], expected["end_step"], expected["step_interval"])):
            assert val_loss_list[step] == expected_vals[i], f"The loss at step {step} should be {expected_vals[i]} but it is {val_loss_list[step]}."

    def test_ci_mt5_170m_val_loss_approx(self):
        # Expected validation loss curve at different global steps.
        expected = self.expected["val_loss"]
        expected_vals = expected["values"]
        val_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "val_loss")

        assert val_loss_list is not None, f"No TensorBoard events file was found in the logs."
        assert len(val_loss_list) == 5, f"The events file must have 5 validation loss values."
        for i, step in enumerate(range(expected["start_step"], expected["end_step"], expected["step_interval"])):
            assert val_loss_list[step] == pytest.approx(expected=expected_vals[i], rel=self.margin_loss), f"The loss at step {step} should be approximately {expected_vals[i]} but it is {val_loss_list[step]}."

    def test_ci_mt5_170m_train_step_timing_1node(self):
        # Expected average training time per global step.
        expected_avg = self.expected["train_step_timing_avg"]
        train_time_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "train_step_timing")
        train_time_list = train_time_list[len(train_time_list)//2:] # Discard the first half.
        train_time_avg = sum(train_time_list) / len(train_time_list)

        assert train_time_list is not None, f"No TensorBoard events file was found in the logs."
        assert train_time_avg == pytest.approx(expected=expected_avg, rel=self.margin_time), f"The time per global step must be approximately {expected_avg} but it is {train_time_avg}."
