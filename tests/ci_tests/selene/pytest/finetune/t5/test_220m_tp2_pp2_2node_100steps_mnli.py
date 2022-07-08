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
    {"reduced_train_loss": {"start_step": 0, "end_step": 100, "step_interval": 5, "values": [9.51386, 6.59005, 4.94246, 4.11331, 3.41532, 2.5846, 1.69401, 1.21059, 0.97457, 0.93013, 0.82013, 0.80156, 0.71851, 0.7104, 0.68923, 0.6731, 0.62073, 0.60987, 0.60449, 0.5228]}, "val_loss": {"start_step": 0, "end_step": 5, "step_interval": 1, "values": [3.34024, 0.92748, 0.73866, 0.61423, 0.52844]}, "train_step_timing_avg": 0.13556099999999996}
    """

    expected = json.loads(expected_json)

    def test_ci_t5_220m_train_loss_deterministic(self):
        # Expected training loss curve at different global steps.
        expected = self.expected["reduced_train_loss"]
        expected_vals = expected["values"]
        train_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "reduced_train_loss")

        assert train_loss_list is not None, f"No TensorBoard events file was found in the logs."
        assert len(train_loss_list) == 100, f"The events file must have 10 training loss values, one per training iteration."
        for i, step in enumerate(range(expected["start_step"], expected["end_step"], expected["step_interval"])):
            assert train_loss_list[step] == expected_vals[i], f"The loss at step {step} should be {expected_vals[i]} but it is {train_loss_list[step]}."

    def test_ci_t5_220m_train_loss_approx(self):
        # Expected training loss curve at different global steps.
        expected = self.expected["reduced_train_loss"]
        expected_vals = expected["values"]
        train_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "reduced_train_loss")

        assert train_loss_list is not None, f"No TensorBoard events file was found in the logs."
        assert len(train_loss_list) == 100, f"The events file must have 100 training loss values, one per training iteration."
        for i, step in enumerate(range(expected["start_step"], expected["end_step"], expected["step_interval"])):
            assert train_loss_list[step] == pytest.approx(expected=expected_vals[i], rel=self.margin_loss), f"The loss at step {step} should be approximately {expected_vals[i]} but it is {train_loss_list[step]}."

    def test_ci_t5_220m_val_loss_deterministic(self):
        # Expected validation loss curve at different global steps.
        expected = self.expected["val_loss"]
        expected_vals = expected["values"]
        val_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "val_loss")

        assert val_loss_list is not None, f"No TensorBoard events file was found in the logs."
        assert len(val_loss_list) == 5, f"The events file must have 5 validation loss values."
        for i, step in enumerate(range(expected["start_step"], expected["end_step"], expected["step_interval"])):
            assert val_loss_list[step] == expected_vals[i], f"The loss at step {step} should be {expected_vals[i]} but it is {val_loss_list[step]}."

    def test_ci_t5_220m_val_loss_approx(self):
        # Expected validation loss curve at different global steps.
        expected = self.expected["val_loss"]
        expected_vals = expected["values"]
        val_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "val_loss")

        assert val_loss_list is not None, f"No TensorBoard events file was found in the logs."
        assert len(val_loss_list) == 5, f"The events file must have 5 validation loss values."
        for i, step in enumerate(range(expected["start_step"], expected["end_step"], expected["step_interval"])):
            assert val_loss_list[step] == pytest.approx(expected=expected_vals[i], rel=self.margin_loss), f"The loss at step {step} should be approximately {expected_vals[i]} but it is {val_loss_list[step]}."

    def test_ci_t5_220m_train_step_timing_1node(self):
        # Expected average training time per global step.
        expected_avg = self.expected["train_step_timing_avg"]
        train_time_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "train_step_timing")
        train_time_list = train_time_list[len(train_time_list)//2:] # Discard the first half.
        train_time_avg = sum(train_time_list) / len(train_time_list)

        assert train_time_list is not None, f"No TensorBoard events file was found in the logs."
        assert train_time_avg == pytest.approx(expected=expected_avg, rel=self.margin_time), f"The time per global step must be approximately {expected_avg} but it is {train_time_avg}."
