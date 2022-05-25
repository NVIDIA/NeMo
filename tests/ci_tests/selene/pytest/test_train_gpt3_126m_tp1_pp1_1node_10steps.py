import os

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


class TestCIGPT126m:

    margin = 0.05

    def test_ci_gpt3_126m_train_loss_deterministic(self):
        # Expected training loss curve at different global steps.
        expected = [
            10.9099, 10.88668, 10.9028, 10.90496, 10.76744, 
            10.46561, 10.33317, 9.9591, 9.98051, 9.61251, 
        ]
        train_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "reduced_train_loss")

        assert train_loss_list is not None, f"No TensorBoard events file was found in the logs."
        assert len(train_loss_list) == 10, f"The events file must have 10 training loss values, one per training iteration."
        for step in range(0, 10):
            assert train_loss_list[step] == expected[step], f"The loss at step {step} should be {expected[step]} but it is {train_loss_list[step]}."

    def test_ci_gpt3_126m_train_loss_approx(self):
        # Expected training loss curve at different global steps.
        expected = [
            10.9099, 10.88668, 10.9028, 10.90496, 10.76744, 
            10.46561, 10.33317, 9.9591, 9.98051, 9.61251, 
        ]
        train_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "reduced_train_loss")

        assert train_loss_list is not None, f"No TensorBoard events file was found in the logs."
        assert len(train_loss_list) == 10, f"The events file must have 10 training loss values, one per training iteration."
        for step in range(0, 10):
            assert train_loss_list[step] == pytest.approx(expected=expected[step], rel=self.margin), f"The loss at step {step} should be approximately {expected[step]} but it is {train_loss_list[step]}."

    def test_ci_gpt3_126m_val_loss_deterministic(self):
        # Expected validation loss curve at different global steps.
        expected = [10.78457, 10.58221, 9.97985, 9.31694, 9.0641]
        val_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "val_loss")

        assert val_loss_list is not None, f"No TensorBoard events file was found in the logs."
        assert len(val_loss_list) == 5, f"The events file must have 5 validation loss values."
        for step in range(0, 5):
            assert val_loss_list[step] == expected[step], f"The loss at step {step} should be {expected[step]} but it is {val_loss_list[step]}."

    def test_ci_gpt3_126m_val_loss_approx(self):
        # Expected validation loss curve at different global steps.
        expected = [10.78457, 10.58221, 9.97985, 9.31694, 9.0641]
        val_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "val_loss")

        assert val_loss_list is not None, f"No TensorBoard events file was found in the logs."
        assert len(val_loss_list) == 5, f"The events file must have 5 validation loss values."
        for step in range(0, 5):
            assert val_loss_list[step] == pytest.approx(expected=expected[step], rel=self.margin), f"The loss at step {step} should be approximately {expected[step]} but it is {val_loss_list[step]}."

    def test_ci_gpt3_126m_train_step_timing_1node(self):
        # Expected average training time per global step.
        expected_avg = 0.89
        train_time_list = _read_tb_logs_as_list(CI_JOB_RESULTS, "train_step_timing")
        train_time_list = train_time_list[5:] # Discard first 5 steps until time stabilizes.
        train_time_avg = sum(train_time_list) / len(train_time_list)

        assert train_time_list is not None, f"No TensorBoard events file was found in the logs."
        assert train_time_avg == pytest.approx(expected=expected_avg, rel=self.margin), f"The time per global step must be approximately {expected_avg} but it is {train_time_avg}."
