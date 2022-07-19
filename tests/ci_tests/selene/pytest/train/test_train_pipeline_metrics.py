import os
import json
import pytest
import enum
import sys
from tensorboard.backend.event_processing import event_accumulator

CI_JOB_RESULTS_DIR = os.environ.get("RESULTS_DIR") #eg '/home/shanmugamr/bignlp-scripts/results/train_gpt3_126m_tp1_pp1_1node_100steps'

#TODO : Should be importing this from CITestHelper
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

class TestType(enum.Enum):
    APPROX = 1
    DETERMINISTIC = 2

# If we require a subtle variation of tests for any of the other pipelines we can just inherit this class.
class TestTrainPipelineMetrics:

    margin_loss, margin_time = 0.05, 0.1
    job_name = CI_JOB_RESULTS_DIR.rsplit("/",1)[1]
    file_name = job_name + ".json" #eg train_gpt3_126m_tp1_pp1_1node_100steps.json
    file_directory =  file_name.split("_")[1] #eg gpt3
    expected_metrics_file = os.path.join("tests/ci_tests/selene/pytest/train", file_directory, file_name)
    with open(expected_metrics_file) as f:
        expected = json.load(f)

    def _test_loss_helper(self, loss_type, test_type):
        expected = self.expected[loss_type]
        expected_loss_list = expected["values"]
        actual_loss_list = _read_tb_logs_as_list(CI_JOB_RESULTS_DIR, loss_type)
        loss_list_size = len(expected_loss_list)*expected["step_interval"]
        assert actual_loss_list is not None, f"{self.job_name} : No TensorBoard events file was found in the logs for {loss_type}."
        assert len(actual_loss_list) == loss_list_size, f"{self.job_name} : The events file must have {loss_list_size} {loss_type} values, one per training iteration."
        for i, step in enumerate(range(expected["start_step"], expected["end_step"], expected["step_interval"])):
            if test_type == TestType.APPROX:
                assert actual_loss_list[step] == pytest.approx(expected=expected_loss_list[i], rel=self.margin_loss), f"{self.job_name} : The loss at step {step} should be approximately {expected_vals[i]} but it is {train_loss_list[step]}."
            else:
                assert actual_loss_list[step] == expected_loss_list[i], f"{self.job_name} : The loss at step {step} should be {expected_loss_list[i]} but it is {actual_loss_list[step]}."

    def test_train_loss_deterministic(self):
        # Expected training loss curve at different global steps.
        self._test_loss_helper("reduced_train_loss", TestType.DETERMINISTIC)

    def test__train_loss_approx(self):
        # Expected training loss curve at different global steps.
        self._test_loss_helper("reduced_train_loss", TestType.APPROX)

    def test_val_loss_deterministic(self):
        # Expected validation loss curve at different global steps.
        self._test_loss_helper("val_loss", TestType.DETERMINISTIC)

    def test_val_loss_approx(self):
        # Expected validation loss curve at different global steps.
        self._test_loss_helper("val_loss", TestType.APPROX)

    def test_train_step_timing_1node(self):
        # Expected average training time per global step.
        expected_avg = self.expected["train_step_timing_avg"]
        train_time_list = _read_tb_logs_as_list(CI_JOB_RESULTS_DIR, "train_step_timing")
        train_time_list = train_time_list[len(train_time_list)//2:] # Discard the first half.
        train_time_avg = sum(train_time_list) / len(train_time_list)

        assert train_time_list is not None, f"{self.job_name} : No TensorBoard events file was found in the logs for train_step_timing_avg"
        assert train_time_avg == pytest.approx(expected=expected_avg, rel=self.margin_time), f"{self.job_name} : The time per global step must be approximately {expected_avg} but it is {train_time_avg}."
