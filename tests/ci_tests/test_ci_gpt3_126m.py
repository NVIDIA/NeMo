import os

import pytest

from tensorboard.backend.event_processing import event_accumulator


CI_RESULTS_DIR = "/lustre/fsw/joc/big_nlp/bignlp_ci/results"

class TestCIGPT126m:

    margin = 0.05

    def test_ci_gpt3_126m_train_loss_deterministic(self):
        # Expected training loss curve at different global steps.
        expected = [10.9099, 10.88668, 10.9028, 10.90496, 10.76744, 10.46561, 10.33317, 9.9591, 
                    9.98051, 9.61251, 9.62183, 9.51763, 9.41488, 9.38017, 9.38307, 9.33679, 
                    9.2004, 9.14779, 9.00378, 8.88322, 8.89572, 9.05839, 8.86122, 8.80876, 
                    9.03917, 8.95654, 8.77681, 8.91803, 8.61961, 8.75352, 8.70215, 8.84263, 
                    8.76698, 8.8242, 8.45948, 8.76986, 8.56294, 8.4557, 8.58012, 8.38864, 
                    8.38296, 8.42465, 7.9568, 8.35038, 8.24879, 8.22272, 8.11146, 8.13253, 
                    8.22866, 8.09874
        ]

        results_dir = f"{CI_RESULTS_DIR}/ci_gpt3_126m_deterministic"
        files = os.listdir(results_dir)

        train_loss = None
        for f in files:
            if f[:6] == "events":
                event_file = os.path.join(results_dir, f)
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                train_loss = ea.Scalars("reduced_train_loss")
                train_loss_vals = [round(x.value, 5) for x in train_loss]
                print(train_loss_vals)
                break

        assert train_loss is not None, f"No TensorBoard events file was found in the logs."
        assert len(train_loss_vals) == 50, f"The events file must have 50 training loss values, one per training iteration."

        for step in range(0, 50):
            assert expected[step] == train_loss_vals[step], f"The loss at step {step} should be {expected[step]} but it is {train_loss_vals[step]}."

    def test_ci_gpt3_126m_val_loss_deterministic(self):
        # Expected validation loss curve at different global steps.
        expected = [9.0641, 8.51007, 8.26597, 7.97282, 7.65916]

        results_dir = f"{CI_RESULTS_DIR}/ci_gpt3_126m_deterministic"
        files = os.listdir(results_dir)

        val_loss = None
        for f in files:
            if f[:6] == "events":
                event_file = os.path.join(results_dir, f)
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                val_loss = ea.Scalars("val_loss")
                val_loss_vals = [round(x.value, 5) for x in val_loss]
                print(val_loss_vals)
                break

        assert val_loss is not None, f"No TensorBoard events file was found in the logs."
        assert len(val_loss_vals) == 5, f"The events file must have 5 validation loss values."

        for step in range(0, 5):
            assert expected[step] == val_loss_vals[step], f"The loss at step {step} should be {expected[step]} but it is {val_loss_vals[step]}."

    def test_ci_gpt3_126m_train_step_timing_1node(self):
        # Expected average training time per global step.
        expected_avg = 0.89

        results_dir = f"{CI_RESULTS_DIR}/ci_gpt3_126m_deterministic"
        files = os.listdir(results_dir)

        train_time = None
        for f in files:
            if f[:6] == "events":
                event_file = os.path.join(results_dir, f)
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                train_time = ea.Scalars("train_step_timing")
                #train_time = train_time[6:]
                train_time_list = [round(x.value, 5) for x in train_time][6:]
                train_time_avg = sum(train_time_list) / len(train_time_list)
                print(train_time_list)
                print(train_time_avg)
                break

        assert train_time is not None, f"No TensorBoard events file was found in the logs."
        assert train_time_avg == pytest.approx(expected=expected_avg, rel=self.margin), f"The time per global step must be approximately {expected_avg} but it is {train_time_avg}."
