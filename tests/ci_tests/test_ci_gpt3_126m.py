import os

from tensorboard.backend.event_processing import event_accumulator


CI_RESULTS_DIR = "/workspace/bignlp-scripts/results"

class TestCIGPT126m:
    def test_ci_gpt3_126m_train_loss_deterministic(self):
        # Expected training loss curve at different global steps.
        expected = {0: 10.881, 5: 10.412, 10: 9.475, 15: 9.318, 20: 9.096, 25: 8.757, 30: 8747, 35: 8.675, 40: 8.431, 45: 8.171}

        results_dir = f"{CI_RESULTS_DIR}/ci_gpt3_126m_deterministic"
        files = os.listdir(results_dir)

        train_loss = None
        for f in files:
            if f[:6] == "events":
                event_file = os.path.join(results_dir, f)
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                train_loss = ea.Scalars("reduced_train_loss")
                break

        assert train_loss is not None, f"No TensorBoard events file was found in the logs."
        assert len(train_loss) == 50, f"The events file must have 50 values, one per training iteration."

        for step in range(0, 50, 5):
            assert expected[step] == train_loss[step], f"The loss at step {step} should be {expected[step]} but it is {train_loss[step]}."
