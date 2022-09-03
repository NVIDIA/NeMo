import os


BASE_RESULTS_DIR = os.environ.get("BASE_RESULTS_DIR")

class TestCIGPT5b:

    def test_ci_train_gpt3_5b_80gb_3runs(self):
        base_cfg = os.path.join(BASE_RESULTS_DIR, "gpt3/5b_80gb/base_cfg_5b.yaml")
        candidate_configs_dir = os.path.join(BASE_RESULTS_DIR, "gpt3/5b_80gb/candidate_configs")
        training_logs_dir = os.path.join(BASE_RESULTS_DIR, "gpt3/5b_80gb/training_logs")
        final_result_dir = os.path.join(BASE_RESULTS_DIR, "gpt3/5b_80gb/final_result")
        final_summary_csv = os.path.join(BASE_RESULTS_DIR, "gpt3/5b_80gb/final_result/final_summary_16nodes.csv")
        optimal_cfg = os.path.join(BASE_RESULTS_DIR, "gpt3/5b_80gb/final_result/optimal_config_5b_16nodes.yaml")

        assert os.path.exists(base_cfg), f"File not found: {base_cfg}"
        assert os.path.exists(candidate_configs_dir), f"Dir not found: {candidate_configs_dir}"
        assert os.path.exists(training_logs_dir), f"Dir not found: {training_logs_dir}"
        assert os.path.exists(final_result_dir), f"Dir not found: {final_result_dir}"
        assert os.path.exists(final_summary_csv), f"File not found: {final_summary_csv}"
        assert os.path.exists(optimal_cfg), f"File not found: {optimal_cfg}"
