from tests.ci_tests.selene.pytest.common.loss_testing_pipeline import LossTestingPipeline
CI_JOB_RESULTS_DIR = os.environ.get("RESULTS_DIR")

class TestPromptEvalPipeline:
    margin = 0.05
    job_name = CI_JOB_RESULTS_DIR.rsplit("/",1)[1]
    file_name = job_name + ".json" #eg train_gpt3_126m_tp1_pp1_1node_100steps.json
    run_stage = file_name.split("_")[0]
    file_directory = file_name.split("_")[1] + "_result_files"
    expected_metrics_file = os.path.join("tests/ci_tests/selene/pytest/", run_stage, file_directory, file_name)
    expected = None
    if os.path.exists(expected_metrics_file):
        with open(expected_metrics_file) as f:
            expected = json.load(f)

    def test_eval_metric_approx(self):
        if self.expected is None:
            raise FileNotFoundError("Use `CREATE_TEST_DATA=True` to create baseline files.")

        with open(CI_JOB_RESULTS_DIR + "/squad_metric.json") as f:
            actual = json.load(f)
        
        for metric in ["exact_match", "f1"]:
            assert actual[metric] == pytest.approx(
                expected=self.expected[metric],
                rel=self.margin,
            ), f"{self.job_name} : The SQuAD {metric} should be approximately {self.expected[metric]} but it is {actual[metric]}."
