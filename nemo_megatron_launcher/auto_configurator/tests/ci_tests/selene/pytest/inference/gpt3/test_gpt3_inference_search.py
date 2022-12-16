import os


BASE_RESULTS_DIR = os.environ.get("BASE_RESULTS_DIR")
RUN_SIZE = os.environ.get("RUN_SIZE")
GPU_MEM = os.environ.get("GPU_MEM")

class TestCIGPT:

    def test_ci_train_gpt3(self):
        inference_dir = os.path.join(BASE_RESULTS_DIR, f"gpt3/{RUN_SIZE}_{GPU_MEM}gb/inference")
        final_summary_dir = os.path.join(BASE_RESULTS_DIR, f"gpt3/{RUN_SIZE}_{GPU_MEM}gb/inference/final_summary")
        final_output_csv = os.path.join(BASE_RESULTS_DIR, f"gpt3/{RUN_SIZE}_{GPU_MEM}gb/inference/final_summary/final_output.csv")

        assert os.path.exists(inference_dir), f"Dir not found: {inference_dir}"
        assert os.path.exists(final_summary_dir), f"Dir not found: {final_summary_dir}"
        assert os.path.exists(final_output_csv), f"File not found: {final_output_csv}"
