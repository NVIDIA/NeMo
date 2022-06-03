import os


BASE_RESULTS_DIR = os.environ.get("RESULTS_DIR")

class TestCIGPT126m:

    def test_ci_convert_gpt3_126m_tp1_pp1(self):
        ckpt_path = os.path.join(BASE_RESULTS_DIR, "data/00.jsonl")
        assert os.path.exists(ckpt_path), f"File not found: {ckpt_path}"
