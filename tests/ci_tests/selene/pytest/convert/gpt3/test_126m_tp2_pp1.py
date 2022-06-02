import os


CI_JOB_RESULTS = os.environ.get("RESULTS_DIR")

class TestCIGPT126m:

    def test_ci_convert_gpt3_126m_tp2_pp1(self):
        ckpt_path = os.path.join(CI_JOB_RESULTS, "megatron_gpt.nemo")
        assert os.path.exists(ckpt_path), f"File not found: {ckpt_path}"
