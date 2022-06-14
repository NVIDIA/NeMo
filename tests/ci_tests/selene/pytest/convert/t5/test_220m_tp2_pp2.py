import os


CI_JOB_RESULTS = os.environ.get("RESULTS_DIR")

class TestCIT5_220m:

    def test_ci_convert_t5_220m_tp1_pp1(self):
        ckpt_path = os.path.join(CI_JOB_RESULTS, "megatron_t5.nemo")
        assert os.path.exists(ckpt_path), f"File not found: {ckpt_path}"
