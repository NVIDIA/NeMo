import os


CI_JOB_RESULTS = os.environ.get("RESULTS_DIR")

class BigNLPCITest:

    def test_ci_convert_mt5_170m_tp1_pp1(self):
        ckpt_path = os.path.join(CI_JOB_RESULTS, "megatron_mt5.nemo")
        assert os.path.exists(ckpt_path), f"File not found: {ckpt_path}"
