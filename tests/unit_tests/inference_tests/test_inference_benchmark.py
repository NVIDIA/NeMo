import filecmp
from os.path import dirname, abspath

from bignlp.inference.benchmark import run_bechmark

MOCK_CLUSTER_CFG = {
    "partition": "luna",
    "account": "joc",
    "time_limit": "0:30:00",
    "exclusive": True,
}

BASE_DIR = dirname(dirname(dirname(abspath(__file__))))


class TestGpt3InferenceBenchmark:
    def test_gpt3_inference_benchmark(self):
        expected_output_file = BASE_DIR + "/tests/unit_tests/inference_tests/inference_benchmark_gpt3_530b_tp8_pp3.sh"
        sbatch_script_path = run_benchmark(
                                model_type="gpt3",
                                model_size="530b",
                                bignlp_scripts_path=BASE_DIR,
                                container="gitlab-master.nvidia.com#dl/dgx/bignlp/infer:infer_update-py3-base"
                                tensor_para_size=8,
                                pipeline_para_size=3,
                                input_len=60,
                                output_len=20,
                                batch_sizes=[1,2,4,8,16,32,64,128,256],
                                triton_wait_time=30,
                                cluster_cfg=MOCK_CLUSTER_CFG)

        assert filecmp.cmp(sbatch_script_path, expected_output_file)
            

class TestT5InferenceBenchmark:
    def test_t5_inference_benchmark(self):
        expected_output_file = BASE_DIR + "/tests/unit_tests/inference_tests/inference_benchmark_t5_41b_tp8_pp1.sh"
        sbatch_script_path = run_benchmark(
                                model_type="t5",
                                model_size="41b",
                                bignlp_scripts_path=BASE_DIR,
                                container="gitlab-master.nvidia.com#dl/dgx/bignlp/infer:infer_update-py3-base"
                                tensor_para_size=8,
                                pipeline_para_size=1,
                                input_len=60,
                                output_len=20,
                                batch_sizes=[1,2,4,8,16,32,64,128,256],
                                triton_wait_time=30,
                                cluster_cfg=MOCK_CLUSTER_CFG)

        assert filecmp.cmp(sbatch_script_path, expected_output_file)


class TestMt5InferenceBenchmark:
    def test_mt5_inference_benchmark(self):
        expected_output_file = BASE_DIR + "/tests/unit_tests/inference_tests/inference_benchmark_mt5_23b_tp8_pp1.sh"
        sbatch_script_path = run_benchmark(
                                model_type="mt5",
                                model_size="23b",
                                bignlp_scripts_path=BASE_DIR,
                                container="gitlab-master.nvidia.com#dl/dgx/bignlp/infer:infer_update-py3-base"
                                tensor_para_size=8,
                                pipeline_para_size=1,
                                input_len=60,
                                output_len=20,
                                batch_sizes=[1,2,4,8,16,32,64,128,256],
                                triton_wait_time=30,
                                cluster_cfg=MOCK_CLUSTER_CFG)

        assert filecmp.cmp(sbatch_script_path, expected_output_file)
