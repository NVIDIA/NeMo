import argparse

import numpy as np
from pytriton.client import ModelClient

from nemo.deploy.deploy_pytriton import DeployPyTriton
from nemo.deploy.nlp import NemoTritonQueryLLMTensorRT
from nemo.deploy.nlp.megatrongpt_deployable import MegatronGPTDeployable


def test_triton_deployable(args):
    megatron_deployable = MegatronGPTDeployable(args.nemo_checkpoint)

    prompts = ["What is the biggest planet in the solar system?", "What is the fastest steam locomotive in history?"]
    url = "localhost:8000"
    model_name = args.model_name
    init_timeout = 600.0

    nm = DeployPyTriton(
        model=megatron_deployable,
        triton_model_name=model_name,
        triton_model_version=1,
        max_batch_size=8,
        port=8000,
        address="0.0.0.0",
        streaming=False,
    )
    nm.deploy()
    nm.run()

    # NemoQueryLLM seems specific to TRTLLM for now, so using ModelClient instead
    str_ndarray = np.array(prompts)[..., np.newaxis]
    prompts = np.char.encode(str_ndarray, "utf-8")
    max_output_token = np.full(prompts.shape, args.max_output_token, dtype=np.int_)
    top_k = np.full(prompts.shape, args.top_k, dtype=np.int_)
    top_p = np.full(prompts.shape, args.top_p, dtype=np.single)
    temperature = np.full(prompts.shape, args.temperature, dtype=np.single)

    with ModelClient(url, model_name, init_timeout_s=init_timeout) as client:
        result_dict = client.infer_batch(
            prompts=prompts,
            max_length=max_output_token,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

    print(result_dict)
    nm.stop()


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Deploy nemo models to Triton and benchmark the models",
    )

    parser.add_argument(
        "--model_name", type=str, required=True,
    )
    # parser.add_argument(
    #     "--existing_test_models", default=False, action='store_true',
    # )
    # parser.add_argument(
    #     "--model_type", type=str, required=False,
    # )
    # parser.add_argument(
    #     "--min_gpus", type=int, default=1, required=True,
    # )
    # parser.add_argument(
    #     "--max_gpus", type=int,
    # )
    # parser.add_argument(
    #     "--checkpoint_dir", type=str, default="/tmp/nemo_checkpoint/", required=False,
    # )
    parser.add_argument(
        "--nemo_checkpoint", type=str, required=True,
    )
    # parser.add_argument(
    #     "--trt_llm_model_dir", type=str,
    # )
    parser.add_argument(
        "--max_batch_size", type=int, default=8,
    )
    # parser.add_argument(
    #     "--max_input_token", type=int, default=256,
    # )
    parser.add_argument(
        "--max_output_token", type=int, default=128,
    )
    # parser.add_argument(
    #     "--p_tuning_checkpoint", type=str,
    # )
    # parser.add_argument(
    #     "--ptuning", default=False, action='store_true',
    # )
    # parser.add_argument(
    #     "--lora_checkpoint", type=str,
    # )
    # parser.add_argument(
    #     "--lora", default=False, action='store_true',
    # )
    # parser.add_argument(
    #     "--tp_size", type=int,
    # )
    # parser.add_argument(
    #     "--pp_size", type=int,
    # )
    parser.add_argument(
        "--top_k", type=int, default=1,
    )
    parser.add_argument(
        "--top_p", type=float, default=0.0,
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
    )
    # parser.add_argument(
    #     "--run_accuracy", default=False, action='store_true',
    # )
    # parser.add_argument("--streaming", default=False, action="store_true")
    # parser.add_argument(
    #     "--test_deployment", type=str, default="False",
    # )
    # parser.add_argument(
    #     "--debug", default=False, action='store_true',
    # )
    # parser.add_argument(
    #     "--ci_upload_test_results_to_cloud", default=False, action='store_true',
    # )
    # parser.add_argument(
    #     "--test_data_path", type=str, default=None,
    # )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    test_triton_deployable(args)
