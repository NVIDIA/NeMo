import argparse

import numpy as np
from pytriton.client import ModelClient

from nemo.deploy.deploy_pytriton import DeployPyTriton
from nemo.deploy.multimodal.megatronneva_deployable import MediaType, MegatronNevaDeployable

# from nemo.deploy.nlp.query_llm import NemoTritonQueryLLMPyTorch


def test_triton_deployable(args):
    megatron_deployable = MegatronNevaDeployable(args.nemo_checkpoint, args.num_gpus)

    image_files = ["/workspace/multimodal/4090.jpg", "/workspace/multimodal/jhh.jpeg"]
    image_bytes = []
    for file in image_files:
        with open(file, "rb") as f:
            image_bytes.append(f.read())

    prompts = ["What is in this picture?\n<image>", "What is the name of the person in this picture?\n<image>"]

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

    # run once with NemoTritonQueryLLMPyTorch
    # nemo_triton_query = NemoTritonQueryLLMPyTorch(url, model_name)

    # result_dict = nemo_triton_query.query_llm(
    #     prompts,
    #     top_k=args.top_k,
    #     top_p=args.top_p,
    #     temperature=args.temperature,
    #     max_length=args.max_output_token,
    #     init_timeout=init_timeout,
    # )
    # print("NemoTritonQueryLLMPyTriton result:")
    # print(result_dict)

    # run once with ModelClient, the results should be identical
    str_ndarray = np.array(prompts)[..., np.newaxis]
    prompts = np.char.encode(str_ndarray, "utf-8")
    max_output_token = np.full(prompts.shape, args.max_output_token, dtype=np.int_)
    top_k = np.full(prompts.shape, args.top_k, dtype=np.int_)
    top_p = np.full(prompts.shape, args.top_p, dtype=np.single)
    temperature = np.full(prompts.shape, args.temperature, dtype=np.single)
    # note: MegatronNevaModel only reads the first array value for media_type and uses that for the whole batch
    # only allowed media types are "image" and "video" which are both 5 characters long
    media_type = np.full(prompts.shape, MediaType.IMAGE, dtype=np.int_)
    media = np.array(image_bytes, dtype=bytes)[..., np.newaxis]

    with ModelClient(url, model_name, init_timeout_s=init_timeout) as client:
        result_dict = client.infer_batch(
            prompts=prompts,
            max_length=max_output_token,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            media_type=media_type,
            media_list=media,
        )
        print("ModelClient result:")
        print(result_dict)

    # test logprobs generation
    # right now we don't support batches where output data is inconsistent in size, so submitting each prompt individually
    all_probs = np.full(prompts.shape, True, dtype=np.bool_)
    compute_logprob = np.full(prompts.shape, True, dtype=np.bool_)
    with ModelClient(url, model_name, init_timeout_s=init_timeout) as client:
        logprob_results = client.infer_batch(
            prompts=prompts,
            max_length=max_output_token,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            media_type=media_type,
            media_list=media,
            all_probs=all_probs,
            compute_logprob=compute_logprob,
        )
        print("Logprob results:")
        print(logprob_results)

    nm.stop()


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Deploy nemo models to Triton and benchmark the models",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--nemo_checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max_output_token",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    # parser.add_argument(
    #     "--media_type",
    #     type=str,
    #     default="image"
    # )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    test_triton_deployable(args)
