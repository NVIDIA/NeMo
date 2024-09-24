# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import os
import shutil
import json
import time
from typing import List, TypedDict
import subprocess

from nemo.deploy import DeployPyTriton
from nemo.export.tensorrt_llm import TensorRTLLM
from nemo.deploy.nlp import NemoQueryLLM


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--run_number", type=int, help="Number of config to run")
    # parser.add_argument("--logs_dir", type=str, help="Path where to save training logs")
    # parser.add_argument("--data_path", type=str, help="Path to the dataset")
    # parser.add_argument("--get_results", action="store_true")
    
    # arguments that will be constant across autoconfigurator runs
    parser.add_argument("-tdp", "--test_data_path", type=str, required=True, help="Path to json-format test data")
    parser.add_argument("-nc", "--nemo_checkpoint", type=str, required=True, help=".nemo checkpoint file to test")
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        required=False,
        choices=["gptnext", "gpt", "llama", "falcon", "starcoder", "mixtral", "gemma"],
        help="Type of the model. gptnext, gpt, llama, falcon, and starcoder are only supported."
        " gptnext and gpt are the same and keeping it for backward compatibility",
    )

    # arguments that we are varying across autoconfigurator runs to find the best values
    parser.add_argument("-ng", "--num_gpus", nargs='+', default=1, type=int, help="Number of GPUs for the deployment")
    parser.add_argument("-tps", "--tensor_parallelism_size", nargs='+', default=1, type=int, help="Tensor parallelism size")
    parser.add_argument("-pps", "--pipeline_parallelism_size", nargs='+', default=1, type=int, help="Pipeline parallelism size")
    parser.add_argument(
        "-dt",
        "--dtype",
        nargs='+',
        choices=["bfloat16", "float16", "fp8", "int8"],
        default="bfloat16",
        type=str,
        help="dtype of the model on TensorRT-LLM",
    )
    parser.add_argument("-mil", "--max_input_len", nargs='+', default=256, type=int, help="Max input length of the model")
    parser.add_argument("-mol", "--max_output_len", nargs='+', default=256, type=int, help="Max output length of the model")
    parser.add_argument("-mbs", "--max_batch_size", nargs='+', default=8, type=int, help="Max batch size of the model")
    parser.add_argument("-mnt", "--max_num_tokens", nargs='+', default=None, type=int, help="Max number of tokens")
    parser.add_argument("-ont", "--opt_num_tokens", nargs='+', default=None, type=int, help="Optimum number of tokens")
    parser.add_argument(
        "-mpet", "--max_prompt_embedding_table_size", nargs='+', default=None, type=int, help="Max prompt embedding table size"
    )
    parser.add_argument(
        "-lc",
        "--lora_ckpt",
        nargs='+',
        action='append',
        type=str,
        required=False,
        help="List of lora checkpoints to use; define multiple times to test different sets of lora checkpoints"
    )
    parser.add_argument(
        '--use_lora_plugin',
        nargs='+',
        const=None,
        choices=['float16', 'float32', 'bfloat16'],
        help="Activates the lora plugin which enables embedding sharing.",
    )
    parser.add_argument(
        '--lora_target_modules',
        nargs='+',
        action='append',
        default=None,
        choices=[
            "attn_qkv",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_dense",
            "mlp_h_to_4h",
            "mlp_gate",
            "mlp_4h_to_h",
        ],
        help="Add lora in which modules. Only be activated when use_lora_plugin is enabled. Define multiple times to test different combinations.",
    )
    parser.add_argument(
        '--max_lora_rank',
        nargs='+',
        type=int,
        default=64,
        help='maximum lora rank for different lora modules. '
        'It is used to compute the workspace size of lora plugin.',
    )

    # true/false options that can either be constant for all runs, or toggled and tested by autoconfigurator
    cpp_runtime_group = parser.add_mutually_exclusive_group()
    cpp_runtime_group.add_argument(
        "-ucr",
        '--use_cpp_runtime',
        default=False,
        action='store_true',
        help='Use TensorRT LLM C++ runtime',
    )
    cpp_runtime_group.add_argument(
        "-tcr",
        '--test_cpp_runtime',
        default=False,
        action='store_true',
        help='Tests toggling cpp runtime vs python runtime',
    )

    paged_kv_cache_group = parser.add_mutually_exclusive_group()
    paged_kv_cache_group.add_argument(
        "-npkc", "--no_paged_kv_cache", default=False, action='store_true', help="Disable paged kv cache"
    )
    paged_kv_cache_group.add_argument(
        "-tpkc", "--test_paged_kv_cache", default=False, action='store_true', help="Tests toggling the paged kv cache"
    )

    remove_input_padding_group = parser.add_mutually_exclusive_group()
    remove_input_padding_group.add_argument(
        "-drip",
        "--disable_remove_input_padding",
        default=False,
        action='store_true',
        help="Disables the remove input padding option.",
    )
    remove_input_padding_group.add_argument(
        "-trip",
        "--test_remove_input_padding",
        default=False,
        action='store_true',
        help="Tests toggling removing input padding",
    )

    parallel_embedding_group = parser.add_mutually_exclusive_group()
    parallel_embedding_group.add_argument(
        "-upe",
        "--use_parallel_embedding",
        default=False,
        action='store_true',
        help='Use parallel embedding feature of TensorRT-LLM.',
    )
    parallel_embedding_group.add_argument(
        "-tpe",
        "--test_parallel_embedding",
        default=False,
        action='store_true',
        help='Tests toggling parallel embedding',
    )

    multi_block_group = parser.add_mutually_exclusive_group()
    multi_block_group.add_argument(
        "-mbm",
        '--multi_block_mode',
        default=False,
        action='store_true',
        help='Split long kv sequence into multiple blocks (applied to generation MHA kernels). \
                        It is beneifical when batchxnum_heads cannot fully utilize GPU.',
    )
    multi_block_group.add_argument(
        "-tmbm",
        '--test_multi_block_mode',
        default=False,
        action='store_true',
        help='Test toggling multi block mode'
    )

    return parser.parse_args()

class AutoConfiguratorOptions(TypedDict):
    test_data_path: str
    nemo_checkpoint: str
    model_type: str
    num_gpus_list: List[int]
    tensor_parallelism_size_list: List[int]
    pipeline_parallelism_size_list: List[int]
    dtype_list: List[str]
    max_input_len_list: List[int]
    max_output_len_list: List[int]
    max_batch_size_list: List[int]
    max_num_tokens_list: List[int]
    opt_num_tokens_list: List[int]
    max_prompt_embedding_table_size_list: List[int]
    lora_ckpt_list: List[List[str]]
    use_lora_plugin_list: List[str]
    lora_target_modules_list: List[List[str]]
    max_lora_rank_list: List[int]
    use_cpp_runtime_list: List[bool]
    paged_kv_cache_list: List[bool]
    remove_input_padding_list: List[bool]
    use_parallel_embedding_list: List[bool]
    multi_block_mode_list: List[bool]


class AutoConfiguratorCombination(TypedDict):
    #nemo_checkpoint: str
    #model_type: str
    num_gpus: int
    tensor_parallelism_size: int
    pipeline_parallelism_size: int
    dtype: str
    max_input_len: int
    max_output_len: int
    max_batch_size: int
    max_num_tokens: int
    opt_num_tokens: int
    max_prompt_embedding_table_size: int
    lora_ckpt: List[str]
    use_lora_plugin: str
    lora_target_modules: List[str]
    max_lora_rank: int
    use_cpp_runtime: bool
    paged_kv_cache: bool
    remove_input_padding: bool
    use_parallel_embedding: bool
    multi_block_mode: bool


def generate_autoconfig_options() -> AutoConfiguratorOptions:
    args = get_args()
    parse_1d_list = lambda x, default=[None] : x if isinstance(x, list) else [x] if x is not None else default
    parse_2d_list = lambda x, default=[None] : x if isinstance(x, list) and isinstance(x[0], list) else [x] if isinstance(x, list) else [[x]] if x is not None else default
    parse_test_bool = lambda test, use: [True, False] if test else [use]
    options: AutoConfiguratorOptions = {
        'test_data_path': args.test_data_path,
        'nemo_checkpoint': args.nemo_checkpoint,
        'model_type': args.model_type,
        'num_gpus_list': parse_1d_list(args.num_gpus),
        'tensor_parallelism_size_list': parse_1d_list(args.tensor_parallelism_size),
        'pipeline_parallelism_size_list': parse_1d_list(args.pipeline_parallelism_size),
        'dtype_list': parse_1d_list(args.dtype),
        'max_input_len_list': parse_1d_list(args.max_input_len),
        'max_output_len_list': parse_1d_list(args.max_output_len),
        'max_batch_size_list': parse_1d_list(args.max_batch_size),
        'max_num_tokens_list': parse_1d_list(args.max_num_tokens),
        'opt_num_tokens_list': parse_1d_list(args.opt_num_tokens),
        'max_prompt_embedding_table_size_list': parse_1d_list(args.max_prompt_embedding_table_size),
        'lora_ckpt_list': parse_2d_list(args.lora_ckpt),
        'use_lora_plugin_list': parse_1d_list(args.use_lora_plugin),
        'lora_target_modules_list': parse_1d_list(args.lora_target_modules),
        'max_lora_rank_list': parse_1d_list(args.max_lora_rank),
        'use_cpp_runtime_list': parse_test_bool(args.test_cpp_runtime, args.use_cpp_runtime),
        'paged_kv_cache_list': parse_test_bool(args.test_paged_kv_cache, not args.no_paged_kv_cache),
        'remove_input_padding_list': parse_test_bool(args.test_remove_input_padding, not args.disable_remove_input_padding),
        'use_parallel_embedding_list': parse_test_bool(args.test_parallel_embedding, args.use_parallel_embedding),
        'multi_block_mode_list': parse_test_bool(args.test_multi_block_mode, args.multi_block_mode),
    }
    return options

def generate_autoconfig_combinations(options: AutoConfiguratorOptions) -> List[AutoConfiguratorCombination]:
    # create initial empty combination
    combinations = [{}]
    for key, value in options.items():
        if key.endswith('_list'):
            combination_key = key.removesuffix('_list')
            # we must generate combinations for the list
            combination_list = []
            for combination_value in value:
                combination_sublist = copy.deepcopy(combinations)
                for entry in combination_sublist:
                    entry[combination_key] = combination_value
                combination_list.extend(combination_sublist)
            combinations = combination_list
    return combinations

def autoconfig_run(options: AutoConfiguratorOptions):
    trt_llm_path = "/tmp/trt_llm_model_dir/"
    model_name = "autoconfigurator_model"
    autoconfig_combinations = generate_autoconfig_combinations(options)
    throughput_results = []

    min_input_length = min(options["max_input_len_list"])

    for combination in autoconfig_combinations:
        trt_llm_exporter = TensorRTLLM(
            model_dir=trt_llm_path,
            lora_ckpt_list=combination['lora_ckpt'],
            load_model=False,
            use_python_runtime=(not combination['use_cpp_runtime']),
            multi_block_mode=combination['multi_block_mode']
        )
        trt_llm_exporter.export(
            nemo_checkpoint_path=options['nemo_checkpoint'],
            model_type=options['model_type'],
            n_gpus=combination['num_gpus'],
            tensor_parallelism_size=combination['tensor_parallelism_size'],
            pipeline_parallelism_size=combination['pipeline_parallelism_size'],
            max_input_len=combination['max_input_len'],
            max_output_len=combination['max_output_len'],
            max_batch_size=combination['max_batch_size'],
            max_num_tokens=combination['max_num_tokens'],
            opt_num_tokens=combination['opt_num_tokens'],
            use_parallel_embedding=combination['use_parallel_embedding'],
            max_prompt_embedding_table_size=combination['max_prompt_embedding_table_size'],
            paged_kv_cache=combination['paged_kv_cache'],
            remove_input_padding=combination['remove_input_padding'],
            dtype=combination['dtype'],
            use_lora_plugin=combination['use_lora_plugin'],
            lora_target_modules=combination['lora_target_modules'],
            max_lora_rank=combination['max_lora_rank'],
        )
        nm = DeployPyTriton(
            model=trt_llm_exporter,
            triton_model_name=model_name
        )
        nm.deploy()
        nm.run()

        # testing using genai-perf
        artifact_dir = "/tmp/autoconfig/artifacts"
        subprocess.run(
            [
            "genai-perf", "profile",
            "-m", "autoconfigurator_model",
            "--service-kind", "triton",
            "--backend", "tensorrtllm",
            "--num-prompts", "100",
            "--synthetic-input-tokens-mean", f"{min_input_length}",
            "--artifact-dir", artifact_dir
            ],
            capture_output=True)
        # read the output json file
        gap_output_path = artifact_dir + "/profile_export_genai_perf.json"
        with open(gap_output_path) as gap_output_file:
            gap_output_json = json.load(gap_output_file)
            # TODO: change evaluation_time key since now we are measuring throughput rather than time taken
            inference_result = gap_output_json["request_throughput"]["avg"]
            throughput_results.append(inference_result)

        nm.stop()
    
    # at this point, we have finished testing for all combinations
    # sort combinations by ascending time taken (fastest first)
    sorted_results = sorted(zip(throughput_results, autoconfig_combinations), key=lambda x: x[0], reverse=True)
    # get list of parameters that were actually varying between combinations
    varying_parameters = []
    for key, value in options.items():
        if isinstance(value, list) and len(value) > 1:
            # remove the trailing "_list" to get the original parameter name
            varying_parameters.append(key[:-5])
    # print best ones
    for index, result_entry in enumerate(sorted_results[:4]):
        print(f"#{index+1} configuration, {result_entry[0]:.5f} requests/second:")
        print({key: result_entry[1][key] for key in varying_parameters})

def main():
    autoconfig_options: AutoConfiguratorOptions = generate_autoconfig_options()
    autoconfig_run(autoconfig_options)

if __name__ == '__main__':
    main()
