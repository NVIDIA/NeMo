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
from typing import Optional

class UsageError(Exception):
    pass


def str_to_bool(name: str, s: str, optional: bool = False) -> Optional[bool]:
    s = s.lower()
    true_strings = ["true", "1"]
    false_strings = ["false", "0"]
    none_strings = ["auto", "none"]
    if s in true_strings:
        return True
    if s in false_strings:
        return False
    if optional and s in none_strings:
        return None
    raise UsageError(f"Invalid boolean value for argument --{name}: '{s}'")


def add_quantization_flags(parser):
    parser.add_argument(
        "-fp8",
        "--export_fp8_quantized",
        default="auto",
        type=str,
        help="Enables exporting to a FP8-quantized TRT LLM checkpoint",
    )
    parser.add_argument(
        "-kv_fp8",
        "--use_fp8_kv_cache",
        default="auto",
        type=str,
        help="Enables exporting with FP8-quantizatized KV-cache",
    )
    return parser


def parse_quantization_args(args):
    args.export_fp8_quantized = str_to_bool("export_fp8_quantized", args.export_fp8_quantized, optional=True)
    args.use_fp8_kv_cache = str_to_bool("use_fp8_kv_cache", args.use_fp8_kv_cache, optional=True)
    return args


def add_multi_block_mode_flag(parser):
    parser.add_argument(
        "-mbm",
        '--multi_block_mode',
        default=False,
        action='store_true',
        help='Split long kv sequence into multiple blocks (applied to generation MHA kernels). \
                        It is beneifical when batchxnum_heads cannot fully utilize GPU. \
                        Only available when using c++ runtime.',
    )
    return parser


def add_export_flags(parser, deploy: bool):
    parser.add_argument("-nc", "--nemo_checkpoint", required=(not deploy), type=str, help="Source .nemo file")
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        required=(not deploy),
        choices=["gptnext", "gpt", "llama", "falcon", "starcoder", "mixtral", "gemma"],
        help="Type of the model. gptnext, gpt, llama, falcon, starcoder, mixtral and gemma are only supported."
        " gptnext and gpt are the same and keeping it for backward compatibility",
    )
    parser.add_argument("-ng", "--num_gpus", default=None, type=int, help="Number of GPUs for the deployment")
    parser.add_argument("-tps", "--tensor_parallelism_size", default=1, type=int, help="Tensor parallelism size")
    parser.add_argument("-pps", "--pipeline_parallelism_size", default=1, type=int, help="Pipeline parallelism size")
    parser.add_argument("-mil", "--max_input_len", default=256, type=int, help="Max input length of the model")
    parser.add_argument("-mol", "--max_output_len", default=256, type=int, help="Max output length of the model")
    parser.add_argument("-mbs", "--max_batch_size", default=8, type=int, help="Max batch size of the model")
    parser.add_argument("-mnt", "--max_num_tokens", default=None, type=int, help="Max number of tokens")
    parser.add_argument("-ont", "--opt_num_tokens", default=None, type=int, help="Optimum number of tokens")
    parser.add_argument(
        "-mpet", "--max_prompt_embedding_table_size", default=None, type=int, help="Max prompt embedding table size"
    )
    parser.add_argument(
        "-npkc", "--no_paged_kv_cache", default=False, action='store_true', help="Enable paged kv cache."
    )
    parser.add_argument(
        "-drip",
        "--disable_remove_input_padding",
        default=False,
        action='store_true',
        help="Disables the remove input padding option.",
    )
    parser.add_argument(
        "-dt",
        "--dtype",
        choices=["bfloat16", "float16", "fp8", "int8"],
        default="bfloat16",
        type=str,
        help="dtype of the model on TensorRT-LLM",
    )
    parser.add_argument(
        '--use_lora_plugin',
        nargs='?',
        const=None,
        choices=['float16', 'float32', 'bfloat16'],
        help="Activates the lora plugin which enables embedding sharing.",
    )
    parser.add_argument(
        '--lora_target_modules',
        nargs='+',
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
        help="Add lora in which modules. Only be activated when use_lora_plugin is enabled.",
    )
    parser.add_argument(
        '--max_lora_rank',
        type=int,
        default=64,
        help='maximum lora rank for different lora modules. '
        'It is used to compute the workspace size of lora plugin.',
    )
    parser = add_quantization_flags(parser)
    if deploy:
        parser.add_argument("-msl", "--max_seq_len", default=None, type=int, help="Maximum number of sequence length")
        parser.add_argument(
            "-upe",
            "--use_parallel_embedding",
            default=False,
            action='store_true',
            help='Use parallel embedding feature of TensorRT-LLM.',
        )
        parser.add_argument("-mp", "--multiple_profiles", default=False, action='store_true', help="Multiple profiles")
        parser.add_argument(
            "-gap", "--gpt_attention_plugin", default="auto", type=str, help="dtype of gpt attention plugin"
        )
        parser.add_argument("-gp", "--gemm_plugin", default="auto", type=str, help="dtype of gpt plugin")
    
    return parser


def create_parser(description: str, export_parser: bool = False, deploy: bool = True):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description,
    )
    if export_parser:
        parser = add_export_flags(parser, deploy)
    return parser


def add_export_kwargs(args, deploy):
    args = parse_quantization_args(args)
    args.export_kwargs = {
        'nemo_checkpoint_path': args.nemo_checkpoint,
        'model_type': args.model_type,
        'n_gpus': args.num_gpus,
        'tensor_parallelism_size': args.tensor_parallelism_size,
        'pipeline_parallelism_size': args.pipeline_parallelism_size,
        'max_input_len': args.max_input_len,
        'max_output_len': args.max_output_len,
        'max_batch_size': args.max_batch_size,
        'max_num_tokens': args.max_num_tokens,
        'opt_num_tokens': args.opt_num_tokens,
        'max_prompt_embedding_table_size': args.max_prompt_embedding_table_size,
        'paged_kv_cache': (not args.no_paged_kv_cache),
        'remove_input_padding': (not args.disable_remove_input_padding),
        'dtype': args.dtype,
        'use_lora_plugin': args.use_lora_plugin,
        'lora_target_modules': args.lora_target_modules,
        'max_lora_rank': args.max_lora_rank,
        'fp8_quantized': args.export_fp8_quantized,
        'fp8_kvcache': args.use_fp8_kv_cache,
    }
    if deploy:
        args.export_kwargs |= {
            'max_seq_len': args.max_seq_len,
            'use_parallel_embedding': args.use_parallel_embedding,
            'multiple_profiles': args.multiple_profiles,
            'gpt_attention_plugin': args.gpt_attention_plugin,
            'gemm_plugin': args.gemm_plugin,
        }
    return args
