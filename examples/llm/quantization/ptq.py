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
import sys
import torch
from tqdm import tqdm

from nemo.collections.llm.quantization import Quantizer, get_calib_data_iter


# TODO: Inference TP/PP != Calibration TP/PP
def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="NeMo PTQ argument parser",
    )
    parser.add_argument("-nc", "--nemo_checkpoint", type=str, help="Source NeMo 2.0 checkpoint")
    parser.add_argument("--decoder_type", type=str, help="Decoder type for TensorRT-Model-Optimizer")
    parser.add_argument(
        "-tps",
        "--tensor_parallelism_size",
        type=int,
        default=1
    )
    parser.add_argument(
        '-out',
        '--output_path',
        type=str,
        help='Path for the exported engine'
    )
    parser.add_argument(
        '-algo',
        '--algorithm',
        type=str,
        default="no_quant",
        choices=["no_quant", "int8", "int8_sq", "fp8", "int4_awq", "w4a8_awq", "int4"],
        help='TensorRT-Model-Optimizer quantization algorithm'
    )
    parser.add_argument(
        '-awq_bs',
        '--awq_block_size',
        type=int,
        default=128,
        help='Block size for AWQ quantization algorithms'
    )
    parser.add_argument(
        '--sq_alpha',
        type=float,
        default=1.0,
        help='Smooth-Quant alpha parameter'
    )
    parser.add_argument(
        '--enable_kv_cache',
        type=bool,
        help='Enables KV-cache quantization'
    )
    parser.add_argument(
        '-dt',
        '--dtype',
        default="bf16",
        choices=["16", "bf16"],
        help='Default precision for non-quantized layers'
    )
    
    return parser.parse_args(sys.argv[1:])


def get_quantizer_config(args):
    if args.output_path is None:
        args.output_path = f"./trt_llm_{args.algorithm}_tp{args.tensor_parallelism_size}"

    quantization_config = {
        "algorithm": None if args.algorithm == "no_quant" else args.algorithm,
        "awq_block_size": args.awq_block_size,
        "sq_alpha": args.sq_alpha,
        "enable_kv_cache": args.enable_kv_cache,
    }

    export_config = {
        "path": args.output_path,
        "decoder_type": args.decoder_type,
        "inference_tensor_parallel": args.tensor_parallelism_size,
        "inference_pipeline_parallel": 1,
        "dtype": args.dtype,
    }
    return quantization_config, export_config


# TODO: maybe use llm.generate (#10471)
def forward_loop(model):
    tokenizer = model.tokenizer
    dataloader = get_calib_data_iter()
    dataloader = [data for data in dataloader]

    for batch in tqdm(dataloader):
        batch = [tokenizer.text_to_ids(text) for text in batch]
        max_len = max([len(text) for text in batch])
        batch = [ids + (max_len - len(ids)) * [tokenizer.eos] for ids in batch]
        position_ids = torch.arange(max_len, device=model.device).expand((len(batch), max_len))
        batch = torch.tensor(batch, device=model.device)
        model_input = {
            "input_ids": batch,
            "position_ids": position_ids,
            "attention_mask": None,
        }
        model(**model_input)


def main():
    params = get_args()
    quantization_config, export_config = get_quantizer_config(params)

    quantizer = Quantizer(quantization_config, export_config)
    model = quantizer.load_quantizable_model(params.nemo_checkpoint, params.tensor_parallelism_size)
    model = quantizer.quantize(model, forward_loop)
    quantizer.export(model)


if __name__ == '__main__':
    main()
