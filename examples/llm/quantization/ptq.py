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


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="NeMo PTQ argument parser",
    )
    parser.add_argument("-nc", "--nemo_checkpoint", type=str, help="Source NeMo 2.0 checkpoint")
    parser.add_argument("--decoder_type", type=str, help="Decoder type for TensorRT-Model-Optimizer")
    parser.add_argument(
        "-ctp",
        "--calib_tp",
        type=int,
        default=1
    )
    parser.add_argument(
        "-cpp",
        "--calib_pp",
        type=int,
        default=1
    )
    parser.add_argument(
        "-tps",
        "--tensor_parallelism_size",
        type=int,
        default=1
    )
    parser.add_argument(
        "-pps",
        "--pipeline_parallelism_size",
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
        default=0.5,
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
    parser.add_argument(
        '-bs',
        '--batch_size',
        default=64,
        type=int,
        help='Calibration batch size'
    )
    parser.add_argument(
        '-sl',
        '--seq_len',
        default=128,
        type=int,
        help='Length of the tokenized text'
    )
    parser.add_argument(
        '-calib_size',
        '--calibration_dataset_size',
        default=512,
        type=int,
        help='Size of calibration dataset'
    )
    parser.add_argument(
        '-calib_ds',
        '--calibration_dataset',
        default="cnn_dailymail",
        choices=["wikitext", "cnn_dailymail"],
        type=str,
        help='Calibration dataset to be used'
    )
    
    return parser.parse_args(sys.argv[1:])


def get_quantizer_config(args):
    if args.output_path is None:
        args.output_path = f"./qnemo_{args.algorithm}_tp{args.tensor_parallelism_size}_pp{args.pipeline_parallelism_size}"

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
        "inference_pipeline_parallel": args.pipeline_parallelism_size,
        "dtype": args.dtype,
    }

    return quantization_config, export_config


def create_data_iterator_getter(model, dataset, seq_len, batch_size, calibration_size):
    def _iterator():
        CHARACTERS_PER_TOKEN = 4

        dataloader = get_calib_data_iter(data=dataset, max_sequence_length=CHARACTERS_PER_TOKEN*seq_len, batch_size=batch_size, calib_size=calibration_size)
        for batch in dataloader:
            batch = [model.tokenizer.text_to_ids(text)[:seq_len] for text in batch]
            batch = [ids + (seq_len - len(ids)) * [model.tokenizer.eos] for ids in batch]
            yield torch.tensor(batch, device=model.device)
    
    def _iterator_getter():
        dataloader = _iterator()
        dataloader = [data for data in dataloader]
        return iter(tqdm(dataloader))

    return _iterator_getter
    

def main():
    params = get_args()
    quantization_config, export_config = get_quantizer_config(params)
    quantizer = Quantizer(quantization_config, export_config)
    model = quantizer.load_quantizable_model(params.nemo_checkpoint, params.calib_tp, params.calib_pp)
    
    get_dataloader = create_data_iterator_getter(model,
                                dataset=params.calibration_dataset,
                                seq_len=params.seq_len,
                                batch_size=params.batch_size,
                                calibration_size=params.calibration_dataset_size)

    forward_loop = quantizer.create_megatron_forward_loop(
        get_dataloader,
        num_batches=params.calibration_dataset_size // params.batch_size,
        seq_length=params.seq_len,
        micro_batch_size=params.batch_size,
    )

    model = quantizer.quantize(model, forward_loop)
    quantizer.export(model)


if __name__ == '__main__':
    main()
