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
from datasets import load_dataset

from nemo import lightning as nl
from nemo.collections.nlp.models.language_modeling.megatron.gpt_layer_modelopt_spec import get_gpt_layer_modelopt_spec
import nemo.collections.llm as llm

HAS_MODELOPT = True
try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_tensorrt_llm_checkpoint
except:
    HAS_MODELOPT = False


# Sample hyperparameters
MODEL_HPARAMS = {
    "llama3": ('meta-llama/Meta-Llama-3-8B', "llama", llm.LlamaModel, llm.Llama3Config8B),
    "llama3-70b": ('meta-llama/Meta-Llama-3-70B', "llama", llm.LlamaModel, llm.Llama3Config70B),
    "mistral": ('mistralai/Mistral-7B-Instruct-v0.3', "llama", llm.MistralModel, llm.MistralConfig7B),
    "gemma": ('google/gemma-2b', "gemma", llm.GemmaModel, llm.GemmaConfig2B),
}


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="NeMo PTQ argument parser",
    )
    parser.add_argument("-nc", "--nemo_checkpoint", type=str, help="Source NeMo 2.0 checkpoint")
    parser.add_argument(
        "-tps",
        "--tensor_parallelism_size",
        type=int,
        default=1
    )
    parser.add_argument(
        '-id',
        '--model_id',
        type=str,
        required=True,
        choices=list(MODEL_HPARAMS.keys()),
        help='Model id for MODEL_HPARAMS map'
    )
    parser.add_argument(
        '-out',
        '--output_path',
        type=str,
        default="",
        help='Path for the exported engine'
    )

    args = parser.parse_args(sys.argv[1:])
    if args.output_path == "":
        args.output_path = f'./trt_llm_fp8_ckpt-{args.model_id}-tp{args.tensor_parallelism_size}'

    args.name, args.modelopt_type, args.model, args.config = MODEL_HPARAMS[args.model_id]
    return args


# TODO: Unify implementation with examples/nlp/language_modeling/megatron_gpt_ptq.py
def get_calib_data_iter(data="cnn_dailymail", batch_size=64, calib_size=512, max_sequence_length=512):
    if data == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        text_column = "text"
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        text_column = "article"
    else:
        # Assume a local JSON dataset with a column named "text"
        dataset = load_dataset("json", data_files=data, split="train")
        text_column = "text"
    calib_size = max(min(len(dataset), calib_size), batch_size)
    for i in range(calib_size // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size][text_column]
        for j in range(len(batch)):
            batch[j] = batch[j][:max_sequence_length]
        yield batch


# TODO: generalize
def save_tokenizer_config(model_name: str, output_path: str):
    from os.path import isfile
    tokenizer_path = output_path + '/tokenizer_config.yaml'
    if not isfile(tokenizer_path):
        with open(tokenizer_path, 'w') as tokenizer_config:
            tokenizer_config.write(f"library: huggingface\ntype: {model_name}\nuse_fast: true\nvocab_file: null\nmerge_file: null")


# TODO: use llm.generate (#10471) once merged
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


if __name__ == '__main__':
    if not HAS_MODELOPT:
        print("Modelopt could not be imported")
        exit(1)

    args = get_args()
    
    # TODO: make/extend the Quantizer class from nemo.export.quantizer
    # Configure global state
    trainer = nl.Trainer(
        devices=args.tensor_parallelism_size,
        strategy=nl.MegatronStrategy(tensor_model_parallel_size=args.tensor_parallelism_size),
        plugins=nl.MegatronMixedPrecision(precision='16-mixed')
    )
    fabric = trainer.to_fabric()
    trainer.strategy.setup_environment()

    # Load model with modelopt layer spec
    model = nl.io.load_context(args.nemo_checkpoint).model
    model.config.transformer_layer_spec = get_gpt_layer_modelopt_spec()
    
    # TODO: [0] works only for PP = 1. Will be changed when PP support is added.
    model = fabric.load_model(args.nemo_checkpoint, model=model)[0]

    # TODO: allow other configs
    atq_config = mtq.FP8_DEFAULT_CFG
    enable_quant_kv_cache = True
    print(f'{"Enable" if enable_quant_kv_cache else "Disable"} KV cache quantization')
    atq_config["quant_cfg"]["*output_quantizer"] = {  # type: ignore[index]
        "num_bits": (4, 3),
        "axis": None,
        "enable": enable_quant_kv_cache,
    }

    model = mtq.quantize(model, atq_config, forward_loop)
    mtq.print_quant_summary(model)
    export_tensorrt_llm_checkpoint(
        model,
        args.modelopt_type,
        torch.float16,
        export_dir=args.output_path,
        inference_tensor_parallel=args.tensor_parallelism_size,
        inference_pipeline_parallel=1,
        use_nfs_workspace=False,
    )
    save_tokenizer_config(args.name, args.output_path)

