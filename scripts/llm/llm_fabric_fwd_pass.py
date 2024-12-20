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
'''
torchrun --nproc_per_node=1 llm_fabric_fwd_pass.py --load_from_hf
'''
import os 
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import argparse

import requests
import torch
from PIL import Image
from transformers import AutoProcessor
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo import lightning as nl
from nemo.collections import llm
from nemo.utils import logging
from huggingface_hub import login
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.nlp.models.language_modeling.megatron.gpt_layer_modelopt_spec import (
        get_gpt_layer_modelopt_spec,
    )

def generate(model, data, no_iterations):
    
    with torch.no_grad():
        for i in range(no_iterations):
            one_sample = next(iter(data))
            # returns loss if labels are provided
            output = model(input_ids = one_sample['tokens'].cuda(), 
                       position_ids = one_sample['position_ids'].cuda(),
                       labels = one_sample['labels'].cuda())
            print(f"Avg loss for batch {i}:  {output.mean().item()}")
    

def main(args) -> None:
    # pylint: disable=C0115,C0116
    model_id = 'meta-llama/Llama-3.2-1B'
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        ckpt_load_optimizer=False,
        ckpt_save_optimizer=False,
    )
    trainer = nl.Trainer(
        devices=args.tp_size,
        max_steps=1000,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        val_check_interval=1000,
        limit_val_batches=50,
    )
    
    tokenizer = AutoTokenizer(model_id)

    data_module = MockDataModule(seq_length=256, global_batch_size=2, micro_batch_size=1)
    val_dataset = data_module.val_dataloader()
    
    fabric = trainer.to_fabric()

    if True:
        model = fabric.import_model(f"hf://{model_id}", llm.LlamaModel)
    else:
        llama_32_config = llm.Llama32Config1B(transformer_layer_spec=get_gpt_layer_modelopt_spec())
        model = llm.LlamaModel(llm.Llama32Config1B, tokenizer=tokenizer)
        model = fabric.load_model(args.local_model_path, model)
    
    model = model.module.cuda()
    model.eval()
    model = model.to(torch.bfloat16)
    
    generate(model, val_dataset, no_iterations=args.no_iterations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llava Next Generation example")
    parser.add_argument(
        "--load_from_hf",
        action="store_true",
        help="Flag to indicate whether to load the model from Hugging Face hub.",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Local path to the model if not loading from Hugging Face.",
    )
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--no_iterations", type=int, required=False, default=2)
    
    args = parser.parse_args()
    main(args)