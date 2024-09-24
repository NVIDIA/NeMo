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

import requests
import torch
from PIL import Image
from transformers import AutoProcessor
from nemo.collections import llm, vlm

from nemo import lightning as nl
from nemo.collections.vlm import Llava1_5Config7B, LlavaModel
from nemo.utils import logging


def main() -> None:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        ckpt_load_optimizer=False,
        ckpt_save_optimizer=False,
    )
    trainer = nl.Trainer(
        devices=1,
        max_steps=1000,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        val_check_interval=1000,
        limit_val_batches=50,
    )

    fabric = trainer.to_fabric()

    # Decide whether to import or load the model based on the input arguments
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    model = vlm.LlamaCrossAttentionModel(
        vlm.LlamaCrossAttentionModelConfig(
            language_model_config=vlm.CrossAttentionTextModelConfig8B(),
            vision_model_config=None, #vlm.CrossAttentionVisionModelConfig(num_layers=32, hidden_size=1280, num_attention_heads=16, vision_chunk_size=448, vision_max_num_chunks=4,),  # vlm.CrossAttentionVisionModelConfig(num_layers=32, hidden_size=1280, num_attention_heads=16, vision_chunk_size=448, vision_max_num_chunks=4,),
        ),
        tokenizer=tokenizer)
    local_model_path = "/lustre/fsw/coreai_dlalgo_llm/nemo_home/models/evian3-11b-vision-final_vv1_text_only_zarr/"
    # local_model_path = "/lustre/fsw/coreai_dlalgo_llm/nemo_home/models/evian3-11b-vision-early_vv1_vision_only/"
    model = fabric.load_model(local_model_path, model)

    model = model.module.cuda()
    model.eval()
    model = model.to(torch.bfloat16)

    input = torch.load("/lustre/fsw/coreai_dlalgo_genai/yuya/evian3/evian3_input.pt")

    model(
        position_ids=input['position_ids'].unsqueeze(0),
        tokens=input['tokens'],
        labels=None,
        batch_images=[[]],
        batch_masks=[input['mask']],
        total_len=input['total_len'],
        # cross_attention_masks=input['cross_attention_masks'],
        # full_text_row_masked_out_mask=input['full_text_row_masked_out_mask'],
        # xattn_caches=input['xattn_caches'],
    )

if __name__ == "__main__":
    main()
