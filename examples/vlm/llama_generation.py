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
            language_model_config=vlm.CrossAttentionTextModelConfig8B(rotary_interleaved=True, apply_rope_fusion=False),
            vision_model_config=vlm.CrossAttentionVisionModelConfig(num_layers=32, hidden_size=1280,
                                                                    num_attention_heads=16, vision_chunk_size=448,
                                                                    vision_max_num_chunks=4, ),
            # vlm.CrossAttentionVisionModelConfig(num_layers=32, hidden_size=1280, num_attention_heads=16, vision_chunk_size=448, vision_max_num_chunks=4,),
        ),
        tokenizer=tokenizer)
    local_model_path = "/root/.cache/nemo/models/evian3-11b-vision-final_vv1_zarr/"
    # local_model_path = "/lustre/fsw/coreai_dlalgo_llm/nemo_home/models/evian3-11b-vision-early_vv1_vision_only/"
    model = fabric.load_model(local_model_path, model)

    model = model.module.cuda()
    model.eval()
    model = model.to(torch.bfloat16)

    input = torch.load("/lustre/fsw/coreai_dlalgo_genai/yuya/evian3/evian3_input.pt")

    input_ids = input["tokens"]
    position_ids = input["position_ids"].unsqueeze(0)

    prev_pos = 0
    min_prompt_len = position_ids.shape[-1]
    total_len = input["total_len"]

    input_ids = input_ids[:, :min_prompt_len]
    generated_ids = input_ids.clone()

    for cur_pos in range(min_prompt_len, min_prompt_len+64):
        with torch.no_grad():
            position_ids = torch.arange(
                0, cur_pos, dtype=torch.long, device="cuda"
            ).reshape(1, -1)

            output = model(
                batch_images=[input["images"]],
                batch_masks=[input["mask"]],
                total_len=total_len,
                tokens=generated_ids,
                position_ids=position_ids,
            )

            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
            if next_token_ids == tokenizer.eos_token_id:
                break
            prev_pos = cur_pos

    generated_ids = generated_ids.tolist()
    print(generated_ids)
    print(tokenizer.decode(generated_ids[0][min_prompt_len:]))


if __name__ == "__main__":
    main()
