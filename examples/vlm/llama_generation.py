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

from nemo import lightning as nl
from nemo.collections.vlm import Llava1_5Config7B, LlavaModel
from nemo.utils import logging


def load_image(image_url: str) -> Image.Image:
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error loading image from {image_url}: {e}")
        return None


def main() -> None:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        ckpt_include_optimizer=False,
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
    from transformers import AutoTokenizer, AutoModel
    hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

    from nemo.collections.vlm.llama.model.base import CrossAttentionModelVisionConfig, CrossAttentionModelVision, CrossAttentionModelTextConfig, CrossAttentionModelText

    vision_config = CrossAttentionModelVisionConfig(num_layers=2, hidden_size=12, num_attention_heads=4)
    # vision_model = vision_config.configure_model()

    # fabric = trainer.to_fabric()
    #
    # # Decide whether to import or load the model based on the input arguments
    # if args.load_from_hf:
    #     model = fabric.import_model("hf://llava-hf/llava-1.5-7b-hf", LlavaModel)
    # else:
    #     model = LlavaModel(Llava1_5Config7B(), tokenizer=hf_tokenizer)
    #     model = fabric.load_model(args.local_model_path, model)
    #
    # model = model.module.cuda()
    # model.eval()
    # generated_ids = input_ids.clone()
    #
    # # Greedy generation loop
    # for _ in range(20):
    #     with torch.no_grad():
    #         output = model(
    #             media=media,
    #             input_ids=input_ids,
    #             position_ids=position_ids,
    #             attention_mask=None,
    #         )
    #
    #         next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)
    #
    #         generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
    #
    #         input_ids = generated_ids
    #         position_ids = (
    #             torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
    #             .unsqueeze(0)
    #             .expand_as(input_ids)
    #         )
    #
    #         # If the generated token is the end of sequence token, stop generating
    #         if next_token_ids.item() == hf_tokenizer.eos_token_id:
    #             break
    #
    # generated_ids[generated_ids == -200] = 0
    # generated_texts = hf_tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    # logging.info("======== GENERATED TEXT OUTPUT ========")
    # logging.info(f"{generated_texts}")
    # logging.info("=======================================")


if __name__ == "__main__":


    main()