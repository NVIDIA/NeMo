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

"""
Example:
  python scripts/vlm/neva_generate.py --load_from_hf
"""

import argparse
import re

import requests
import torch
from PIL import Image
from transformers import AutoProcessor

import nemo.lightning as nl
from nemo.collections.vlm import Llava15Config7B, LlavaModel
from nemo.collections.vlm.neva.model.cosmos_megatron import CosmosMegatronModel
from nemo.utils import logging


def load_image(image_url: str) -> Image.Image:
    # pylint: disable=C0115,C0116
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error loading image from {image_url}: {e}")
        return None


def main(args) -> None:
    # pylint: disable=C0115,C0116
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

    # # Tokenize the input texts
    # processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    #

    from nemo.collections.common.tokenizers import AutoTokenizer
    hf_tokenizer = AutoTokenizer("meta-llama/Llama-3.1-8B-Instruct")
    new_special_tokens = {
        "additional_special_tokens": [
            "<image>", "<img>", "</img>",
            "<quad>", "</quad>",
            "<ref>", "</ref>",
            "<box>", "</box>"
        ]
    }
    hf_tokenizer.tokenizer.add_special_tokens(new_special_tokens)
    fabric = trainer.to_fabric()

    # Decide whether to import or load the model based on the input arguments
    if args.load_from_hf:
        model = fabric.import_model("pyt:////lustre/fsw/coreai_dlalgo_genai/yuya/cosmos-megatron/checkpoints/sft_llama_3p1_8b_radio_v13_0227_tp_1/checkpoints/iter_0010658/mp_rank_00/model_optim_rng.pt", CosmosMegatronModel)
    else:
        model = CosmosMegatronModel(Llava15Config7B(), tokenizer=hf_tokenizer)
        model = fabric.load_model(args.local_model_path, model)

    model = model.module.cuda()
    model.eval()

    img_file = "/lustre/fsw/coreai_dlalgo_genai/datasets/llava-bench-in-the-wild/images/001.jpg"
    conversation = [
        {"role": "system", "content": "Answer the questions."},
        {
            "role": "user",
            "content": f"<img><image></img>\nProvide a one-sentence caption for provided image.",
        },
    ]
    prompt = hf_tokenizer.tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    prompt = re.sub(r"Cutting Knowledge Date:.*\n\n", "", prompt)
    input_ids = hf_tokenizer.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()

    img = Image.open(img_file)
    from image_processing import get_visual_transform
    imgs = get_visual_transform(
        img,
        img_h=512,
        img_w=512,
        use_tiling=True,
        max_num_tiles=12,
        use_thumbnail=True,
        augment=False,
        vision_model_type="radio",
    )
    images = torch.stack(imgs).cuda()
    num_image_tiles = torch.tensor([len(imgs)], dtype=torch.int).cuda()

    # inputs = torch.load("inputs.pt")
    # images = inputs["images"]
    # input_ids = inputs["tokens"]
    # num_image_tiles = inputs["num_image_tiles"]

    generated_ids = input_ids.clone()
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        .unsqueeze(0)
        .expand_as(input_ids)
    )
    # Greedy generation loop
    for _ in range(20):
        with torch.no_grad():
            output = model(
                images=images,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                num_image_tiles=num_image_tiles,
                image_token_index=128256,
            )

            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )

            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() == hf_tokenizer.tokenizer.eos_token_id:
                break

    # generated_ids[generated_ids == -200] = 0
    generated_texts = hf_tokenizer.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    logging.info("======== GENERATED TEXT OUTPUT ========")
    logging.info(f"{generated_texts}")
    logging.info("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA Multimodal Inference")
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
    parser.add_argument(
        "--image_url",
        type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="URL of the image to use for inference.",
    )
    args = parser.parse_args()

    main(args)
