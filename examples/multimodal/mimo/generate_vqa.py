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
from nemo.collections import vlm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm import Llama2Config7B, import_ckpt
from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.multimodal.mimo.model.config import MimoConfig
from nemo.collections.multimodal.mimo.model.model import MimoModel
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


def generate(model, processor, tokenizer, raw_image, multimodal_sample_config):
    # pylint: disable=C0115,C0116
    messages = []
    if multimodal_sample_config.conversation_template_config.system:
        messages.append({'role': 'system', 'content': multimodal_sample_config.conversation_template_config.system})
    messages.append(
        {
            "role": "user",
            "content": "<image> What does the image show",
        },
    )

    input_text = tokenizer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.tokenizer(input_text, return_tensors='pt').input_ids.cuda()
    inputs = processor(input_text, raw_image, return_tensors='pt').to(0, torch.float32)
    input_ids[input_ids == 32000] = -200
    media = inputs['pixel_values'].cuda()
    media = media.reshape(media.size(1), 3, 336, 336)
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )

    generated_ids = input_ids.clone()
    width, height = raw_image.size
    image_token_mask = input_ids == -200
    image_token_mask = image_token_mask.cuda()
    num_image_tiles = torch.tensor([media.size(0)]).cuda().to(dtype=torch.int32)
    # breakpoint()
    for _ in range(20):
        with torch.no_grad():
            attention_mask = (input_ids != 0).long().cuda()
            output, hidden_states = model(
                images=media,
                input_ids=input_ids,
                position_ids=position_ids,
                num_image_tiles=num_image_tiles,
                # attention_mask=attention_mask,
                image_token_mask=image_token_mask,
            )
            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )
            print(f"next_token_ids {next_token_ids}")

            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() == processor.tokenizer.eos_token_id:
                print(f"breaking")
                break
    generated_ids[generated_ids == -200] = 0
    generated_texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    logging.info("======== GENERATED TEXT OUTPUT ========")
    logging.info(f"{generated_texts}")
    logging.info("=======================================")


def main(args) -> None:
    # pylint: disable=C0115,C0116
    model_id = 'llava-hf/llava-v1.6-vicuna-7b-hf'
    stage = args.stage
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

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer(model_id)
    special_tokens = [f"IMG_{i}" for i in range(8)]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    image_special_tokens = [f"IMG_{i}" for i in range(8)]
    image_special_token_indices = [tokenizer.tokenizer.convert_tokens_to_ids(f"IMG_{i}") for i in range(8)]

    fabric = trainer.to_fabric()
    multimodal_sample_config = MultiModalSampleConfig()

    tokenizer.tokenizer.chat_template = multimodal_sample_config.conversation_template_config.chat_template
    mimo_config = MimoConfig(
        stage=stage,
        language_transformer_config=Llama2Config7B(),
        vocab_size=tokenizer.vocab_size,
        image_special_token_indices=image_special_token_indices,
        image_special_tokens=image_special_tokens,
        freeze_language_model=True,
    )

    model = MimoModel(config=mimo_config, tokenizer=tokenizer)
    model = fabric.load_model(args.local_model_path, model)

    model = model.module.cuda()
    model.eval()
    model = model.to(torch.bfloat16)

    # Load the image
    raw_image = load_image(args.image_url)
    if raw_image is None:
        return  # Exit if the image can't be loaded

    generate(model, processor, tokenizer, raw_image=raw_image, multimodal_sample_config=multimodal_sample_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mimo vqa Generation example")
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        required=True,
        help="Local path to the model if not loading from Hugging Face.",
    )
    parser.add_argument(
        "--image_url",
        type=str,
        # pylint: disable=line-too-long
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="URL of the image to use for inference.",
    )
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--stage", type=str, required=False, default="encoder_alignment")

    args = parser.parse_args()
    main(args)
