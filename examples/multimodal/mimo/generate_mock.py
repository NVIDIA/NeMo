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
from nemo.collections.multimodal.mimo.data.mock import MockDataModule
from nemo.collections.multimodal.mimo.model.config import MimoConfig
from nemo.collections.multimodal.mimo.model.model import MimoModel
from nemo.utils import logging


def generate(model, batch, processor, tokenizer):
    # pylint: disable=C0115,C0116
    if batch['images'] is not None:
        images = batch['images'].cuda()
    else:
        images = None
    input_ids = batch['tokens'].cuda()[:, :5]
    position_ids = batch['position_ids'].cuda()[:, :5]
    num_image_tiles = batch['num_image_tiles'].cuda()
    image_token_index = -200
    image_token_mask = batch['image_token_mask'].cuda()[:, :5]
    generated_ids = input_ids.clone()
    input_txt = batch['input_text']

    for _ in range(4):
        with torch.no_grad():
            output, hidden_states = model(
                images=images,
                input_ids=input_ids,
                position_ids=position_ids,
                num_image_tiles=num_image_tiles,
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
            if next_token_ids.item() == tokenizer.tokenizer.eos_token_id:
                print(f"breaking")
                break
    generated_ids[generated_ids == -200] = 0
    generated_texts = tokenizer.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    logging.info("======== GENERATED TEXT OUTPUT ========")
    logging.info(f"{generated_texts}")
    logging.info("=======================================")


def main(args) -> None:
    # pylint: disable=C0115,C0116
    stage = args.stage
    model_id = 'llava-hf/llava-v1.6-vicuna-7b-hf'
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
        plugins=nl.MegatronMixedPrecision(precision="32"),
        val_check_interval=1000,
        limit_val_batches=50,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    image_processor = processor.image_processor
    tokenizer = AutoTokenizer(model_id)
    special_tokens = [f"IMG_{i}" for i in range(8)]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    image_special_tokens = [f"IMG_{i}" for i in range(8)]
    image_special_token_indices = [tokenizer.tokenizer.convert_tokens_to_ids(f"IMG_{i}") for i in range(8)]

    data = MockDataModule(
        tokenizer=tokenizer,
        image_processor=image_processor,
        stage=stage,
        vocab_size=tokenizer.vocab_size,
        micro_batch_size=1,
    )

    fabric = trainer.to_fabric()

    mimo_config = MimoConfig(
        stage=stage,
        language_transformer_config=Llama2Config7B(num_layers=1),
        vocab_size=tokenizer.vocab_size,
        image_special_token_indices=image_special_token_indices,
        image_special_tokens=image_special_tokens,
        freeze_language_model=False,
    )

    model = MimoModel(config=mimo_config, tokenizer=tokenizer)
    model = fabric.load_model(args.local_model_path, model)

    model = model.module.cuda()
    model.eval()
    model = model.to(torch.bfloat16)
    data.setup()
    dataloader = data.test_dataloader()
    batch = next(iter(dataloader))

    generate(model, batch, processor, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mimo mock generation")
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        required=True,
        help="Local path to the model if not loading from Hugging Face.",
    )
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--stage", type=str, required=False, default="encoder_alignment")

    args = parser.parse_args()
    main(args)
