import argparse
import os
import sys

from nemo.collections.multimodal.data.energon import SimpleMultiModalDataModule
from nemo.utils import logging

logging.setLevel(logging.DEBUG)
import requests
import torch
from megatron.energon import VQASample
from PIL import Image
from transformers import AutoProcessor

from nemo import lightning as nl
from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.vlm import Llava1_5Config7B, LlavaModel, LlavaNextTaskEncoder
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


def main(args) -> None:
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

    # Tokenize the input texts
    # processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    data_path = '/home/ykarnati/Downloads/LLaVA-Pretrain/wds'
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer

    multimodal_sample_config = MultiModalSampleConfig()

    task_encoder = LlavaNextTaskEncoder(
        tokenizer=tokenizer, image_processor=image_processor, multimodal_sample_config=multimodal_sample_config
    )
    data_module = SimpleMultiModalDataModule(
        path=data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_workers=0,
        micro_batch_size=1,
        multimodal_sample_config=multimodal_sample_config,
        task_encoder=task_encoder,
    )

    train_loader = data_module.train_dataloader()
    one_batch = next(iter(train_loader))

    fabric = trainer.to_fabric()

    # Decide whether to import or load the model based on the input arguments
    if args.load_from_hf:
        # model = fabric.import_model("hf://llava-hf/llava-1.5-7b-hf", LlavaModel)
        model = fabric.import_model("hf://llava-hf/llava-v1.6-vicuna-7b-hf", LlavaModel)
        #
    else:
        model = LlavaModel(Llava1_5Config7B(), tokenizer=tokenizer)
        model = fabric.load_model(args.local_model_path, model)

    model = model.module.cuda()
    model.eval()

    # Greedy generation loop
    media = one_batch["media"].cuda()
    input_ids = one_batch["tokens"].cuda()
    position_ids = one_batch["position_ids"].cuda()
    num_media_tiles = one_batch["num_media_tiles"]
    generated_ids = input_ids.clone()
    for _ in range(20):
        with torch.no_grad():

            output = model(
                media=media,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                num_media_tiles=num_media_tiles,
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
            if next_token_ids.item() == tokenizer.eos_token_id:
                break

    generated_ids[generated_ids == -200] = 0
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
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
