import os
import sys
from nemo.utils import logging


from nemo.collections.multimodal.data.energon import SimpleMultiModalDataModule

logging.setLevel(logging.DEBUG)
from transformers import AutoProcessor
from megatron.energon import VQASample
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.vlm.llama.data.task_encoder import LlamaTaskEncoder


if __name__ == '__main__':
    data_path = '/home/ykarnati/Downloads/LLaVA-Pretrain/wds'
    model_directory = "/home/ykarnati/Downloads/HF_HOME/evian3-11b-vision-instruct-final-hf_vv1"
    # model_id = "evian3-11b-vision-instruct-final-hf_vv1"
    processor = AutoProcessor.from_pretrained(model_directory)
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer

    multimodal_sample_config = MultiModalSampleConfig()
    multimodal_sample_config.conversation_template_config.system = "You are a helpful assistant"
    multimodal_sample_config.conversation_template_config.chat_template = None
    multimodal_sample_config.image_token.token_id = 128256
    multimodal_sample_config.conversation_template_config.stop_string = None

    task_encoder = LlamaTaskEncoder(
        tokenizer=tokenizer, image_processor=image_processor, multimodal_sample_config=multimodal_sample_config
    )
    data_module = SimpleMultiModalDataModule(
        path=data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_workers=0,
        micro_batch_size=2,
        multimodal_sample_config=multimodal_sample_config,
        task_encoder=task_encoder,
    )

    train_loader = data_module.train_dataloader()

    for batch in train_loader:
        print("keys  ", batch["__keys__"])
        print("image tensor shape", batch["media"].shape)
        print("prompt tokens tensor shape", batch["tokens"].shape)
        print("labels tensor shape", batch["labels"].shape)
        print("loss mask shape", batch["loss_mask"].shape)
        print("************************************")
        print(f"labels 0 shape", batch["labels"][0])
        print("**********************************")
        print(f"loss_mask   {batch['loss_mask']}")
        print("**********************************")
        print(f"vision mask {batch['vision_mask']}")
        print("**********************************")
        break
