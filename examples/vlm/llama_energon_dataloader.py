import os
import sys

from nemo.collections.multimodal.data.energon import SimpleMultiModalDataModule
from nemo.utils import logging

logging.setLevel(logging.DEBUG)
import requests
import torch
from megatron.energon import VQASample
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.multimodal.data.energon.conversation import LLaVATemplateConfig, MLlamaTemplateConfig
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.collections.vlm.llama.data.task_encoder import LlamaTaskEncoder

if __name__ == '__main__':
    data_path = '/lustre/fsw/coreai_dlalgo_genai/datasets/LLaVA-Instruct-150K/yash/wds'
    # model_directory = "/home/ykarnati/Downloads/HF_HOME/evian3-11b-vision-instruct-final-hf_vv1"
    # model_id = "evian3-11b-vision-instruct-final-hf_vv1"
    model_directory = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    processor = AutoProcessor.from_pretrained(model_directory)
    image_processor = processor.image_processor
    tokenizer = processor.tokenizer

    multimodal_sample_config = MultiModalSampleConfig()
    multimodal_sample_config.conversation_template_config = MLlamaTemplateConfig()

    task_encoder = LlamaTaskEncoder(
        tokenizer=tokenizer,
        image_processor=image_processor,
        multimodal_sample_config=multimodal_sample_config,
        seq_length=None,
    )
    data_module = SimpleMultiModalDataModule(
        path=data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_workers=0,
        micro_batch_size=1,
        global_batch_size=1,
        multimodal_sample_config=multimodal_sample_config,
        task_encoder=task_encoder,
        seq_length=6404,
        decoder_seq_length=512,
    )

    train_loader = data_module.train_dataloader()

    for batch in train_loader:
        print("keys  ", batch["__keys__"])
        print("image tensor shape", batch["batch_images"].shape)
        print("prompt tokens tensor shape", batch["tokens"].shape)
        print("labels tensor shape", batch["labels"].shape)
        print("loss mask shape", batch["loss_mask"].shape)
        print("************************************")
        print(f"labels 0 shape", batch["labels"][0])
        print("**********************************")
        print(f"loss_mask   {batch['loss_mask']}")
        print("**********************************")
        break
