import dataclasses
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import torch
from diffusers.image_processor import VaeImageProcessor
from megatron.energon import (
    CaptioningSample,
    DefaultTaskEncoder,
    VQASample,
    WorkerConfig,
    batch_list,
    batch_pad_stack,
    batch_stack,
    get_loader,
    get_train_dataset,
)
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor

from nemo.collections.multimodal.data.energon.config import ImageTextRawBatch, ImageTextSample, MultiModalSampleConfig
from nemo.collections.multimodal.data.energon.sample_encoder import SampleEncoder, VQASampleEncoder
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.collections.multimodal.mimo.data.captioning import MimoCaptioningRawBatch
from nemo.collections.multimodal.mimo.data.templates import (
    COMPHREHENSION_PROMPTS,
    GENERATION_KEYWORDS,
    GENERATION_PROMPTS,
    IMAGE_KEYWORDS,
    RESPONSES,
)
from nemo.collections.vlm import LlavaNextTaskEncoder
from nemo.collections.vlm.llava_next.data.energon import LlavaNextTextRawBatch, LlavaNextTextSample
from nemo.utils import logging


class MimoVqaTaskEncoder(LlavaNextTaskEncoder):
    def batch(self, samples: List[LlavaNextTextSample]) -> MimoCaptioningRawBatch:
        llava_next_raw_text_batch = super().batch(samples)
        # extra keys needed for mimo - image_token_mask, output images (None), input_text (None)
        image_token_mask = llava_next_raw_text_batch.tokens == -200
        return MimoCaptioningRawBatch(
            __keys__=llava_next_raw_text_batch.__keys__,
            images=llava_next_raw_text_batch.images,
            output_images=None,
            tokens=llava_next_raw_text_batch.tokens,
            labels=llava_next_raw_text_batch.labels,
            loss_mask=llava_next_raw_text_batch.loss_mask,
            input_text=None,
            num_image_tiles=llava_next_raw_text_batch.num_media_tiles,
            image_token_mask=image_token_mask,
        )

    def encode_batch(self, batch_data: MimoCaptioningRawBatch) -> dict:
        batch_dict = dataclasses.asdict(batch_data)
        micro_batch_size, seq_length = batch_dict['tokens'].size()
        position_ids = torch.arange(seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
        batch_dict['position_ids'] = position_ids
        batch_dict['attention_mask'] = None
        return batch_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='path to the dataset directory')
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    tokenizer = processor.tokenizer
    special_tokens = [f"IMG_{i}" for i in range(8)]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    multimodal_sample_config = MultiModalSampleConfig()
    multimodal_sample_config.conversation_template_config.system = None

    worker_config = WorkerConfig.default_worker_config(0)

    train_loader = get_loader(
        get_train_dataset(
            args.data_path,
            batch_size=128,
            shuffle_buffer_size=100,
            max_samples_per_sequence=100,
            task_encoder=MimoVqaTaskEncoder(
                tokenizer=tokenizer,
                image_processor=processor.image_processor,
                multimodal_sample_config=MultiModalSampleConfig(),
            ),
            worker_config=worker_config,
        ),
        worker_config=worker_config,
    )

    print(f"data loader length {len(train_loader)}")
    for index, each_batch in enumerate(train_loader):
        print(f"batch index {index} tokens shape {each_batch['tokens'].shape} ")
