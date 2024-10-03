import re
from abc import ABC, abstractmethod
from dataclasses import field
import torch
from einops import rearrange
from megatron.energon import InterleavedSample, SimilarityInterleavedSample, VQASample
from typing import Dict, List, Tuple
from nemo.collections.multimodal.data.energon.config import ImageTextSample, MultiModalSampleConfig
from nemo.utils import logging
from nemo.collections.multimodal.data.energon.sample_encoder import VQASampleEncoder
from typing import Dict, List
from torch.nn.utils.rnn import pad_sequence
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.collections.multimodal.data.energon.config import ImageTextRawBatch
from dataclasses import dataclass, field
from megatron.energon import (
    VQASample,
    batch_list,
    batch_pad_stack,
)
from nemo.collections.multimodal.data.energon.sample_encoder import SampleEncoder


class LlavaNextTextSample(ImageTextSample):
    num_media_tiles: int = 0


@dataclass
class LlavaNextTextRawBatch(ImageTextRawBatch):
    num_media_tiles: List[int] = field(default_factory=list)


class LlavaNextSampleEncoder(VQASampleEncoder):
    def __init__(self, tokenizer, image_processor, multimodal_sample_config=MultiModalSampleConfig()):
        """
        Initialize the VQASampleEncoder.

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        multimodal_sample_config (MultiModalSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to MultiModalSampleConfig().
        """
        super().__init__(tokenizer, image_processor, multimodal_sample_config)
        self.conversation_template_config = multimodal_sample_config.conversation_template_config

    def process_image(self, image):
        image_array = self.image_processor.preprocess(image, return_tensors='pt', do_rescale=False)['pixel_values'][0]
        return image_array

    def encode(self, input_sample: VQASample, output_sample: LlavaNextTextSample):
        conversation_prompt = self.apply_prompt_template(input_sample)
        logging.debug(f"task encoder encode_sample conversation_prompt {conversation_prompt}")
        # tokenize prompt
        tokens = self.tokenize(conversation_prompt)
        labels = self.compute_labels(tokens, input_sample)

        tokens = tokens[:-1].contiguous()
        labels = labels[1:].contiguous()
        logging.debug(f"task encoder encode_sample after tokenize prompt tokens {tokens}")
        logging.debug(f"task encoder encode_sample lables {labels}")
        loss_mask = self.compute_loss_mask(labels)
        processed_image = self.process_image(input_sample.image)
        output_sample.__key__ = input_sample.__key__
        output_sample.images = processed_image
        output_sample.tokens = tokens
        output_sample.labels = labels
        output_sample.loss_mask = loss_mask
        output_sample.num_media_tiles = processed_image.shape[0]
        return output_sample


class LlavaNextTaskEncoder(MultiModalTaskEncoder):
    def __init__(self, tokenizer, image_processor, multimodal_sample_config):
        super().__init__(tokenizer, image_processor, multimodal_sample_config)
        self.encoders: Dict[str, SampleEncoder] = {
            VQASample.__name__: LlavaNextSampleEncoder(tokenizer, image_processor, multimodal_sample_config)
        }

    def batch(self, samples: List[LlavaNextTextSample]) -> LlavaNextTextRawBatch:
        keys, images, tokens, labels, loss_mask, num_media_tiles = [], [], [], [], [], []
        for sample in samples:
            keys.append(sample.__key__)
            images.append(sample.images)
            tokens.append(sample.tokens)
            labels.append(sample.labels)
            loss_mask.append(sample.loss_mask)
            num_media_tiles.append(sample.num_media_tiles)

        batch_keys = batch_list(keys)

        batch_images = torch.cat(images, dim=0)

        batch_tokens = pad_sequence(tokens, batch_first=True)
        batch_labels = pad_sequence(labels, batch_first=True)

        batch_loss_mask = batch_pad_stack(loss_mask)
        batch_num_media_tiles = torch.tensor(batch_list(num_media_tiles),dtype=torch.int)
        return LlavaNextTextRawBatch(
            __keys__=batch_keys,
            images=batch_images,
            tokens=batch_tokens,
            labels=batch_labels,
            loss_mask=batch_loss_mask,
            num_media_tiles=batch_num_media_tiles,
        )
