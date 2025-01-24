from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from megatron.energon import VQASample, batch_list, batch_pad_stack
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.multimodal.data.energon.config import ImageTextRawBatch, ImageTextSample, MultiModalSampleConfig
from nemo.collections.multimodal.data.energon.sample_encoder import SampleEncoder, VQASampleEncoder
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.utils import logging

from nemo.collections.vlm.llava_next.data.sample import LlavaNextTextSample, LlavaNextTextRawBatch


class LlavaNextSampleEncoder(VQASampleEncoder):
    """LlavaNextSampleEncoder"""

    def __init__(self, tokenizer, image_processor, multimodal_sample_config=MultiModalSampleConfig()):
        """
        Initialize the LlavaNextSampleEncoder, inherited from VQASampleEncoder for multimodal samples
        focused on VQA-style data to support LLaVANeXT

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        multimodal_sample_config (MultiModalSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to MultiModalSampleConfig().
        """
        super().__init__(tokenizer, image_processor, multimodal_sample_config)

    def process_image(self, image):
        """
        Process and prepare an image sample for encoding.

        This method preprocesses the image using the HF image_processor, converting it to
        a tensor.

        Parameters:
        image: The input image to be processed.

        Returns:
        torch.Tensor: The processed image tensor.
        """
        image_array = self.image_processor.preprocess(image, return_tensors='pt', do_rescale=False)['pixel_values'][0]
        return image_array

    def encode(self, input_sample: VQASample, output_sample: LlavaNextTextSample):
        """
        Encode a single sample into a format suitable for model input.

        This method prepares the conversation prompt, tokenizes it, and processes
        the associated image. It fills the output sample with tokens, labels, loss mask,
        and other required fields for multimodal processing.

        Parameters:
        input_sample (VQASample): The input VQA sample containing an image and conversation text.
        output_sample (LlavaNextTextSample): The output sample structure where encoded results are stored.

        Returns:
        LlavaNextTextSample: The encoded output sample, containing processed tokens, labels,
            images, loss masks, and metadata.
        """
        conversation_prompt = self.apply_prompt_template(input_sample)
        logging.debug(f"[Energon] task encoder encode_sample conversation_prompt {conversation_prompt}")
        # tokenize prompt
        tokens = self.tokenize(conversation_prompt)
        labels = self.compute_labels(tokens, input_sample)
        tokens = tokens[:-1].contiguous()
        labels = labels[1:].contiguous()
        logging.debug(f"[Energon] task encoder encode_sample after tokenize prompt tokens {tokens}")
        logging.debug(f"[Energon] task encoder encode_sample lables {labels}")
        loss_mask = self.compute_loss_mask(labels)
        processed_image = self.process_image(input_sample.image)
        output_sample.__key__ = input_sample.__key__
        output_sample.images = processed_image
        output_sample.tokens = tokens
        output_sample.labels = labels
        output_sample.loss_mask = loss_mask
        output_sample.num_media_tiles = processed_image.shape[0]
        output_sample.attention_mask = torch.ones(len(tokens), dtype=torch.long)
        height = input_sample.image.shape[1]
        width = input_sample.image.shape[2]
        output_sample.image_sizes = torch.tensor([[height, width]], dtype=torch.long)
        return output_sample
