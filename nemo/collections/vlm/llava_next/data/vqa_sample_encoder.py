from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from megatron.energon import VQASample, batch_list, batch_pad_stack
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.multimodal.data.energon.config import ImageTextRawBatch, ImageTextSample, MultiModalSampleConfig
from nemo.collections.multimodal.data.energon.sample_encoder import SampleEncoder, VQASampleEncoder
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.collections.vlm.llava_next.data.sample import LlavaNextTextRawBatch, LlavaNextTextSample
from nemo.collections.vlm.llava_next.model.utils import select_best_resolution
from nemo.utils import logging


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
        from transformers import LlavaNextConfig

        self.hf_config = LlavaNextConfig()

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

        height = input_sample.image.shape[1]
        width = input_sample.image.shape[2]

        if True:

            text = [conversation_prompt]
            image_sizes = iter([[height, width]])
            resized_height, resized_width = (
                self.hf_config.vision_config.image_size,
                self.hf_config.vision_config.image_size,
            )
            prompt_strings = []
            for sample in text:
                while self.image_token.token_str in sample:
                    image_size = next(image_sizes)
                    if not isinstance(image_size, (list, tuple)):
                        # cast to list to avoid numerical precision errors when calculating unpadding
                        image_size = image_size.tolist()
                    orig_height, orig_width = image_size
                    num_image_tokens = self._get_number_of_features(
                        orig_height, orig_width, resized_height, resized_width
                    )
                    sample = sample.replace(self.image_token.token_str, "<placeholder>" * num_image_tokens, 1)
                prompt_strings.append(sample)
            prompt_strings = [sample.replace("<placeholder>", self.image_token.token_str) for sample in prompt_strings]
            conversation_prompt = prompt_strings[0]

            ### FOr debugging
            conversation_prompt_q = conversation_prompt
            tokens_q = self.tokenize(conversation_prompt_q)
            labels_q = self.compute_labels(tokens_q, input_sample)
            tokens_q = tokens_q[:-1].contiguous()
            output_sample.extra_tokens = tokens_q
            labels_q = labels_q[1:].contiguous()
            print(f"Shape of tokens_q {tokens_q.shape}, labels_q {labels_q.shape}")
            ## For Debug

        logging.debug(f"[Energon] task encoder encode_sample conversation_prompt {conversation_prompt}")
        # tokenize prompt
        tokens = self.tokenize(conversation_prompt)
        labels = self.compute_labels(tokens, input_sample)
        tokens = tokens[:-1].contiguous()
        labels = labels[1:].contiguous()
        logging.debug(f"[Energon] task encoder encode_sample after tokenize prompt tokens {tokens}")
        logging.debug(f"[Energon] task encoder encode_sample lables {labels}")
        loss_mask = self.compute_loss_mask(labels)

        # Here the image goes from 3,338,336 --> 3,3,336,336 (So resize and patch creation)
        processed_image = self.process_image(input_sample.image)
        output_sample.__key__ = input_sample.__key__
        output_sample.images = processed_image

        output_sample.tokens = tokens
        output_sample.labels = labels
        output_sample.loss_mask = loss_mask
        output_sample.num_media_tiles = processed_image.shape[0]
        output_sample.attention_mask = torch.ones(len(tokens), dtype=torch.long)

        output_sample.image_sizes = torch.tensor([[height, width]], dtype=torch.long)
        return output_sample

    def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
        image_grid_pinpoints = self.hf_config.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height, scale_width = height_best_resolution // height, width_best_resolution // width

        patches_height = height // self.hf_config.vision_config.patch_size
        patches_width = width // self.hf_config.vision_config.patch_size
        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height, orig_width, patches_height, patches_width, scale_height, scale_width
        )
        # The base patch covers the entire image (+1 for the CLS)
        # We do not add any CLS token as we assume the vision strategy is "default"
        # TODO(abhi, yash): Check if we need other vision strategies
        # base_features = patches_height * patches_width + self.num_additional_image_tokens

        base_features = patches_height * patches_width

        num_image_tokens = unpadded_features + newline_features + base_features
        print(f"{base_features = }, {unpadded_features = }, {newline_features = }")
        return num_image_tokens

    def _get_unpadded_features(self, height, width, patches_height, patches_width, scale_height, scale_width):
        """
        Get number of features for a given image with height/width. LLaVA-NeXT is different from LLaVA
        because it divided each image into patches depending on its resolution. Therefore we need to calculate how many
        patches an image is divided into and get the number of features from that.
        """
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = (height * current_width) // width
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = (width * current_height) // height
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height
        return (unpadded_features, newline_features)
