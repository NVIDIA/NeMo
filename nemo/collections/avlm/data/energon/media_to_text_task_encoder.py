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

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from megatron.energon import batch_list, batch_pad_stack
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.avlm.data.energon.media_to_text_config import (
    MediaToTextSample, 
    MediaToTextEnergonSample
    MediaToTextRawBatch,
    MediaToTextSampleConfig,
)

from nemo.collections.vlm.llava_next.data.energon import LlavaNextSampleEncoder
from nemo.collections.multimodal.data.energon.sample_encoder import SampleEncoder
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.utils import logging


class MediaToTextSampleEncoder(LlavaNextSampleEncoder):
    """LlavaNextSampleEncoder"""

    def __init__(self, tokenizer, audio_processor, image_processor, multimodal_sample_config=MediaToTextSampleConfig()):
        """
        Initialize the LlavaNextSampleEncoder, inherited from LlavaNextSampleEncoder for multimodal samples
        focused on LLaVANeXT data to support 

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        media_to_text_sample_config (MediaToTextSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to MediaToTextSampleConfig().
        """
        super().__init__(tokenizer, image_processor, multimodal_sample_config)
        self.audio_processor = audio_processor

    def process_audio(self, audio):
        """
        Process and prepare an audio sample for encoding.

        Parameters:
        audio: The input audio to be processed.

        Returns:
        torch.Tensor: The processed audio tensor.
        """
        #TODO
        return None

    def process_video(self, video):
        #TODO
        return None

    @stateless
    def encode_sample(self, sample: MediaToTextEnergonSample) -> ImageTextSample:
        """
        Encode an individual sample based on its type.

        This method selects the appropriate encoder based on the sample type and encodes the sample
        into a format suitable for further processing.

        Parameters:
        sample (Union[VQASample, InterleavedSample, SimilarityInterleavedSample, CaptioningSample]):
            The sample to be encoded. The sample type is used to determine the appropriate encoder.

        Returns:
        ImageTextSample: The encoded sample.

        Raises:
        NotImplementedError: If no encoder is registered for the sample type.
        """
        sample_type = type(sample).__name__
        encoder = self.encoders.get(sample_type)
        if not encoder:
            raise NotImplementedError(f"No encoder implemented for sample type {sample_type}")
        encoded_sample = encoder.encode(input_sample=sample, output_sample=MediaToTextSample())
        return encoded_sample

    def encode(self, input_sample: MediaToTextEnergonSample, output_sample: MediaToTextSample):
        """
        Encode a single sample into a format suitable for model input.

        Parameters:

        Returns:
        MediaToTextSample: 
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

        # TODO: check if energon will return None if the keys are not 
        # present in the dataset or it would throw an error
        if input_sample.audios is not None:
            output_sample.audios = self.process_audio(input_sample.audios)

        if input_sample.videos is not None:
            output_sample.videos = self.processed_video(input_sample.videos)

        if input_sample.images is not None:
            output_sample.images = self.process_image(input_sample.images)
        
        output_sample.__key__ = input_sample.__key__
        output_sample.tokens = tokens
        output_sample.labels = labels
        output_sample.loss_mask = loss_mask
        if output_sample.images is not None:
            output_sample.num_media_tiles = output_sample.images.shape[0]
            output_sample.attention_mask = torch.ones(len(tokens), dtype=torch.long)
            height = input_sample.images.shape[1]
            width = input_sample.images.shape[2]
            output_sample.image_sizes = torch.tensor([[height, width]], dtype=torch.long)
        return output_sample


class MeidaToTextTaskEncoder(MultiModalTaskEncoder):
    """MeidaToTextTaskEncoder"""

    def __init__(self, tokenizer, audio_processor, image_processor, media_to_text_sample_config):
        """
        Initialize the MeidaToTextTaskEncoder.

        This encoder extends MultiModalTaskEncoder to specifically handle LlavaNeXT,
        overriding  encoders for VQA sample type.

        Parameters:
        tokenizer (Tokenizer): The tokenizer for processing text data across sample types.
        image_processor (ImageProcessor): The image processor for preprocessing images.
        media_to_text_sample_config (MediaToTextSampleConfig): Configuration settings for multimodal samples.
        """
        super().__init__(tokenizer, image_processor, media_to_text_sample_config)
        self.encoders: Dict[str, SampleEncoder] = {
            MediaToTextSample.__name__: MediaToTextSampleEncoder(
                tokenizer, 
                audio_processor, 
                image_processor, 
                media_to_text_sample_config
            )
        }

    def batch(self, samples: List[MediaToTextSample]) -> MediaToTextRawBatch:
        """
        Batch multiple encoded samples into a single batch structure for model input.

        This method combines individual sample fields (keys, images, tokens, labels, etc.) and
        pads or stacks them as needed to create a unified batch.

        Parameters:
        samples (List[MediaToTextSample]): A list of MediaToTextSample instances to be batched.

        Returns:
        MediaToTextRawBatch: A batch containing all input samples' images, tokens, labels,
            loss masks, and other metadata prepared for model processing.
        """
        keys, audios, videos, images, tokens, labels, loss_mask, num_media_tiles, image_sizes, attention_mask = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for sample in samples:
            keys.append(sample.__key__)
            tokens.append(sample.tokens)
            labels.append(sample.labels)
            loss_mask.append(sample.loss_mask)
            if sample.audios is not None:
                audios.append(sample.audios)
            if sample.videos is not None:
                videos.append(sample.videos)
            if sample.images is not None:
                images.append(sample.images)
                num_media_tiles.append(sample.num_media_tiles)
                image_sizes.append(sample.image_sizes)
                attention_mask.append(sample.attention_mask)

        rawBatch = MediaToTextRawBatch()
        rawBatch.__keys__ = batch_list(keys)
        rawBatch.tokens = pad_sequence(tokens, batch_first=True)
        rawBatch.labels = pad_sequence(labels, batch_first=True)        
        rawBatch.loss_mask = batch_pad_stack(loss_mask)

        if audios:
            rawBatch.audios = torch.cat(audios, dim=0)
        if videos:
            rawBatch.videos = torch.cat(videos, dim=0)
        if images:
            rawBatch.images = torch.cat(images, dim=0)
            rawBatch.image_sizes = torch.cat(image_sizes, dim=0)
            rawBatch.attention_mask = batch_pad_stack(attention_mask)
            rawBatch.num_media_tiles = torch.tensor(batch_list(num_media_tiles), dtype=torch.int)
        
        return rawBatch
