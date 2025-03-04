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
    MediaToMediaSample, 
    MediaToMediaEnergonSample
    MediaToMediaRawBatch,
    MediaToMediaSampleConfig,
)

from nemo.collections.multimodal.data.energon.sample_encoder import BaseSampleEncoder
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.utils import logging


class AVLMSampleEncoder(BaseSampleEncoder):
    """LlavaNextSampleEncoder"""

    def __init__(self, tokenizer, audio_processor, image_processor, multimodal_sample_config=MediaToMediaSampleConfig()):
        """
        Initialize the LlavaNextSampleEncoder, inherited from LlavaNextSampleEncoder for multimodal samples
        focused on LLaVANeXT data to support 

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        media_to_text_sample_config (MediaToMediaSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to MediaToMediaSampleConfig().
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

    def process_image(self, image):
        """
        Process and prepare an image sample for encoding.

        This method preprocesses the image using the HF image_processor, converting it to
        a tensor.

        Parameters:
        image (torch.Tensor): A tensor representing the input image with dimensions (channels, height, width).

        Returns:
        torch.Tensor: The processed image tensor.
        """
        return self.image_processor.preprocess(image, return_tensors='pt', do_rescale=False)['pixel_values'][0]
        

    @stateless
    def encode_sample(self, sample: MediaToMediaEnergonSample) -> MediaToMediaSample:
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
        encoded_sample = encoder.encode(input_sample=sample, output_sample=MediaToMediaSample())
        return encoded_sample

    def encode(self, input_sample: MediaToMediaEnergonSample, output_sample: MediaToMediaSample):
        """
        Encode a single sample into a format suitable for model input.

        Parameters:

        Returns:
        MediaToMediaSample: 
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
            output_sample.num_image_tiles = output_sample.images.shape[0]
            output_sample.attention_mask = torch.ones(len(tokens), dtype=torch.long)
            height = input_sample.images.shape[1]
            width = input_sample.images.shape[2]
            output_sample.image_sizes = torch.tensor([[height, width]], dtype=torch.long)
        return output_sample


class AVLMTaskEncoder(MultiModalTaskEncoder):
    """MeidaToTextTaskEncoder"""

    def __init__(self, tokenizer, audio_processor, image_processor, media_to_text_sample_config):
        """
        Initialize the MeidaToTextTaskEncoder.

        This encoder extends MultiModalTaskEncoder to specifically handle LlavaNeXT,
        overriding  encoders for VQA sample type.

        Parameters:
        tokenizer (Tokenizer): The tokenizer for processing text data across sample types.
        image_processor (ImageProcessor): The image processor for preprocessing images.
        media_to_text_sample_config (MediaToMediaSampleConfig): Configuration settings for multimodal samples.
        """
        super().__init__(tokenizer, image_processor, media_to_text_sample_config)
        self.encoders: Dict[str, SampleEncoder] = {
            MediaToMediaSample.__name__: MediaToMediaSampleEncoder(
                tokenizer, 
                audio_processor, 
                image_processor, 
                media_to_text_sample_config
            )
        }

    def batch(self, samples: List[MediaToMediaSample]) -> MediaToMediaRawBatch:
        """
        Batch multiple encoded samples into a single batch structure for model input.

        This method combines individual sample fields (keys, images, tokens, labels, etc.) and
        pads or stacks them as needed to create a unified batch.

        Parameters:
        samples (List[MediaToMediaSample]): A list of MediaToMediaSample instances to be batched.

        Returns:
        MediaToMediaRawBatch: A batch containing all input samples' images, tokens, labels,
            loss masks, and other metadata prepared for model processing.
            __key__: str = ''
            tokens: 
            labels: 
            loss_mask: 
            audios: [total_audio_length_in_a_batch x channels]
            audio_lengths: [audio_length_of_each_audio_in_a_batch x 1]
            videos: [total_image_tiles_in_a_batch x frames x channels x tile_height x tile_width]
            video_lengths: [num_of_frames_of_each_video_in_a_batch x 1]
            num_video_tiles: [num_of_tiles_of_each_video_frame_in_a_batch x 1]
            images: [total_image_tiles_in_a_batch x channels x tile_height x tile_width]
            num_image_tiles: [num_of_tiles_of_each_image_in_a_batch x 1]
            image_sizes: [total_images_in_a_batch x 2]
            attention_mask: Optional[torch.tensor] = None
        """
        keys, tokens, labels, loss_mask, \
            audios, audio_lengths, \
            videos, video_lengths, num_video_tiles \
            images, num_image_tiles, image_sizes, attention_mask = (
            [],
            [],
            [],
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
            if sample.audio_lengths is not None:
                audio_lengths.append(sample.audio_lengths)
            if sample.videos is not None:
                videos.append(sample.videos)
            if sample.video_lengths is not None:
                video_lengths.append(sample.video_lengths)
            if sample.num_video_tiles is not None:
                num_video_tiles.append(sample.num_video_tiles)
            if sample.images is not None:
                images.append(sample.images)
            if sample.num_image_tiles is not None:
                num_image_tiles.append(sample.num_image_tiles)
            if sample.image_sizes is not None:
                image_sizes.append(sample.image_sizes)
            if sample.attention_mask is not None:
                attention_mask.append(sample.attention_mask)

        rawBatch = MediaToMediaRawBatch()
        rawBatch.__keys__ = batch_list(keys)
        rawBatch.tokens = pad_sequence(tokens, batch_first=True)
        rawBatch.labels = pad_sequence(labels, batch_first=True)        
        rawBatch.loss_mask = batch_pad_stack(loss_mask)

        if audios:
            rawBatch.audios = torch.cat(audios)
        if audio_lengths
            rawBatch.audio_lengths = torch.tensor(batch_list(audio_lengths), dtype=torch.int)
        if videos:
            rawBatch.videos = torch.cat(videos)
        if video_lengths:
            rawBatch.video_lengths = torch.tensor(batch_list(video_lengths), dtype=torch.int)
        if num_video_tiles:
            rawBatch.num_video_tiles = torch.tensor(batch_list(num_video_tiles), dtype=torch.int)
        if images:
            rawBatch.images = torch.cat(images)
        if num_image_tiles:
            rawBatch.num_image_tiles = torch.tensor(batch_list(num_image_tiles), dtype=torch.int)
        if image_sizes:
            rawBatch.image_sizes = torch.cat(image_sizes)
        if attention_mask:
            rawBatch.attention_mask = batch_pad_stack(attention_mask)            
        
        return rawBatch
