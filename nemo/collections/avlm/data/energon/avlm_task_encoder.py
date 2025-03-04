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

import itertools
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from megatron.energon import Sample
from megatron.energon import batch_list, batch_pad_stack
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.avlm.data.energon.avlm_sample_config import (
    AudioSize,
    VideoSize,
    MediaDict,
    AVLMEnergonInterleavedSample,
    AVLMEnergonQASample,
    AVLMSample,
    AVLMRawBatch,
    AVLMSampleConfigInterleaved,
)

from nemo.collections.multimodal.data.energon.sample_encoder import SampleEncoder, BaseSampleEncoder
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.utils import logging


class AVLMSampleEncoder(BaseSampleEncoder):
    """AVLMSampleEncoderInterleaved"""

    def __init__(
        self, 
        tokenizer=None, 
        audio_processor=None, 
        image_processor=None, 
        multimodal_sample_config=AVLMSampleConfigInterleaved()
    ):
        """
        Initialize the AVLMSampleEncoder

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        avlm_sample_config (AVLMSampleConfigInterleaved, optional): Configuration object for multimodal samples.
            Defaults to AVLMSampleConfigInterleaved().
        """
        super().__init__(tokenizer, image_processor, multimodal_sample_config)
        if self.tokenizer is None:
            self.tokenizer = self.build_tokenizer(self.multimodal_sample_config.model_id)
        if self.audio_processor is None:
            self.audio_processor = self.build_audio_processor()
        if self.image_processor is None:
            self.image_processor = self.build_image_processor(self.multimodal_sample_config.model_id)

        self.concate_video_audio_tokens = None
        if self.multimodal_sample_config.video_audio_token_concatenate_pattern == "video_audio":
            self.concate_video_audio_tokens = self.concate_video_audio_tokens_video_audio
        elif self.multimodal_sample_config.video_audio_token_concatenate_pattern == "audio_video":
            self.concate_video_audio_tokens = self.concate_video_audio_tokens_audio_video
        elif self.multimodal_sample_config.video_audio_token_concatenate_pattern == "interleaved_optimal":
            self.concate_video_audio_tokens = concate_video_audio_tokens_interleaved_optimal
        else:
            raise ValueError(f"Unsupported method in video_audio_token_concatenate_pattern: "
                "{self.multimodal_sample_config.video_audio_token_concatenate_pattern}")

    @staticmethod
    def concate_video_audio_tokens_video_audio(tokenized_chunks_video, tokenized_chunks_audio):
        """
        Parameters:
            tokenized_chunks_video: List of List of frame/image tokens, with inner List 
                corresponds to the sequence of frames of one video stream
            tokenized_chunks_audio: List of audio tokens, each corresponds to one audio stream
        Returns:
            List of concatenated tokens with video streams tokens are before the audio streams tokens 
        """
        return [f for v in tokenized_chunks_video for f in v] + tokenized_chunks_audio

    @staticmethod
    def concate_video_audio_tokens_audio_video(tokenized_chunks_video, tokenized_chunks_audio):
        """
        Parameters:
            tokenized_chunks_video: List of List of frame/image tokens, with inner List 
                corresponds to the sequence of frames of one video stream
            tokenized_chunks_audio: List of audio tokens, each corresponds to one audio stream
        Returns:
            List of concatenated tokens with audio streams tokens are before the video streams tokens
        """
        return tokenized_chunks_audio + [f for v in tokenized_chunks_video for f in v]

    @staticmethod
    def concate_video_audio_tokens_interleaved_optimal(tokenized_chunks_video, tokenized_chunks_audio):
        """
        Parameters:
            tokenized_chunks_video: List of List of frame/image tokens, with inner List 
                corresponds to the sequence of frames of one video stream
            tokenized_chunks_audio: List of audio tokens, each corresponds to one audio stream
        Returns:
            List of concatenated tokens with evenly spaced video and audio tokens
        """
        total_length = len(tokenized_chunks_video) + len(tokenized_chunks_audio)
        shorter, longer = sorted((tokenized_chunks_video, tokenized_chunks_audio), key=len)
        groups = itertools.groupby(
            (
                (
                    longer[len(longer) * i // total_length], 
                    shorter[len(shorter) * i // total_length]
                ) for i in range(total_length)
            ), key=lambda x:x[0]
        )
        interleaved_optimal = [j[i] for _,g in groups for i,j in enumerate(g)]
        for item in interleaved_optimal:
            if isinstance(item, list):
                tokenized_chunks.extend(item)
            else:
                tokenized_chunks.append(item)

        return tokenized_chunks

    @staticmethod
    def build_tokenizer(model_id):
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
        self.tokenizer = AutoTokenizer(model_id)

    def build_audio_processor(self):
        return None

    @staticmethod
    def build_image_processor(model_id):
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        self.image_processor = processor.image_processor

    def process_audio(self, audio: bytes):
        """
        Process and prepare an audio sample for encoding.

        Parameters:
        audio: The input audio to be processed.

        Returns:
        torch.Tensor: The processed audio tensor.
        """
        #TODO
        return None

    def process_video(self, video: bytes):
        #TODO
        """
        Returns:
            dict[
                "video": List of video streams of tuple:
                        (processed video stream tensor of shape: [num_of_tiles x frames x channel x height x width]
                        , original VideoSize)
                "audio": List of audio streams of tuple:
                        (processed audio stream tensor of shape: [audio_length x channel]
                        , original AudioSize)
            ]
        """
        return {"video": [], "audio": []}

    def process_image(self, image: Image.Image):
        """
        Process and prepare an image sample for encoding.

        This method preprocesses the image using the HF image_processor, converting it to
        a tensor.

        Parameters:
        image (torch.Tensor): A tensor representing the input image with dimensions (channels, height, width).

        Returns:
        torch.Tensor: The processed image tensor.
        """
        return self.image_processor.preprocess(image, return_tensors='pt', do_rescale=False)['pixel_values'][0] \
            if self.image_processor is not None else None


class AVLMSampleEncoderInterleaved(AVLMSampleEncoder):
    """AVLMSampleEncoderInterleaved"""

    def __init__(
        self, 
        tokenizer=None, 
        audio_processor=None, 
        image_processor=None, 
        multimodal_sample_config=AVLMSampleConfigInterleaved()
    ):
        """
        Initialize the AVLMSampleEncoderInterleaved

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        avlm_sample_config (AVLMSampleConfigInterleaved, optional): Configuration object for multimodal samples.
            Defaults to AVLMSampleConfigInterleaved().
        """
        super().__init__(tokenizer, audio_processor, image_processor, multimodal_sample_config)

    def tokenize(self, sequence: AVLMEnergonInterleavedSample):
        """
        Tokenize the input sequence and process images in an interleaved sample.

        This method processes a sequence that consists of text strings, audio bytes, video bytes and PIL images.
        The text is tokenized, and the audio, video and images are processed. The method returns a tensor
        of tokenized text and a concatenated tensor of processed images.

        Parameters:
        sample (InterleavedSample): The interleaved sample containing a sequence of text strings and image tensors.

        Returns:
        dict["token": torch.Tensor, 
            "audios": torch.Tensor, 
            "audio_lengths": torch.Tensor, 
            "images": torch.Tensor, 
            "num_image_tiles": torch.Tensor]

        """
        # sample.sequence is a list consisting of text string or image tensor (only image modality supported for now)
        tokenized_chunks = []
        audios = []
        audio_lengths = []
        images = []
        num_image_tiles = []
        image_sizes = []

        audio_tensor = None
        audio_lengths_tensor = None
        image_tensor = None
        num_image_tiles_tensor = None
        image_sizes_tensor = None

        for chunk in sample.sequence:
            if isinstance(chunk, Image.Image):
                # process image
                tokenized_chunks.append(self.image_token.token_id)
                processed_image = self.process_image(chunk)
                images.append(processed_image)
                num_image_tiles.append(processed_image.shape[0])
                image_sizes.append([chunk.shape[1], chunk.shape[2]])
            elif isinstance(chunk, MediaDict):
                media_type = chunk.pop["type"]
                if media_type == "audio":
                    # process audio
                    tokenized_chunks.append(self.audio_token.token_id)
                    processed_audio, _ = self.process_audio(chunk["value"], **chunk)
                    audios.append(processed_audio)
                    audio_lengths.append(processed_audio.shape[0])
                elif media_type == "video":
                    total_frames_in_each_processed_video = []
                    # process video
                    video_audio_dict = self.process_video(chunk["value"], **chunk)
                    ## process each video stream
                    processed_video_streams = video_audio_dict["video"][0]
                    for i, (processed_video, video_size) in enumerate(processed_video_streams):
                        # flatten the frames into tiles
                        images.append(processed_video.flatten(end_dim=1))
                        total_frames_in_each_processed_video.append(processed_video.shape[1])
                        num_image_tiles.extend([processed_video.shape[0]] * processed_video.shape[1])
                        image_sizes.extend([video_size.height, video_size.width] * processed_video.shape[1])
                    ## process each audio stream
                    processed_audio_streams = video_audio_dict["audio"][0]
                    for i, (processed_audio, _) in enumerate(processed_audio_streams):
                        tokenized_chunks.append(self.audio_token.token_id)
                        audios.append(processed_audio)
                        audio_lengths.append(processed_audio.shape[0])
                    
                    # concatenate the video and audio tokens according to the required pattern
                    tokenized_chunks_video = [[self.image_token.token_id] * f for f in total_frames_in_each_processed_video]
                    tokenized_chunks_audio = [self.audio_token.token_id] * len(processed_audio_streams)
                    tokenized_chunks.extend(self.concate_video_audio_tokens(tokenized_chunks_video, tokenized_chunks_audio))
                    
                else:
                    raise ValueError(f"Unsupported type in MediaDict: {type(chunk)}")    
            elif len(chunk) > 0:
                logging.debug(f"Multimodal datalaoder encoder interleaved sample text chunk {chunk}")
                tokenized_chunks.extend(self.tokenizer(chunk, add_special_tokens=False).input_ids)
            else:
                raise ValueError(f"Unsupported type in interleaved sequence: {type(chunk)}")
        tokens = torch.tensor(tokenized_chunks, dtype=torch.long)
        logging.debug(f"Multimodal dataloader encode interleaved sample tokenized chunks {tokenized_chunks}")

        if audios:
            audio_tensor = torch.concatenate(audios)
            audio_lengths_tensor = torch.tensor(audio_lengths)
        if images:
            image_tensor = torch.concatenate(images)  # T c h w
            num_image_tiles_tensor = torch.tensor(num_image_tiles)
            image_sizes_tensor = torch.tensor(image_sizes)
        return {
            "tokens": tokens, 
            "audios": audio_tensor, 
            "audio_lengths": audio_lengths_tensor, 
            "images": image_tensor,
            "num_image_tiles": num_image_tiles_tensor,
            "image_sizes": image_sizes_tensor,
        }

    def encode(self, input_sample: AVLMEnergonInterleavedSample, output_sample: AVLMSample):
        """
        Encode a single sample into a format suitable for model input.

        Parameters:

        Returns:
        AVLMSample: 
        """
        conversation_prompt = self.apply_prompt_template(input_sample)
        logging.debug(f"[Energon] task encoder encode_sample conversation_prompt {conversation_prompt}")
        # tokenize prompt
        media_dict = self.tokenize(conversation_prompt)
        tokens = media_dict["tokens"]
        output_sample.audios = media_dict["audios"]
        output_sample.audio_lengths = media_dict["audio_lengths"]
        output_sample.images = media_dict["images"]
        output_sample.num_image_tiles = media_dict["num_image_tiles"]
        output_sample.image_sizes = media_dict["image_sizes"]
        if output_sample.images is not None:
            output_sample.image_attention_mask = torch.ones(len(tokens), dtype=torch.long)

        labels = self.compute_labels(tokens, input_sample)
        tokens = tokens[:-1].contiguous()
        labels = labels[1:].contiguous()
        logging.debug(f"[Energon] task encoder encode_sample after tokenize prompt tokens {tokens}")
        logging.debug(f"[Energon] task encoder encode_sample lables {labels}")
        loss_mask = self.compute_loss_mask(labels)
        
        output_sample.__key__ = input_sample.__key__
        output_sample.tokens = tokens
        output_sample.labels = labels
        output_sample.loss_mask = loss_mask
        return output_sample


class AVLMSampleEncoderQA(AVLMSampleEncoder):
    def __init__(
        self, 
        tokenizer=None, 
        audio_processor=None, 
        image_processor=None, 
        multimodal_sample_config=AVLMSampleConfigInterleaved()
    ):
        """
        Initialize the AVLMSampleEncoderQA

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        avlm_sample_config (AVLMSampleConfigInterleaved, optional): Configuration object for multimodal samples.
            Defaults to AVLMSampleConfigInterleaved().
        """
        super().__init__(tokenizer, audio_processor, image_processor, multimodal_sample_config)
        self.conversation_template_config = multimodal_sample_config.conversation_template_config

    def apply_prompt_template(self, input_text: AVLMEnergonQASample, use_plain=False):
        """
        Apply a conversation template to the input text for VQA.

        This method generates a templated prompt by combining system, user, and assistant messages.

        Parameters:
        input_text (AVLMEnergonQASample): The sample containing the context and answer.
        use_plain (bool, optional): Whether to use a plain format for the prompt. Defaults to False.

        Returns:
        str: The generated templated prompt as a string.
        """
        logging.debug(f"apply_conversation_template context {input_text.context} answer {input_text.answers}")

        messages = []

        # Add system message if it exists
        if self.conversation_template_config.system:
            messages.append({'role': 'system', 'content': self.conversation_template_config.system})

        # Handle cases where context and answers are lists
        if isinstance(input_text.context, list) and isinstance(input_text.answers, list):
            # Ensure both lists are the same length or adjust based on your specific needs
            min_length = min(len(input_text.context), len(input_text.answers))
            for i in range(min_length):
                messages.append({'role': self.conversation_template_config.roles[0], 'content': input_text.context[i]})
                messages.append({'role': self.conversation_template_config.roles[1], 'content': input_text.answers[i]})
        elif isinstance(input_text.context, str) and isinstance(input_text.answers, str):
            # Handle single context and answer as strings
            messages.append({'role': self.conversation_template_config.roles[0], 'content': input_text.context})
            messages.append({'role': self.conversation_template_config.roles[1], 'content': input_text.answers})
        else:
            raise ValueError(
                f"VQA Sample context/answers should either be a List[str] or str. Other types not supported"
            )
        # Set the chat template if defined
        if self.conversation_template_config.chat_template:
            self.tokenizer.chat_template = self.conversation_template_config.chat_template
        elif self.tokenizer.chat_template is None:
            raise ValueError(
                "Both tokenizer and conversation template does not have chat template defined. Refer to https://huggingface.co/docs/transformers/main/en/chat_templating"
            )

        # Apply the conversation template to generate the prompt
        templated_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        logging.debug(f"apply prompt template templated_prompt {templated_prompt}")
        return templated_prompt


class AVLMTaskEncoder(MultiModalTaskEncoder):
    """MeidaToTextTaskEncoder"""

    def __init__(self, tokenizer=None, audio_processor=None, image_processor=None, avlm_sample_config=None):
        """
        Initialize the MeidaToTextTaskEncoder.

        This encoder extends MultiModalTaskEncoder to specifically handle LlavaNeXT,
        overriding  encoders for VQA sample type.

        Parameters:
        tokenizer (Tokenizer): The tokenizer for processing text data across sample types.
        image_processor (ImageProcessor): The image processor for preprocessing images.
        avlm_sample_config (AVLMSampleConfigInterleaved): Configuration settings for multimodal samples.
        """
        super().__init__(tokenizer, image_processor, avlm_sample_config)
        self.encoders: Dict[str, SampleEncoder] = {
            AVLMEnergonInterleavedSample.__name__: AVLMSampleEncoderInterleaved(
                tokenizer=tokenizer, 
                audio_processor=audio_processor, 
                image_processor=image_processor, 
                avlm_sample_config=avlm_sample_config,
            ),
            AVLMEnergonQASample.__name__: AVLMSampleEncoderQA(
                tokenizer=tokenizer, 
                audio_processor=audio_processor, 
                image_processor=image_processor, 
                avlm_sample_config=avlm_sample_config,
            ),
        }

    def batch(self, samples: List[AVLMSample]) -> AVLMRawBatch:
        """
        Batch multiple encoded samples into a single batch structure for model input.

        This method combines individual sample fields (keys, images, tokens, labels, etc.) and
        pads or stacks them as needed to create a unified batch.

        Parameters:
        samples (List[AVLMSample]): A list of AVLMSample instances to be batched.

        Returns:
        AVLMRawBatch: A batch containing all input samples' images, tokens, labels,
            loss masks, and other metadata prepared for model processing.
            __key__: str = ''
            tokens: 
            labels: 
            loss_mask: 
            audios: [total_audio_length_in_a_batch x channels]
            audio_lengths: [audio_length_of_each_audio_in_a_batch x 1]
            videos: [total_frame_tiles_in_a_batch x channels x tile_height x tile_width]
            video_lengths: [num_of_frames_of_each_video_in_a_batch x 1]
            num_video_tiles: [num_of_tiles_of_each_video_frame_in_a_batch x 1]
            images: [total_image_tiles_in_a_batch x channels x tile_height x tile_width]
            num_image_tiles: [num_of_tiles_of_each_image_in_a_batch x 1]
            image_sizes: [total_images_in_a_batch x 2]
            image_attention_mask: Optional[torch.tensor] = None
        """
        keys, tokens, labels, loss_mask, \
            audios, audio_lengths, \
            videos, video_lengths, num_video_tiles \
            images, num_image_tiles, image_sizes, image_attention_mask = (
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
            if sample.image_attention_mask is not None:
                image_attention_mask.append(sample.image_attention_mask)

        rawBatch = AVLMRawBatch()
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
        if image_attention_mask:
            rawBatch.image_attention_mask = batch_pad_stack(image_attention_mask)            
        
        return rawBatch
