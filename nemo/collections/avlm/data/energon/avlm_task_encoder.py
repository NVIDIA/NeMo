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

import io
import av
import itertools
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, List, Union

import torch
import torchvision
from megatron.core import parallel_state
from megatron.energon import Sample
from megatron.energon import batch_list, batch_pad_stack
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.avlm.data.energon.avlm_sample_config import (
    AudioSize,
    VideoSize,
    AVLMEnergonInterleavedSample,
    AVLMEnergonQASample,
    AVLMSample,
    AVLMRawBatch,
    AVLMSampleConfig,
)

from nemo.collections.multimodal.data.energon.sample_encoder import (
    _find_pattern_indices, 
    SampleEncoder, 
    BaseSampleEncoder,
    VQASampleEncoder,
)
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations as audio_process_augmentations

from nemo.utils import logging


class AVLMSampleEncoder(BaseSampleEncoder):
    """AVLMSampleEncoderInterleaved"""

    def __init__(
        self, 
        tokenizer=None, 
        audio_processor=None,
        image_processor=None, 
        avlm_sample_config=AVLMSampleConfig()
    ):
        """
        Initialize the AVLMSampleEncoder

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        avlm_sample_config (AVLMSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to AVLMSampleConfig().
        """
        super().__init__(tokenizer, image_processor, avlm_sample_config)
        self.audio_processor = audio_processor
        self.audio_token = avlm_sample_config.audio_token
        self.video_token = avlm_sample_config.video_token

        if self.tokenizer is None:
            self.tokenizer = self.build_tokenizer(self.avlm_sample_config.model_id)
        if self.audio_processor is None:
            self.audio_processor = self.build_audio_processor(self.avlm_sample_config)
        if self.image_processor is None:
            self.image_processor = self.build_image_processor(self.avlm_sample_config.model_id)

        self.concate_audio_video_tokens = None
        if self.avlm_sample_config.audio_video_tokens_concatenate_pattern == "sequential":
            self.concate_audio_video_tokens = self.concate_audio_video_tokens_sequential
        if self.avlm_sample_config.audio_video_tokens_concatenate_pattern == "audio_video":
            self.concate_audio_video_tokens = self.concate_audio_video_tokens_audio_video
        elif self.avlm_sample_config.audio_video_tokens_concatenate_pattern == "video_audio":
            self.concate_audio_video_tokens = self.concate_audio_video_tokens_video_audio
        elif self.avlm_sample_config.audio_video_tokens_concatenate_pattern == "interleaved_optimal":
            self.concate_audio_video_tokens = concate_audio_video_tokens_interleaved_optimal
        else:
            raise ValueError(f"Unsupported method in audio_video_tokens_concatenate_pattern: "
                "{self.avlm_sample_config.audio_video_tokens_concatenate_pattern}")

    @staticmethod
    def concate_audio_video_tokens_audio_video(tokenized_chunks_audio, tokenized_chunks_video):
        """
        Parameters:
            tokenized_chunks_video: {stream_index: List[self.image_token.token_id]}
            tokenized_chunks_audio: {stream_index: List[self.audio_token.token_id]}
        Returns:
            List of concatenated tokens with all audio tokens followed by all video tokens
        """
        return [t for i in sorted(tokenized_chunks_audio.keys()) for t in tokenized_chunks_audio[i]] + \
            [t for i in sorted(tokenized_chunks_video.keys()) for t in tokenized_chunks_video[i]]

    @staticmethod
    def concate_audio_video_tokens_video_audio(tokenized_chunks_audio, tokenized_chunks_video):
        """
        Parameters:
            tokenized_chunks_video: {stream_index: List[self.image_token.token_id]}
            tokenized_chunks_audio: {stream_index: List[self.audio_token.token_id]}
        Returns:
            List of concatenated tokens with all video tokens followed by all audio tokens
        """
        return [t for i in sorted(tokenized_chunks_video.keys()) for t in tokenized_chunks_video[i]] + \
            [t for i in sorted(tokenized_chunks_audio.keys()) for t in tokenized_chunks_audio[i]]

    @staticmethod
    def concate_audio_video_tokens_sequential(tokenized_chunks_audio, tokenized_chunks_video):
        """
        Parameters:
            tokenized_chunks_video: {stream_index: List[self.image_token.token_id]}
            tokenized_chunks_audio: {stream_index: List[self.audio_token.token_id]}
        Returns:
            List of concatenated tokens according to the stream index in the original video file
        """
        indexes = set(list(tokenized_chunks_audio.keys()) + list(tokenized_chunks_video.keys()))
        tokens = []
        for index in sorted(indexes):
            if index in tokenized_chunks_video:
                tokens.extend(tokenized_chunks_video[index])
            # ideally, video and audio stream index should be unique.
            # if they share identical index, add the video tokens first
            if index in tokenized_chunks_audio:
                tokens.extend(tokenized_chunks_audio[index])
        return tokens

    @staticmethod
    def concate_audio_video_tokens_interleaved_optimal(tokenized_chunks_audio, tokenized_chunks_video):
        """
        Parameters:
            tokenized_chunks_video: {stream_index: List[self.image_token.token_id]}
            tokenized_chunks_audio: {stream_index: List[self.audio_token.token_id]}
        Returns:
            List of concatenated tokens with evenly spaced video and audio tokens
        """
        audio_tokens = [t for i in sorted(tokenized_chunks_audio.keys()) for t in tokenized_chunks_audio[i]]
        # do not flatten the inner list so as to preserve the {stream: {frame: }} structure
        video_tokens = [tokenized_chunks_video[i] for i in sorted(tokenized_chunks_video.keys())]

        total_length = len(video_tokens) + len(audio_tokens)
        shorter, longer = sorted((video_tokens, audio_tokens), key=len)
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

    @staticmethod
    def build_audio_processor(config: AVLMSampleConfig):
        audio_augmentor = audio_process_augmentations(
            config.audio_augmentor,
            global_rank=parallel_state.get_data_parallel_rank(),
            world_size=parallel_state.get_data_parallel_world_size(),
        )
        return WaveformFeaturizer(
            sample_rate=config.audio_sample_rate, 
            int_values=config.audio_waveform_featurizer_int_values,
            audio_augmentor=audio_augmentor,
        ).process

    @staticmethod
    def build_image_processor(model_id):
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        self.image_processor = processor.image_processor

    def process_audio(self, audio: Union[bytes, dict]):
        """
        Process and prepare an audio sample for encoding.

        Parameters:
        audio: The input audio to be processed.

        Returns:
            Tuple[
                The processed audio tensor: torch.Tensor or None,
                The processed audio tensor length: int or None
            ]
        """
        if self.audio_processor is not None:
            audio_bytes = audio
            offset = 0
            duration = 0
            if isinstance(audio, dict):
                audio_bytes = audio["media_value"]
                offset = audio.get("offset")
                duration = audio.get("duration")
            processed_audio = self.audio_processor(
                io.BytesIO(audio_bytes), 
                offset=offset, 
                duration=duration)
            return processed_audio, processed_audio.shape[0]
        else:
            return None, None

    def process_video(self, video: Union[bytes, dict]) -> Dict[int: Dict[str, Uninion[Literal["video", "audio"], List[torch.tensor], Union[AudioSize, VideoSize]]]]:
        """
        Returns:
            {video_stream_index: {"type": Literal["video", "audio"], "data": torch.tensor, "original_size": Union[AudioSize, VideoSize]}}
            audio tensor in "data" is of shape: [audio_length x channel]
            video tensor in "data" is of shape: [frames x num_of_tiles x channel x height x width]
        """
        ret_dict = dict()

        # get all stream information from the file
        video_bytes = video
        offset = 0
        duration = 0
        if isinstance(video, dict):
            video_bytes = video["media_value"]
            offset = video["offset"]
            duration = video["duration"]

        start_seconds = offset
        end_seconds = start_seconds + duration
        container = av.open(io.BytesIO(video))
        media_stream_count = {"audio": 0, "video": 0}
        for stream in container.streams:
            if stream.type not in ["audio", "video"]:
                continue

            frames = []
            stream_start_seconds = stream.time_base * stream.start_time
            stream_end_seconds = stream_start_seconds + stream.time_base * stream.duration
            if stream_start_seconds >= start_seconds and stream_start_seconds < end_seconds:
                # only retrieve the streams whose time spans within the required start and end
                reader = torchvision.io.VideoReader(video_bytes, f"{stream.type}:{media_stream_count[stream.type]}")
                for frame in itertools.takewhile(
                    lambda x: x['pts'] <= min(stream_end_seconds, end_seconds), reader.seek(start_seconds)):
                    if stream.type == "video" and self.image_processor is not None
                        frame = self.process_image(frame["data"])
                    else:
                        frame = frame["data"]
                    frames.append(frame)
                if frames:                 
                    if stream.type == "video":
                        frames = torch.stack(frames)
                        if frames.dim == 4:
                            frames.unsqueeze(1)                        
                            original_size = VideoSize(
                                frames=stream.frames, 
                                height=stream.height, 
                                width=stream.width)
                    elif stream.type == "audio":
                        frames = torch.cat(frames)
                        # TODO: verify duration is the same as total frame size
                        original_size = AudioSize(length=stream.duration, channel=stream.codec_context.channel)
                    
                    ret_dict[stream.index] = {"type": stream.type, "data": frames, "original_size": original_size}
            
            media_stream_count[stream.type] = media_stream_count[stream.type] + 1                    
            
        return ret_dict

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
        avlm_sample_config=AVLMSampleConfig()
    ):
        """
        Initialize the AVLMSampleEncoderInterleaved

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        avlm_sample_config (AVLMSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to AVLMSampleConfig().
        """
        super().__init__(tokenizer, audio_processor, image_processor, avlm_sample_config)

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
                if processed_image is not None:
                    images.append(processed_image)
                    num_image_tiles.append(processed_image.shape[0])
                    image_sizes.append([chunk.shape[1], chunk.shape[2]])
            elif isinstance(chunk, dict):
                media_type = chunk["media_type"]
                if media_type == "audio":
                    # process audio
                    processed_audio, _ = self.process_audio(chunk)
                    if processed_audio is not None:
                        audios.append(processed_audio)
                        audio_lengths.append(processed_audio.shape[0])
                        tokenized_chunks.append(self.audio_token.token_id)
                elif media_type == "video":
                    audio_stream_index_tokens_dict = {}
                    video_stream_index_tokens_dict = {}
                    # process video and audio (if any) streams in a video file
                    video_audio_dict = self.process_video(chunk)
                    for stream_index in video_audio_dict:
                        media_data = video_audio_dict[stream_index]

                        if media_data["type"] == "video":
                            ## process each video stream
                            processed_video = media_data["data"]
                            original_video_size = video_audio_dict[stream_index]["original_size"]                            
                            # flatten the frames into tiles
                            images.append(processed_video.flatten(end_dim=1))
                            num_image_tiles.extend([processed_video.shape[1]] * processed_video.shape[0])
                            image_sizes.extend([original_video_size.height, original_video_size.width] * processed_video.shape[0])
                            video_stream_index_tokens_dict[stream_index] = [self.image_token.token_id] * processed_video.shape[0]
                        else:
                            ## process each audio stream
                            processed_audio = media_data["data"]
                            audios.append(processed_audio)
                            audio_lengths.append(processed_audio.shape[0])
                            audio_stream_index_tokens_dict[stream_index] = [self.audio_token.token_id]

                    
                    # concatenate the video and audio tokens according to the required pattern
                    tokenized_chunks.extend(self.concate_audio_video_tokens(audio_stream_index_tokens_dict, video_stream_index_tokens_dict))                    
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
        logging.debug(f"[Energon] task encoder encode_sample conversation_prompt {conversation_prompt}")
        # tokenize prompt
        media_dict = self.tokenize(input_sample)
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


class AVLMSampleEncoderQA(AVLMSampleEncoder, VQASampleEncoder):
    def __init__(
        self, 
        tokenizer=None, 
        audio_processor=None,
        image_processor=None, 
        avlm_sample_config=AVLMSampleConfig()
    ):
        """
        Initialize the AVLMSampleEncoderQA

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        avlm_sample_config (AVLMSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to AVLMSampleConfig().
        """
        super().__init__(tokenizer, audio_processor, image_processor, avlm_sample_config)
        self.conversation_template_config = avlm_sample_config.conversation_template_config

    def tokenize(self, prompt: str, input_sample: AVLMEnergonQASample) -> torch.Tensor:
        """
        Tokenize the input prompt, replacing special tokens with their IDs.

        This method splits the prompt into chunks based on the presence of special tokens (like <image>)
        and tokenizes each chunk. Special tokens are replaced with their corresponding token IDs.

        Parameters:
        prompt (str): The prompt string to tokenize.

        Returns:
        torch.Tensor: A tensor containing the tokenized prompt.
        """
        # Split the prompt into chunks and track special tokens
        regex_pattern = '(' + '|'.join(re.escape(token) for token in [self.image_token.token_str]) + ')'
        chunks = re.split(regex_pattern, prompt)
        # Tokenize each chunk and replace special tokens with their indices
        tokenized_chunks = []
        input_audio_index = 0
        input_video_index = 0
        input_image_index = 0
        processed_audios = []
        processed_audio_lengths = []
        processed_images = []
        processed_num_image_tiles = []
        processed_image_sizes = []
        processed_audio_tensor = None
        processed_audio_lengths_tensor = None
        processed_image_tensor = None
        processed_num_image_tiles_tensor = None
        processed_image_sizes_tensor = None

        for chunk in chunks:
            if chunk == self.audio_token.token_str:
                tokenized_chunks.append(self.audio_token.token_str)
                # process the corresponding audio bytes
                processed_audio, _ = self.process_audio(input_sample.audios[input_audio_index])
                if processed_audio is not None:
                    processed_audios.append(processed_audio)
                    processed_audio_lengths.append(processed_audio.shape[0])
                input_audio_index = input_audio_index + 1
            elif chunk == self.video_token.token_str:
                total_frames_in_each_processed_video = []
                audio_stream_index_tokens_dict = {}
                video_stream_index_tokens_dict = {}
                # process video and audio (if any) streams in a video file
                video_audio_dict = self.process_video(chunk)
                for stream_index in video_audio_dict:
                    media_data = video_audio_dict[stream_index]

                    if media_data["type"] == "video":
                        ## process each video stream
                        processed_video = media_data["data"]
                        original_video_size = video_audio_dict[stream_index]["original_size"]
                        # flatten the frames into tiles
                        processed_images.append(processed_video.flatten(end_dim=1))
                        total_frames_in_each_processed_video.append(processed_video.shape[0])
                        processed_num_image_tiles.extend([processed_video.shape[1]] * processed_video.shape[0])
                        processed_image_sizes.extend([original_video_size.height, original_video_size.width] * processed_video.shape[0])
                        video_stream_index_tokens_dict[stream_index] = [self.image_token.token_id] * processed_video.shape[0]
                    else:
                        ## process each audio stream
                        processed_audio = media_data["data"]
                        processed_audios.append(processed_audio)
                        processed_audio_lengths.append(processed_audio.shape[0])
                        audio_stream_index_tokens_dict[stream_index] = [self.audio_token.token_id]
                
                # concatenate the video and audio tokens according to the required pattern
                tokenized_chunks.extend(self.concate_audio_video_tokens(audio_stream_index_tokens_dict, video_stream_index_tokens_dict))

                input_video_index = input_video_index + 1
            elif chunk == self.image_token.token_str:
                tokenized_chunks.append(self.image_token.token_id)
                # process the corresponding image
                processed_image = self.process_image(input_sample.audios[input_image_index])
                if processed_image is not None:
                    processed_images.append(processed_image)
                input_image_index = input_image_index + 1
            elif len(chunk) > 0:
                tokenized_chunks.extend(self.tokenizer(chunk, add_special_tokens=False).input_ids)

        tokens = torch.tensor(tokenized_chunks, dtype=torch.long)
        logging.debug(f"Multimodal dataloader encode interleaved sample tokenized chunks {tokenized_chunks}")

        if processed_audios:
            processed_audio_tensor = torch.concatenate(processed_audios)
            processed_audio_lengths_tensor = torch.tensor(processed_audio_lengths)
        if images:
            processed_image_tensor = torch.concatenate(processed_images)  # T c h w
            processed_num_image_tiles_tensor = torch.tensor(processed_num_image_tiles)
            processed_image_sizes_tensor = torch.tensor(processed_image_sizes)
        return {
            "tokens": tokens, 
            "audios": processed_audio_tensor, 
            "audio_lengths": processed_audio_lengths_tensor, 
            "images": processed_image_tensor,
            "num_image_tiles": processed_num_image_tiles_tensor,
            "image_sizes": processed_image_sizes_tensor,
        }

    def encode(self, input_sample: AVLMEnergonQASample, output_sample: AVLMSample):
        """
        Encode a single sample into a format suitable for model input.

        Parameters:

        Returns:
        AVLMSample: 
        """
        conversation_prompt = self.apply_prompt_template(input_sample)
        logging.debug(f"[Energon] task encoder encode_sample conversation_prompt {conversation_prompt}")
        # tokenize prompt
        media_dict = self.tokenize(conversation_prompt, input_sample)
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
        avlm_sample_config (AVLMSampleConfig): Configuration settings for multimodal samples.
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
