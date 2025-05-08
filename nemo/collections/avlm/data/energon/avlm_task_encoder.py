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
import re
import itertools
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Union, Literal, TypedDict, Callable, Optional

import torch
import torchvision
from torch.nn.utils.rnn import pad_sequence

from megatron.core import parallel_state
from megatron.energon import batch_list, batch_pad_stack
from megatron.energon.task_encoder.base import stateless

from nemo.collections.avlm.data.energon.avlm_sample_config import (
    AudioSize,
    VideoSize,
    ImageSize,
    MediaDict,
    AVLMEnergonInterleavedSample,
    AVLMEnergonQASample,
    AVLMSample,
    PackedAVLMSample,
    AVLMRawBatch,
    PackedAVLMRawBatch,
    AVLMSampleConfig,
)

from nemo.collections.avlm.data.energon.calculate_media_seq_length import (
    calculate_encoded_audio_seq_length,
    calculate_encoded_image_seq_length,
)

from nemo.collections.multimodal.data.energon.sample_encoder import (
    SampleEncoder, 
    BaseSampleEncoder,
    VQASampleEncoder,
)
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.collections.vlm.neva.data.sequence_packing import predict_seq_len_with_padding
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations as audio_process_augmentations

from nemo.utils import logging


class AVLMSampleEncoder(BaseSampleEncoder):
    """AVLMSampleEncoder"""

    def __init__(
        self, 
        tokenizer: Optional[Callable]=None, 
        audio_processor: Optional[
            Union[Callable, Dict[Literal["from_file", "from_decoded"], Callable]]
        ]=None,
        image_processor: Optional[Callable]=None, 
        multimodal_sample_config=AVLMSampleConfig(),
    ):
        """
        Initialize the AVLMSampleEncoder

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        multimodal_sample_config (AVLMSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to AVLMSampleConfig().
        """
        super().__init__(tokenizer, image_processor, multimodal_sample_config)
        self.audio_processor = {}
        if audio_processor is not None:
            if not isinstance(audio_processor, dict):
                self.audio_processor["from_file"] = self.audio_processor["from_decoded"] = audio_processor
            else:
                self.audio_processor = audio_processor
        self.audio_token = multimodal_sample_config.audio_token
        self.video_token = multimodal_sample_config.video_token

        if self.tokenizer is None:
            self.build_tokenizer()
        if self.audio_processor == {}:
            self.build_audio_processor()
        if self.image_processor is None:
            self.build_image_processor()

        self.concate_audio_video_tokens = None
        if self.multimodal_sample_config.audio_video_tokens_concatenate_pattern == "sequential":
            self.concate_audio_video_tokens = self.concate_audio_video_tokens_sequential
        elif self.multimodal_sample_config.audio_video_tokens_concatenate_pattern == "audio_video":
            self.concate_audio_video_tokens = self.concate_audio_video_tokens_audio_video
        elif self.multimodal_sample_config.audio_video_tokens_concatenate_pattern == "video_audio":
            self.concate_audio_video_tokens = self.concate_audio_video_tokens_video_audio
        elif self.multimodal_sample_config.audio_video_tokens_concatenate_pattern == "interleaved_optimal":
            self.concate_audio_video_tokens = concate_audio_video_tokens_interleaved_optimal
        else:
            raise ValueError(f"Unsupported method in audio_video_tokens_concatenate_pattern: "
                "{self.multimodal_sample_config.audio_video_tokens_concatenate_pattern}")

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

    def build_tokenizer(self):
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
        self.tokenizer = AutoTokenizer(self.multimodal_sample_config.model_id).tokenizer

    def build_audio_processor(self):
        self.audio_augmentor = audio_process_augmentations(
            self.multimodal_sample_config.audio_augmentor,
            global_rank=parallel_state.get_data_parallel_rank(),
            world_size=parallel_state.get_data_parallel_world_size(),
        )

        featurizer = WaveformFeaturizer(
            sample_rate=self.multimodal_sample_config.audio_sample_rate, 
            int_values=self.multimodal_sample_config.audio_waveform_featurizer_int_values,
            augmentor=self.audio_augmentor,
        )
        
        self.audio_processor["from_file"] = featurizer.process
        self.audio_processor["from_decoded"] = self._process_audio_from_decoded 

    def build_image_processor(self):
        from transformers import AutoProcessor
        self.image_processor = AutoProcessor.from_pretrained(self.multimodal_sample_config.model_id).image_processor

    def _process_audio_from_decoded(self, audio: npt.NDArray, sample_rate, **kwargs) -> npt.NDArray:
        samples = AudioSegment(audio, sample_rate, **kwargs)
        if self.audio_augmentor is not None:
            samples = self.audio_augmentor.perturb(samples)
        return torch.tensor(samples.samples, dtype=torch.float)

    def process_audio(self, audio: MediaDict, mode: Literal["from_file", "from_decoded"]):
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

        if mode in self.audio_processor:
            audio_data = audio["media_value"]
            if mode == "from_file" and isinstance(audio_data, bytes):
                audio_data = io.BytesIO(audio_data)
            elif mode == "from_decoded" and torch.is_tensor(audio_data):
                audio_data = audio_data.numpy()

            processed_audio = self.audio_processor[mode](
                audio_data, 
                channel_selector=self.multimodal_sample_config.audio_channel_selector, 
                **{k: audio[k] for k in audio if k != "media_type" and k != "media_value"}
            )
            
            return processed_audio, processed_audio.shape[0]
        else:
            return None, None

    class _processed_video_dict(TypedDict):
        media_type: Literal["audio", "video"]
        data: torch.tensor
        original_size: Union[AudioSize, VideoSize]

    def process_video(self, video: MediaDict) -> Dict[int, dict]:
        """
        Returns:
            {video_stream_index: {"media_type": Literal["video", "audio"], "data": torch.tensor, "original_size": Union[AudioSize, VideoSize]}}
            audio tensor in "data" is of shape: [audio_length]
            video tensor in "data" is of shape: [frames x num_of_tiles x channel x height x width]
        """
        ret_dict = dict()

        # get all stream information from the file
        video_bytes = video["media_value"]
        offset = video.get("offset", 0)
        duration = video.get("duration", 0)
        # copy audio processing configs
        audio_sample = {k: video[k] for k in video if k != "media_type" and k != "media_value"}

        start_seconds = offset
        end_seconds = start_seconds + duration if duration != 0 else 0
        container = av.open(io.BytesIO(video_bytes))
        media_stream_count = {"audio": 0, "video": 0}
        for stream in container.streams:
            if stream.type not in ["audio", "video"]:
                continue

            frames = []
            stream_start_seconds = stream.time_base * stream.start_time
            stream_end_seconds = stream_start_seconds + stream.time_base * stream.duration
            if end_seconds == 0:
                end_seconds = stream_end_seconds
            if stream_start_seconds >= start_seconds and stream_start_seconds < end_seconds:
                # only retrieve the streams whose time spans within the required start and end
                reader = torchvision.io.VideoReader(video_bytes, f"{stream.type}:{media_stream_count[stream.type]}")
                for frame in itertools.takewhile(
                    lambda x: x['pts'] <= min(stream_end_seconds, end_seconds), reader.seek(start_seconds)):
                    if stream.type == "video" and self.image_processor is not None:
                        frame, _ = self.process_image(frame["data"])
                    else:
                        frame = frame["data"]
                    frames.append(frame)

                if frames:                 
                    if stream.type == "video":
                        frames = torch.stack(frames)
                        if frames.dim() == 4:
                            frames.unsqueeze(1)                        
                        original_size = VideoSize(
                            frames=stream.frames, 
                            height=stream.height, 
                            width=stream.width)
                    elif stream.type == "audio":
                        frames = torch.cat(frames)
                        audio_sample["media_value"] = frames
                        audio_sample["sample_rate"] = stream.codec_context.sample_rate
                        frames, _ = self.process_audio(audio_sample, "from_decoded")
                        # TODO: verify duration is the same as total frame size
                        original_size = AudioSize(length=stream.duration, channel=stream.codec_context.channels)
                    ret_dict[stream.index] = self._processed_video_dict(media_type=stream.type, data=frames, original_size=original_size)
            
            media_stream_count[stream.type] = media_stream_count[stream.type] + 1                    
            
        return ret_dict

    def process_image(self, image: Union[bytes, MediaDict, torch.Tensor]):
        """
        Process and prepare an image sample for encoding.

        This method preprocesses the image using the HF image_processor, converting it to
        a tensor.

        Parameters:
        image (torch.Tensor): A tensor representing the input image with dimensions (channels, height, width).

        Returns:
        (torch.Tensor: (The processed image tensor, original image size).
        """
        if self.image_processor is not None:
            if isinstance(image, dict):
                image = image["media_value"]
            if isinstance(image, bytes):
                # decode image from bytes
                image = torchvision.io.decode_image(
                    torch.tensor(np.frombuffer(image, dtype=np.uint8)),
                    mode = torchvision.io.ImageReadMode.RGB
                )

            return (self.image_processor.preprocess(image, return_tensors='pt', do_rescale=False)['pixel_values'][0],
                ImageSize(height=image.shape[1], width=image.shape[2]))
        else:
            return None, None


class AVLMSampleEncoderInterleaved(AVLMSampleEncoder):
    """AVLMSampleEncoderInterleaved"""

    def __init__(
        self, 
        tokenizer=None, 
        audio_processor=None,
        image_processor=None, 
        multimodal_sample_config=AVLMSampleConfig()
    ):
        """
        Initialize the AVLMSampleEncoderInterleaved

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        multimodal_sample_config (AVLMSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to AVLMSampleConfig().
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
            if isinstance(chunk, dict):
                media_type = chunk["media_type"]

                if media_type == "audio":
                    # process audio
                    processed_audio, _ = self.process_audio(chunk, "from_file")
                    if processed_audio is not None:
                        audios.append(processed_audio)
                        audio_lengths.append(processed_audio.shape[0])

                        # calculate the encoded audio sequence length from processed audio length
                        # and add the corresponding special audio tokens to the tokenized_chunks
                        encoded_audio_seq_length = calculate_encoded_audio_seq_length(
                            audio_length=processed_audio.shape[0],
                            model_type=self.multimodal_sample_config.audio_encoder_config['model_type'],
                            sample_rate=self.multimodal_sample_config.audio_encoder_config['sample_rate'],
                            window_stride=self.multimodal_sample_config.audio_encoder_config['window_stride'],
                            fixed_max_audio_length=self.multimodal_sample_config.audio_encoder_config['fixed_max_audio_length'],
                            encoder_down_sampling=self.multimodal_sample_config.audio_encoder_config['encoder_down_sampling'],
                            num_mel_bins=self.multimodal_sample_config.audio_encoder_config['num_mel_bins'],
                            patch_size=self.multimodal_sample_config.audio_encoder_config['patch_size'],
                            time_stride=self.multimodal_sample_config.audio_encoder_config['time_stride'],
                            frequency_stride=self.multimodal_sample_config.audio_encoder_config['frequency_stride'],
                            max_spectrogram_length=self.multimodal_sample_config.audio_encoder_config['max_spectrogram_length'],
                            )
                        tokenized_chunks.extend([self.audio_token.token_id] * encoded_audio_seq_length)

                elif media_type == "video":
                    audio_stream_index_tokens_dict = {}
                    video_stream_index_tokens_dict = {}
                    # process video and audio (if any) streams in a video file
                    video_audio_dict = self.process_video(chunk)
                    for stream_index in video_audio_dict:
                        media_data = video_audio_dict[stream_index]

                        if media_data["media_type"] == "video":
                            ## process each video stream
                            processed_video = media_data["data"]
                            original_video_size = video_audio_dict[stream_index]["original_size"]                            
                            # flatten the frames into tiles
                            images.append(processed_video.flatten(end_dim=1))
                            num_image_tiles.extend([processed_video.shape[1]] * processed_video.shape[0])
                            image_sizes.extend([[original_video_size.height, original_video_size.width]] * processed_video.shape[0])
                            video_stream_index_tokens_dict[stream_index] = [self.image_token.token_id] * processed_video.shape[0]
                        else:
                            ## process each audio stream
                            processed_audio = media_data["data"]
                            audios.append(processed_audio)
                            audio_lengths.append(processed_audio.shape[0])
                            audio_stream_index_tokens_dict[stream_index] = [self.audio_token.token_id]
                    
                    # concatenate the video and audio tokens according to the required pattern
                    tokenized_chunks.extend(self.concate_audio_video_tokens(audio_stream_index_tokens_dict, video_stream_index_tokens_dict))
                elif media_type == "image":
                    # process image
                    processed_image, original_image_size = self.process_image(chunk)
                    if processed_image is not None:
                        images.append(processed_image)
                        num_image_tiles.append(processed_image.shape[0])
                        image_sizes.append([original_image_size.height, original_image_size.width])

                    # calculate the encoded image sequence length from processed image length
                    # and add the corresponding special image tokens to the tokenized_chunks
                    encoded_image_seq_length = calculate_encoded_image_seq_length(
                        num_one_image_tiles = processed_image.shape[0],
                        model_type=self.multimodal_sample_config.image_encoder_config['model_type'],
                        img_width=self.multimodal_sample_config.image_encoder_config['img_width'],
                        img_height=self.multimodal_sample_config.image_encoder_config['img_height'],
                        patch_size=self.multimodal_sample_config.image_encoder_config['patch_size'],
                        )
                    tokenized_chunks.extend([self.image_token.token_id] * encoded_image_seq_length)

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
        output_sample.attention_mask = torch.ones(len(tokens), dtype=torch.long)

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
        multimodal_sample_config=AVLMSampleConfig(),
    ):
        """
        Initialize the AVLMSampleEncoderQA

        Parameters:
        tokenizer (Tokenizer): The HF tokenizer used for processing text.
        image_processor (ImageProcessor): The HF image processor used for preprocessing images.
        multimodal_sample_config (AVLMSampleConfig, optional): Configuration object for multimodal samples.
            Defaults to AVLMSampleConfig().
        """
        super().__init__(
            tokenizer, 
            audio_processor, 
            image_processor, 
            multimodal_sample_config,
        )
        self.conversation_template_config = multimodal_sample_config.conversation_template_config

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
        regex_pattern = '(' + '|'.join(re.escape(token) for token in [
            self.audio_token.token_str, 
            self.video_token.token_str, 
            self.image_token.token_str
        ]) + ')'
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
                # process the corresponding audio bytes
                processed_audio, _ = self.process_audio(input_sample.audios[input_audio_index], "from_file")
                if processed_audio is not None:
                    processed_audios.append(processed_audio)
                    processed_audio_lengths.append(processed_audio.shape[0])
                input_audio_index = input_audio_index + 1

                # calculate the encoded audio sequence length from processed audio length
                # and add the corresponding special audio tokens to the tokenized_chunks
                encoded_audio_seq_length = calculate_encoded_audio_seq_length(
                    audio_length=processed_audio.shape[0],
                    model_type=self.multimodal_sample_config.audio_encoder_config['model_type'],
                    sample_rate=self.multimodal_sample_config.audio_encoder_config['sample_rate'],
                    window_stride=self.multimodal_sample_config.audio_encoder_config['window_stride'],
                    fixed_max_audio_length=self.multimodal_sample_config.audio_encoder_config['fixed_max_audio_length'],
                    encoder_down_sampling=self.multimodal_sample_config.audio_encoder_config['encoder_down_sampling'],
                    num_mel_bins=self.multimodal_sample_config.audio_encoder_config['num_mel_bins'],
                    patch_size=self.multimodal_sample_config.audio_encoder_config['patch_size'],
                    time_stride=self.multimodal_sample_config.audio_encoder_config['time_stride'],
                    frequency_stride=self.multimodal_sample_config.audio_encoder_config['frequency_stride'],
                    max_spectrogram_length=self.multimodal_sample_config.audio_encoder_config['max_spectrogram_length'],
                    )
                tokenized_chunks.extend([self.audio_token.token_id] * encoded_audio_seq_length)

            elif chunk == self.video_token.token_str:

                total_frames_in_each_processed_video = []
                audio_stream_index_tokens_dict = {}
                video_stream_index_tokens_dict = {}
                # process video and audio (if any) streams in a video file
                video_audio_dict = self.process_video(input_sample.videos[input_video_index])
                for stream_index in video_audio_dict:
                    media_data = video_audio_dict[stream_index]

                    if media_data["media_type"] == "video":
                        ## process each video stream
                        processed_video = media_data["data"]
                        original_video_size = video_audio_dict[stream_index]["original_size"]
                        # flatten the frames into tiles
                        processed_images.append(processed_video.flatten(end_dim=1))
                        total_frames_in_each_processed_video.append(processed_video.shape[0])
                        processed_num_image_tiles.extend([processed_video.shape[1]] * processed_video.shape[0])
                        processed_image_sizes.extend([[original_video_size.height, original_video_size.width]] * processed_video.shape[0])
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
                # process the corresponding image
                processed_image, original_image_size = self.process_image(input_sample.images[input_image_index])
                if processed_image is not None:
                    processed_images.append(processed_image)
                    processed_num_image_tiles.append(processed_image.shape[0])
                    processed_image_sizes.append([original_image_size.height, original_image_size.width])
                input_image_index = input_image_index + 1

                # calculate the encoded image sequence length from processed image length
                # and add the corresponding special image tokens to the tokenized_chunks
                encoded_image_seq_length = calculate_encoded_image_seq_length(
                    num_one_image_tiles = processed_image.shape[0],
                    model_type=self.multimodal_sample_config.image_encoder_config['model_type'],
                    img_width=self.multimodal_sample_config.image_encoder_config['img_width'],
                    img_height=self.multimodal_sample_config.image_encoder_config['img_height'],
                    patch_size=self.multimodal_sample_config.image_encoder_config['patch_size'],
                    )
                tokenized_chunks.extend([self.image_token.token_id] * encoded_image_seq_length)

            elif len(chunk) > 0:
                tokenized_chunks.extend(self.tokenizer(chunk, add_special_tokens=False).input_ids)

        tokens = torch.tensor(tokenized_chunks, dtype=torch.long)
        logging.debug(f"Multimodal dataloader encode interleaved sample tokenized chunks {tokenized_chunks}")

        if processed_audios:
            processed_audio_tensor = processed_audios

            processed_audio_lengths_tensor = torch.tensor(processed_audio_lengths)
        if processed_images:
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

        labels = self.compute_labels(tokens, input_sample)
        tokens = tokens[:-1].contiguous()
        labels = labels[1:].contiguous()
        logging.debug(f"[Energon] task encoder encode_sample after tokenize prompt tokens {tokens}")
        logging.debug(f"[Energon] task encoder encode_sample lables {labels}")
        loss_mask = self.compute_loss_mask(labels)
        output_sample.attention_mask = torch.ones(len(tokens), dtype=torch.long)

        output_sample.__key__ = input_sample.__key__
        output_sample.tokens = tokens
        output_sample.labels = labels
        output_sample.loss_mask = loss_mask

        return output_sample


class AVLMTaskEncoder(MultiModalTaskEncoder):
    """MeidaToTextTaskEncoder"""

    def __init__(self, 
        tokenizer=None, 
        audio_processor=None, 
        image_processor=None, 
        multimodal_sample_config=AVLMSampleConfig(),
        packed_sequence=False,
        packed_sequence_size=-1,
    ):
        """
        Initialize the MeidaToTextTaskEncoder.

        This encoder extends MultiModalTaskEncoder to specifically handle LlavaNeXT,
        overriding  encoders for VQA sample type.

        Parameters:
        tokenizer (Tokenizer): The tokenizer for processing text data across sample types.
        image_processor (ImageProcessor): The image processor for preprocessing images.
        multimodal_sample_config (AVLMSampleConfig): Configuration settings for multimodal samples.
        """
        super().__init__(
            tokenizer, 
            image_processor, 
            multimodal_sample_config,
            packed_sequence,
            packed_sequence_size,
        )
        self.encoders: Dict[str, SampleEncoder] = {
            AVLMEnergonInterleavedSample.__name__: AVLMSampleEncoderInterleaved(
                tokenizer=tokenizer, 
                audio_processor=audio_processor,
                image_processor=image_processor, 
                multimodal_sample_config=multimodal_sample_config,
            ),
            AVLMEnergonQASample.__name__: AVLMSampleEncoderQA(
                tokenizer=tokenizer, 
                audio_processor=audio_processor,
                image_processor=image_processor, 
                multimodal_sample_config=multimodal_sample_config,
            ),
        }

    @stateless
    def encode_sample(self, sample: AVLMEnergonQASample) -> AVLMSample:
        """
        """
        sample_type = type(sample).__name__
        encoder = self.encoders.get(sample_type)
        if not encoder:
            raise NotImplementedError(f"No encoder implemented for sample type {sample_type}")
        encoded_sample = encoder.encode(input_sample=sample, output_sample=AVLMSample())
        return encoded_sample

    def batch(self, samples: Union[List[AVLMSample], List[PackedAVLMSample]]) -> Union[AVLMRawBatch, PackedAVLMRawBatch]:
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
            attention_mask: Optional[torch.tensor] = None
        """
        if self.packed_sequence:
            if len(samples) > 1:
                raise ValueError(
                    "Micro batch size should be 1 when training with packed sequence, but your micro batch size "
                    f"is {len(samples)}. \nThe following config is equivalent to your current setting for "
                    f"a packed dataset. Please update your config to the following: \n"
                    f"Set micro batch size to 1 (currently {len(samples)})\n"
                    f"Set global batch size to `global_batch_size // {len(samples)}` "
                    f"Set packed sequence length to `original_sample_seq_len * {len(samples)}` "
                    f"(currently {self.packed_sequence_size}) \n"
                    f"For details please visit "
                    f"https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/packed_sequence.html"
                )
            # The batching are taken care by packing.
            sample = samples[0]

            return PackedAVLMRawBatch(
                __keys__=sample.__key__,
                images=sample.images,
                audios=sample.audios,
                tokens=sample.tokens,
                labels=sample.labels,
                loss_mask=sample.loss_mask,
                num_image_tiles=sample.num_image_tiles,
                image_sizes=sample.image_sizes,
                audio_lengths=sample.audio_lengths,
                attention_mask=sample.attention_mask,
                position_ids=sample.position_ids,
                packed_seq_params=sample.packed_seq_params,
            )
        else:
            keys, tokens, labels, loss_mask, \
                audios, audio_lengths, \
                videos, video_lengths, num_video_tiles, \
                images, num_image_tiles, image_sizes, \
                attention_mask = (
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

            rawBatch = AVLMRawBatch()
            rawBatch.__keys__ = batch_list(keys)
            rawBatch.tokens = pad_sequence(tokens, batch_first=True) # pad with 0s
            rawBatch.labels = pad_sequence(labels, batch_first=True, padding_value=self.sample_config.ignore_place_holder) # pad with IGNORE tokens  
            rawBatch.loss_mask = batch_pad_stack(loss_mask) # pad with 0s

            if audios:
                audios = [audio for audio_list in audios for audio in audio_list]
                # get the audio samples' maximum length and pad all samples to that length
                rawBatch.audios = pad_sequence(audios, batch_first=True)
                if audio_lengths:                
                    rawBatch.audio_lengths = torch.tensor(torch.cat(audio_lengths), dtype=torch.int)
            if videos:
                rawBatch.videos = torch.cat(videos)
                if video_lengths:
                    rawBatch.video_lengths = torch.tensor(torch.cat(video_lengths), dtype=torch.int)
                if num_video_tiles:
                    rawBatch.num_video_tiles = torch.tensor(torch.cat(num_video_tiles), dtype=torch.int)
            if images:
                rawBatch.images = torch.cat(images)
                if num_image_tiles:
                    rawBatch.num_image_tiles = torch.tensor(torch.cat(num_image_tiles), dtype=torch.int)
                if image_sizes:
                    rawBatch.image_sizes = torch.cat(image_sizes)
            if attention_mask:
                rawBatch.attention_mask = batch_pad_stack(attention_mask)            
            
            return rawBatch

    def encode_batch(self, batch_data: AVLMRawBatch) -> dict:
        """
        Encode a batched set of samples for model input.

        This method transforms the raw batched data into a format ready for model input, including
        generating position IDs and other necessary fields.

        Parameters:
        batch_data (ImageTextRawBatch): The raw batch of data to be encoded.

        Returns:
        dict: A dictionary containing the encoded batch data, ready for model input.
        """
        batch_dict = batch_data.__dict__
        micro_batch_size, seq_length = batch_dict['tokens'].size()
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
        batch_dict['position_ids'] = position_ids
        if 'attention_mask' not in batch_dict:
            batch_dict['attention_mask'] = None
        return batch_dict

    def select_samples_to_pack(self, samples: List[Union[AVLMSample, PackedAVLMSample]]):
        """Selects which samples will be packed together.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html
        """
        from nemo.collections.vlm.neva.data.sequence_packing import greedy_knapsack

        # note: similar to Llava_next, the sample.tokens already take into account the final images/audio tokens, so
        # no need to run predict_seq_len() as in NeVa
        lengths = [predict_seq_len_with_padding(sample.tokens) for sample in samples]


        packed_samples = greedy_knapsack(lengths, samples, self.packed_sequence_size)
        avg_samples_per_bin = round(len(lengths) / len(packed_samples))
        logging.info(
            f"[Seq Packing Info] - Packing seq len: {self.packed_sequence_size}, "
            f"Buffered samples: {len(lengths)}, Total number of bins: {len(packed_samples)}, "
            f"Average samples per bin: {avg_samples_per_bin}"
        )
        return packed_samples

    @stateless
    def pack_selected_samples(self, samples):
        """
        Function to pack a list of ImageTaskSample into a single ImageTaskSamplePacked.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html

        Args:
            samples: List of ImageTaskSample instances to pack into one sample.

        Returns:
            ImageTaskSamplePacked instance.
        """
        # import pdb; pdb.set_trace()

        audio_lengths, num_image_tiles, image_sizes = (
            [],
            [],
            [],
        )

        for sample in samples:
            if sample.audios is not None:
                audio_lengths.append(sample.audio_lengths)
            if sample.images is not None:
                num_image_tiles.append(sample.num_image_tiles)
                image_sizes.append(sample.image_sizes)

        audio_lengths = torch.cat(audio_lengths, dim=0)
        image_sizes = torch.cat(image_sizes, dim=0)
        batch_list_num_image_tiles = batch_list(num_image_tiles)
        # if batch_list_num_media_tiles is nested lists, each sample has multiple images with different tiles
        # we need to flatten the list so len is num_images (in the batch)
        # image_sizes is also expected to be num_images, 2
        batch_list_num_image_tiles = flatten_if_nested(batch_list_num_image_tiles)
        batch_num_image_tiles = torch.tensor(batch_list_num_image_tiles, dtype=torch.int)

        # packing audio and images
        audios = [audio for sample in samples for audio in sample.audios]
        packed_audios = pad_sequence(audios, batch_first=True)
        packed_images = torch.cat([sample.images for sample in samples], dim=0)

        # packing tokens, labels, position ids, loss mask, seq params
        from nemo.collections.vlm.llava_next.data.utils import convert_to_packed_llava_next
        packed_tokens, packed_labels, packed_position_ids, packed_loss_mask, packed_seq_params = (
            convert_to_packed_llava_next(
                tokens=[sample.tokens for sample in samples],
                labels=[sample.labels for sample in samples],
                ignore_index=self.sample_config.ignore_place_holder,
            )
        )

        return PackedAVLMSample(
            __key__=",".join([s.__key__ for s in samples]),
            __restore_key__=(),  # Will be set by energon based on `samples`
            images=packed_images,
            audios=packed_audios,
            tokens=packed_tokens,
            labels=packed_labels,
            loss_mask=packed_loss_mask,
            attention_mask=None,
            position_ids=packed_position_ids,
            packed_seq_params=packed_seq_params,
            num_image_tiles=batch_num_image_tiles,
            image_sizes=image_sizes,
            audio_lengths=audio_lengths,
        )

from itertools import chain
def flatten_if_nested(lst):
    """Check if the first element is a list (assuming consistent structure)"""
    if any(isinstance(i, list) for i in lst):
        return list(chain.from_iterable(lst))
    return lst