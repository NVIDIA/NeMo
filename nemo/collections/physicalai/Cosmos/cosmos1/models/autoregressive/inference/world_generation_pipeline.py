# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0115,C0116,C0301

import gc
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from cosmos1.models.autoregressive.configs.base.model_config import create_video2world_model_config
from cosmos1.models.autoregressive.configs.base.tokenizer import TokenizerConfig
from cosmos1.models.autoregressive.configs.inference.inference_config import (
    DataShapeConfig,
    DiffusionDecoderSamplingConfig,
    InferenceConfig,
    SamplingConfig,
)
from cosmos1.models.autoregressive.diffusion_decoder.inference import diffusion_decoder_process_tokens
from cosmos1.models.autoregressive.diffusion_decoder.model import LatentDiffusionDecoderModel
from cosmos1.models.autoregressive.model import AutoRegressiveModel
from cosmos1.models.autoregressive.utils.inference import _SUPPORTED_CONTEXT_LEN, prepare_video_batch_for_saving
from cosmos1.models.common.base_world_generation_pipeline import BaseWorldGenerationPipeline
from cosmos1.models.diffusion.inference.inference_utils import (
    load_model_by_config,
    load_network_model,
    load_tokenizer_model,
)
from cosmos1.utils import log, misc
from einops import rearrange


def detect_model_size_from_ckpt_path(ckpt_path: str) -> str:
    """Detect model size from checkpoint path.

    Args:
        ckpt_path: Path to model checkpoint file

    Returns:
        str: Model size ('4b', '5b', '12b', or '13b')

    Examples:
        >>> detect_model_size_from_ckpt_path("model_4B.pt")
        '4b'
    """
    model_size = "4b"
    if "4B" in ckpt_path:
        model_size = "4b"
    elif "5B" in ckpt_path:
        model_size = "5b"
    elif "12B" in ckpt_path:
        model_size = "12b"
    elif "13B" in ckpt_path:
        model_size = "13b"
    else:
        log.warning(f"Could not detect model size from checkpoint path: {ckpt_path}")
    return model_size


def create_inference_config(
    model_ckpt_path: str,
    tokenizer_ckpt_path: str,
    model_size: str = "4b",
    batch_size: int = 1,
    inference_type: str = "base",
) -> InferenceConfig:
    """Create inference configuration for model.

    Args:
        model_ckpt_path: Path to model checkpoint
        tokenizer_ckpt_path: Path to tokenizer checkpoint
        model_size: Size of model ('4b', '5b', '12b', '13b')
        batch_size: Batch size for inference
        inference_type: Type of inference ('base' or 'video2world')

    Returns:
        InferenceConfig: Configuration object for inference
    """
    model_size = model_size.lower()
    # For inference config
    kwargs = {}
    if inference_type == "video2world":
        kwargs.update(
            dict(
                insert_cross_attn=True,
                insert_cross_attn_every_k_layers=1,
                context_dim=1024,
                training_type="text_to_video",
                apply_abs_pos_emb=True,
            )
        )
        if model_size == "5b":
            model_size = "4b"  # The base model (excluding the cross attention layers) is the 4B model
        elif model_size == "13b":
            model_size = "12b"  # The base model (excluding the cross attention layers) is the 12B model
        else:
            raise ValueError(f"Unsupported model size for video2world inference_type: {model_size}")
    else:
        assert inference_type == "base", f"Unsupported inference_type: {inference_type}"

    model_config, tokenizer_config = create_video2world_model_config(
        model_ckpt_path=model_ckpt_path,
        tokenizer_ckpt_path=tokenizer_ckpt_path,
        model_size=model_size,
        rope_dim="3D",
        add_special_tokens=False,
        pixel_chunk_duration=33,
        num_video_frames=33,
        num_condition_latents_t=1,
        batch_size=batch_size,
        video_height=640,
        video_width=1024,
        **kwargs,
    )

    inference_config = InferenceConfig()

    inference_config.model_config = model_config
    inference_config.tokenizer_config = tokenizer_config

    inference_config.data_shape_config = DataShapeConfig(
        num_video_frames=model_config.num_video_frames,
        height=model_config.video_height,
        width=model_config.video_width,
        latent_shape=model_config.video_latent_shape,
    )
    inference_config.model_config.fuse_qkv = False
    return inference_config


class ARBaseGenerationPipeline(BaseWorldGenerationPipeline):
    """Base class for autoregressive world generation models.

    Handles the core functionality for generating videos using autoregressive models.
    Provides configurable GPU memory management through model offloading and supports
    different inference types for video generation.

    Attributes:
        inference_config (InferenceConfig): Configuration for model inference
        tokenizer_config (TokenizerConfig): Configuration for tokenizer
        disable_diffusion_decoder (bool): Whether diffusion decoder is disabled
        latent_shape (List[int]): Shape of video latents [T, H, W]
        _supported_context_len (int): Supported context window length
        latent_chunk_duration (int): Duration of latent chunks
        pixel_chunk_duration (int): Duration of pixel chunks
        diffusion_decoder_model (Optional[nn.Module]): The diffusion decoder model
    """

    def __init__(
        self,
        inference_type: str,
        checkpoint_dir: str,
        checkpoint_name: str,
        has_text_input: bool = False,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        disable_diffusion_decoder: bool = False,
        offload_guardrail_models: bool = False,
        offload_diffusion_decoder: bool = False,
    ):
        """Initialize the autoregressive world generation pipeline.

        Args:
            inference_type: Type of world generation ('base' or 'video2world')
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the AR checkpoint to load
            has_text_input: Whether the pipeline takes text input for world generation
            disable_diffusion_decoder: Whether to disable the diffusion decoder stage
            offload_network: Whether to offload AR model from GPU after use
            offload_guardrail_models: Whether to offload content filtering models
            offload_diffusion_decoder: Whether to offload diffusion decoder

        Raises:
            AssertionError: If inference_type is not 'base' or 'video2world'
        """
        assert inference_type in [
            "base",
            "video2world",
        ], "Invalid inference_type, must be 'base' or 'video2world'"

        # Create inference config
        model_size = detect_model_size_from_ckpt_path(checkpoint_name)
        model_ckpt_path = os.path.join(checkpoint_dir, checkpoint_name, "model.pt")
        tokenizer_ckpt_path = os.path.join(checkpoint_dir, "Cosmos-1.0-Tokenizer-DV8x16x16/ema.jit")

        inference_config: InferenceConfig = create_inference_config(
            model_ckpt_path=model_ckpt_path,
            tokenizer_ckpt_path=tokenizer_ckpt_path,
            model_size=model_size,
            inference_type=inference_type,
        )

        self.inference_config = inference_config
        self.disable_diffusion_decoder = disable_diffusion_decoder

        if not disable_diffusion_decoder:
            self.diffusion_decoder_ckpt_path = os.path.join(
                checkpoint_dir, "Cosmos-1.0-Diffusion-7B-Decoder-DV8x16x16ToCV8x8x8/model.pt"
            )
            self.diffusion_decoder_config = "DD_FT_7Bv1_003_002_tokenizer888_spatch2_discrete_cond_on_token"
            self.diffusion_decoder_tokenizer_path = os.path.join(checkpoint_dir, "Cosmos-1.0-Tokenizer-CV8x8x8")
            self.dd_sampling_config = DiffusionDecoderSamplingConfig()
            aux_vars_path = os.path.join(os.path.dirname(self.diffusion_decoder_ckpt_path), "aux_vars.pt")
            # We use a generic prompt when no text prompts are available for diffusion decoder.
            # Generic prompt used - "high quality, 4k, high definition, smooth video"
            aux_vars = torch.load(aux_vars_path, weights_only=True)
            self.generic_prompt = dict()
            self.generic_prompt["context"] = aux_vars["context"].cuda()
            self.generic_prompt["context_mask"] = aux_vars["context_mask"].cuda()

        self.latent_shape = inference_config.data_shape_config.latent_shape  # [L, 40, 64]
        self._supported_context_len = _SUPPORTED_CONTEXT_LEN
        self.tokenizer_config = inference_config.tokenizer_config

        self.offload_diffusion_decoder = offload_diffusion_decoder
        self.diffusion_decoder_model = None
        if not self.offload_diffusion_decoder and not disable_diffusion_decoder:
            self._load_diffusion_decoder()

        super().__init__(
            inference_type=inference_type,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            has_text_input=has_text_input,
            offload_guardrail_models=offload_guardrail_models,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=True,
        )

    def _load_model(self):
        """Load and initialize the autoregressive model.

        Creates and configures the autoregressive model with appropriate settings.
        """
        self.model = AutoRegressiveModel(
            config=self.inference_config.model_config,
        )

    def _load_network(self):
        """Load network weights for the autoregressive model."""
        self.model.load_ar_model(tokenizer_config=self.inference_config.tokenizer_config)

    def _load_tokenizer(self):
        """Load and initialize the tokenizer model.

        Configures the tokenizer using settings from inference_config and
        attaches it to the autoregressive model.
        """
        self.model.load_tokenizer(tokenizer_config=self.inference_config.tokenizer_config)

    def _load_diffusion_decoder(self):
        """Load and initialize the diffusion decoder model."""
        self.diffusion_decoder_model = load_model_by_config(
            config_job_name=self.diffusion_decoder_config,
            config_file="cosmos1/models/autoregressive/diffusion_decoder/config/config_latent_diffusion_decoder.py",
            model_class=LatentDiffusionDecoderModel,
        )
        load_network_model(self.diffusion_decoder_model, self.diffusion_decoder_ckpt_path)
        load_tokenizer_model(self.diffusion_decoder_model, self.diffusion_decoder_tokenizer_path)

    def _offload_diffusion_decoder(self):
        """Offload diffusion decoder model from GPU memory."""
        if self.diffusion_decoder_model is not None:
            del self.diffusion_decoder_model
            self.diffusion_decoder_model = None
        gc.collect()
        torch.cuda.empty_cache()

    def _run_model_with_offload(
        self, inp_vid: torch.Tensor, num_input_frames: int, seed: int, sampling_config: SamplingConfig
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run the autoregressive model to generate video tokens.

        Takes input video frames and generates new video tokens using the autoregressive model.
        Handles context frame selection and token generation.

        Args:
            inp_vid (torch.Tensor): Input video tensor of shape
            num_input_frames (int): Number of context frames to use from input. The tensor shape should be (B x T x 3 x H x W).
            seed (int): Random seed for generation
            sampling_config (SamplingConfig): Configuration for sampling parameters

        Returns:
            tuple: (
                List of generated video tensors,
                List of token index tensors,
                List of prompt embedding tensors
            )
        """
        # Choosing the context length from list of available contexts
        latent_context_t_size = 0
        context_used = 0
        for _clen in self._supported_context_len:
            if num_input_frames >= _clen:
                context_used = _clen
                latent_context_t_size += 1
        log.info(f"Using input size of {context_used} frames")

        data_batch = {"video": inp_vid}
        data_batch = misc.to(data_batch, "cuda")

        T, H, W = self.latent_shape
        num_gen_tokens = int(np.prod([T - latent_context_t_size, H, W]))

        out_videos_cur_batch, indices_tensor_cur_batch = self.generate_partial_tokens_from_data_batch(
            data_batch=data_batch,
            num_tokens_to_generate=num_gen_tokens,
            sampling_config=sampling_config,
            tokenizer_config=self.tokenizer_config,
            latent_shape=self.latent_shape,
            task_condition="video",
            num_chunks_to_generate=1,
            seed=seed,
        )
        if self.offload_network:
            self._offload_network()
        if self.offload_tokenizer:
            self._offload_tokenizer()
        return out_videos_cur_batch, indices_tensor_cur_batch

    def _run_diffusion_decoder(
        self,
        out_videos_cur_batch: List[torch.Tensor],
        indices_tensor_cur_batch: List[torch.Tensor],
        t5_emb_batch: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Process generated tokens through the diffusion decoder.

        Enhances video quality through diffusion-based decoding.

        Args:
            out_videos_cur_batch: List of generated video tensors
            indices_tensor_cur_batch: List of token indices tensors
            t5_emb_batch: List of text embeddings for conditioning

        Returns:
            list: Enhanced video tensors after diffusion processing
        """
        out_videos_cur_batch_dd = diffusion_decoder_process_tokens(
            model=self.diffusion_decoder_model,
            indices_tensor=indices_tensor_cur_batch,
            dd_sampling_config=self.dd_sampling_config,
            original_video_example=out_videos_cur_batch[0],
            t5_emb_batch=t5_emb_batch,
        )
        return out_videos_cur_batch_dd

    def _run_diffusion_decoder_with_offload(
        self,
        out_videos_cur_batch: List[torch.Tensor],
        indices_tensor_cur_batch: List[torch.Tensor],
        t5_emb_batch: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Run diffusion decoder with memory management.

        Loads decoder if needed, processes videos, and offloads decoder afterward
        if configured in offload_diffusion_decoder.

        Args:
            out_videos_cur_batch: List of generated video tensors
            indices_tensor_cur_batch: List of token indices tensors
            t5_emb_batch: List of text embeddings for conditioning

        Returns:
            list: Enhanced video tensors after diffusion processing
        """
        if self.offload_diffusion_decoder:
            self._load_diffusion_decoder()
        out_videos_cur_batch = self._run_diffusion_decoder(
            out_videos_cur_batch, indices_tensor_cur_batch, t5_emb_batch
        )
        if self.offload_diffusion_decoder:
            self._offload_diffusion_decoder()
        return out_videos_cur_batch

    def generate(
        self,
        inp_vid: torch.Tensor,
        sampling_config: SamplingConfig,
        num_input_frames: int = 9,
        seed: int = 0,
    ) -> np.ndarray | None:
        """Generate a video continuation from input frames.

        Pipeline steps:
        1. Generates video tokens using autoregressive model
        2. Optionally enhances quality via diffusion decoder
        3. Applies safety checks if enabled

        Args:
            inp_vid: Input video tensor of shape (batch_size, time, channels=3, height, width)
            sampling_config: Parameters controlling the generation process
            num_input_frames: Number of input frames to use as context (default: 9)
            seed: Random seed for reproducibility (default: 0)

        Returns:
            np.ndarray | None: Generated video as numpy array (time, height, width, channels)
                if generation successful, None if safety checks fail
        """
        log.info("Run generation")
        out_videos_cur_batch, indices_tensor_cur_batch = self._run_model_with_offload(
            inp_vid, num_input_frames, seed, sampling_config
        )
        log.info("Finish AR model generation")

        if not self.disable_diffusion_decoder:
            log.info("Run diffusion decoder on generated tokens")
            out_videos_cur_batch = self._run_diffusion_decoder_with_offload(
                out_videos_cur_batch, indices_tensor_cur_batch, t5_emb_batch=[self.generic_prompt["context"]]
            )
            log.info("Finish diffusion decoder on generated tokens")
        out_videos_cur_batch = prepare_video_batch_for_saving(out_videos_cur_batch)
        output_video = out_videos_cur_batch[0]

        log.info("Run guardrail on generated video")
        output_video = self._run_guardrail_on_video_with_offload(output_video)
        if output_video is None:
            log.critical("Generated video is not safe")
            return None
        log.info("Finish guardrail on generated video")

        return output_video

    @torch.inference_mode()
    def generate_partial_tokens_from_data_batch(
        self,
        data_batch: dict,
        num_tokens_to_generate: int,
        sampling_config: SamplingConfig,
        tokenizer_config: TokenizerConfig,
        latent_shape: list[int],
        task_condition: str,
        num_chunks_to_generate: int = 1,
        seed: int = 0,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Generate video tokens from partial input tokens with conditioning.

        Handles token generation and decoding process:
        1. Processes input batch and applies conditioning
        2. Generates specified number of new tokens
        3. Decodes tokens to video frames

        Args:
            data_batch: Dictionary containing input data including video and optional context
            num_tokens_to_generate: Number of tokens to generate
            sampling_config: Configuration for sampling parameters
            tokenizer_config: Configuration for tokenizer, including video tokenizer settings
            latent_shape: Shape of video latents [T, H, W]
            task_condition: Type of generation task ('video' or 'text_and_video')
            num_chunks_to_generate: Number of chunks to generate (default: 1)
            seed: Random seed for generation (default: 0)

        Returns:
            tuple containing:
                - List[torch.Tensor]: Generated videos
                - List[torch.Tensor]: Input videos
                - List[torch.Tensor]: Generated tokens
                - List[torch.Tensor]: Token index tensors
        """
        log.debug(f"Starting generate_partial_tokens_from_data_batch with seed {seed}")
        log.debug(f"Number of tokens to generate: {num_tokens_to_generate}")
        log.debug(f"Latent shape: {latent_shape}")

        video_token_start = tokenizer_config.video_tokenizer.tokenizer_offset
        video_vocab_size = tokenizer_config.video_tokenizer.vocab_size
        video_token_end = video_token_start + video_vocab_size

        logit_clipping_range = [video_token_start, video_token_end]

        if self.offload_network:
            self._offload_network()
        if self.offload_tokenizer:
            self._load_tokenizer()

        assert logit_clipping_range == [
            0,
            self.model.tokenizer.video_vocab_size,
        ], f"logit_clipping_range {logit_clipping_range} is not supported for fast generate. Expected [0, {self.model.tokenizer.video_vocab_size}]"

        out_videos = {}
        out_indices_tensors = {}

        # for text2world, we only add a <bov> token at the beginning of the video tokens, this applies to 5B and 13B models
        if self.model.tokenizer.tokenizer_config.training_type == "text_to_video":
            num_bov_tokens = 1
            num_eov_tokens = 0
        else:
            num_eov_tokens = 1 if self.model.tokenizer.tokenizer_config.add_special_tokens else 0
            num_bov_tokens = 1 if self.model.tokenizer.tokenizer_config.add_special_tokens else 0

        chunk_idx = 0
        out_videos[chunk_idx] = []
        out_indices_tensors[chunk_idx] = []

        # get the context embedding and mask
        context = data_batch.get("context", None) if task_condition != "video" else None
        context_mask = data_batch.get("context_mask", None) if task_condition != "video" else None
        if context is not None:
            context = misc.to(context, "cuda").detach().clone()
        if context_mask is not None:
            context_mask = misc.to(context_mask, "cuda").detach().clone()

        # get the video tokens
        data_tokens, token_boundaries = self.model.tokenizer.tokenize(data_batch=data_batch)
        data_tokens = misc.to(data_tokens, "cuda").detach().clone()
        batch_size = data_tokens.shape[0]

        for sample_num in range(batch_size):
            input_tokens = data_tokens[sample_num][0 : token_boundaries["video"][sample_num][1]]  # [B, L]
            input_tokens = [
                input_tokens[0 : -num_tokens_to_generate - num_eov_tokens].tolist()
            ]  # -1 is to exclude eov token
            log.debug(
                f"Run sampling. # input condition tokens: {len(input_tokens[0])}; # generate tokens: {num_tokens_to_generate + num_eov_tokens}; "
                f"full length of the data tokens: {len(data_tokens[sample_num])}: {data_tokens[sample_num]}"
            )
            video_start_boundary = token_boundaries["video"][sample_num][0] + num_bov_tokens

            video_decoded, indices_tensor = self.generate_video_from_tokens(
                prompt_tokens=input_tokens,
                latent_shape=latent_shape,
                video_start_boundary=video_start_boundary,
                max_gen_len=num_tokens_to_generate,
                sampling_config=sampling_config,
                logit_clipping_range=logit_clipping_range,
                seed=seed,
                context=context,
                context_mask=context_mask,
            )  # BCLHW, range [0, 1]

            # For the first chunk, we store the entire generated video
            out_videos[chunk_idx].append(video_decoded[sample_num].detach().clone())
            out_indices_tensors[chunk_idx].append(indices_tensor[sample_num].detach().clone())

        output_videos = []
        output_indice_tensors = []
        for sample_num in range(len(out_videos[0])):
            tensors_to_concat = [out_videos[chunk_idx][sample_num] for chunk_idx in range(num_chunks_to_generate)]
            concatenated = torch.cat(tensors_to_concat, dim=1)
            output_videos.append(concatenated)

            indices_tensor_to_concat = [
                out_indices_tensors[chunk_idx][sample_num] for chunk_idx in range(num_chunks_to_generate)
            ]
            concatenated_indices_tensor = torch.cat(indices_tensor_to_concat, dim=1)  # BLHW
            output_indice_tensors.append(concatenated_indices_tensor)

        return output_videos, output_indice_tensors

    def generate_video_from_tokens(
        self,
        prompt_tokens: list[torch.Tensor],
        latent_shape: list[int],
        video_start_boundary: int,
        max_gen_len: int,
        sampling_config: SamplingConfig,
        logit_clipping_range: list[int],
        seed: int = 0,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Function to generate video from input tokens. These input tokens can be initial text tokens (in case of text to video),
        or partial ground truth tokens.

        Handles the core token-to-video generation process:
        1. Generates new tokens using the autoregressive model
        2. Handles padding and token sequence completion
        3. Reshapes and processes generated tokens
        4. Decodes final tokens into video frames

        Args:
            model (AutoRegressiveModel): LLama model instance
            prompt_tokens (list): Prompt tokens used by the model
            latent_shape (list): Shape of the video latents
            video_start_boundary (int): Index where the video tokens start
            max_gen_len (int): Maximum length of the tokens that needs to be generated
            sampling_config (SamplingConfig): Config used by sampler during inference
            logit_clipping_range (list): Range of indices in the logits to be clipped, e.g. [video_token_start, video_token_end]
            context (Optional[torch.Tensor]): The context tensor added via cross-attn.
            context_mask (Optional[torch.Tensor]): The context cross-attn mask tensor.
        Returns:
            tuple containing:
                - List[torch.Tensor]: Generated videos
                - List[torch.Tensor]: Generated tokens
                - List[torch.Tensor]: Token index tensors
        """
        # Combine the tokens and do padding, sometimes the generated tokens end before the max_gen_len
        total_seq_len = np.prod(latent_shape)

        assert not sampling_config.logprobs

        stop_tokens = self.model.tokenizer.stop_tokens
        if self.offload_tokenizer:
            self._offload_tokenizer()
        if self.offload_network:
            self._load_network()

        generation_tokens, _ = self.model.generate(
            prompt_tokens=prompt_tokens,
            temperature=sampling_config.temperature,
            top_p=sampling_config.top_p,
            echo=sampling_config.echo,
            seed=seed,
            context=context,
            context_mask=context_mask,
            max_gen_len=max_gen_len,
            compile_sampling=sampling_config.compile_sampling,
            compile_prefill=sampling_config.compile_prefill,
            stop_tokens=stop_tokens,
            verbose=True,
        )
        generation_tokens = generation_tokens[:, video_start_boundary:]
        # Combine the tokens and do padding, sometimes the generated tokens end before the max_gen_len
        if generation_tokens.shape[1] < total_seq_len:
            log.warning(
                f"Generated video tokens (shape:{generation_tokens.shape}) shorted than expected {total_seq_len}. Could be the model produce end token early. Repeat the last token to fill the sequence in order for decoding."
            )
            padding_len = total_seq_len - generation_tokens.shape[1]
            padding_tokens = generation_tokens[:, [-1]].repeat(1, padding_len)
            generation_tokens = torch.cat([generation_tokens, padding_tokens], dim=1)
        # Cast to LongTensor
        indices_tensor = generation_tokens.long()
        # First, we reshape the generated tokens into batch x time x height x width
        indices_tensor = rearrange(
            indices_tensor,
            "B (T H W) -> B T H W",
            T=latent_shape[0],
            H=latent_shape[1],
            W=latent_shape[2],
        )
        log.debug(f"generated video tokens {len(generation_tokens[0])} -> reshape: {indices_tensor.shape}")
        # If logit clipping range is specified, offset the generated indices by the logit_clipping_range[0]
        # Video decoder always takes tokens in the range (0, N-1). So, this offset is needed.
        if len(logit_clipping_range) > 0:
            indices_tensor = indices_tensor - logit_clipping_range[0]

        if self.offload_network:
            self._offload_network()
        if self.offload_tokenizer:
            self._load_tokenizer()

        # Now decode the video using tokenizer.
        video_decoded = self.model.tokenizer.video_tokenizer.decode(indices_tensor.cuda())
        # Normalize decoded video from [-1, 1] to [0, 1], and clip value
        video_decoded = (video_decoded * 0.5 + 0.5).clamp_(0, 1)
        return video_decoded, indices_tensor


class ARVideo2WorldGenerationPipeline(ARBaseGenerationPipeline):
    """Video-to-world generation pipeline with text conditioning capabilities.

    Extends the base autoregressive generation pipeline by adding:
    - Text prompt processing and embedding
    - Text-conditioned video generation
    - Additional safety checks for text input
    - Memory management for text encoder model

    Enables generating video continuations that are guided by both
    input video frames and text descriptions.

    Additional attributes compared to ARBaseGenerationPipeline:
        offload_text_encoder_model (bool): Whether to offload text encoder from GPU after use
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_name: str,
        inference_type: str = None,
        has_text_input: bool = True,
        disable_diffusion_decoder: bool = False,
        offload_guardrail_models: bool = False,
        offload_diffusion_decoder: bool = False,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
    ):
        """Initialize text-conditioned video generation pipeline.

        Args:
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the checkpoint to load
            inference_type: Type of world generation workflow
            has_text_input: Whether the pipeline takes text input for world generation
            disable_diffusion_decoder: Whether to disable diffusion decoder stage
            offload_guardrail_models: Whether to offload content filtering models
            offload_diffusion_decoder: Whether to offload diffusion decoder
            offload_network: Whether to offload AR model from GPU
            offload_tokenizer: Whether to offload tokenizer from GPU
            offload_text_encoder_model: Whether to offload text encoder
        """
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            inference_type=inference_type,
            has_text_input=has_text_input,
            disable_diffusion_decoder=disable_diffusion_decoder,
            offload_guardrail_models=offload_guardrail_models,
            offload_diffusion_decoder=offload_diffusion_decoder,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
        )
        self.offload_text_encoder_model = offload_text_encoder_model
        if not self.offload_text_encoder_model:
            self._load_text_encoder_model()

    def _run_model_with_offload(
        self,
        prompt_embedding: torch.Tensor,
        prompt_mask: torch.Tensor,
        inp_vid: torch.Tensor,
        num_input_frames: int,
        seed: int,
        sampling_config: SamplingConfig,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Run model generation with memory management.

        Executes generation process and handles model offloading to manage GPU memory.

        Args:
            prompt_embedding: Text prompt embeddings tensor
            prompt_mask: Attention mask for prompt embeddings
            inp_vid: Input video tensor
            num_input_frames: Number of input frames to use
            seed: Random seed for reproducibility
            sampling_config: Configuration for sampling parameters

        Returns:
            tuple: (
                List of generated video tensors
                List of token index tensors
                List of prompt embedding tensors
            )
        """
        out_videos, indices_tensor, prompt_embedding = self._run_model(
            prompt_embedding, prompt_mask, inp_vid, num_input_frames, seed, sampling_config
        )
        if self.offload_network:
            self._offload_network()
        if self.offload_tokenizer:
            self._offload_tokenizer()
        return out_videos, indices_tensor, prompt_embedding

    def _run_model(
        self,
        prompt_embedding: torch.Tensor,
        prompt_mask: torch.Tensor,
        inp_vid: torch.Tensor,
        num_input_frames: int,
        seed: int,
        sampling_config: SamplingConfig,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Run core model generation process.

        Handles text-conditioned video generation:
        1. Prepares data batch with text embeddings and video
        2. Determines appropriate context length
        3. Generates video tokens with text conditioning
        4. Processes output tensors

        Args:
            prompt_embedding: Text prompt embeddings tensor
            prompt_mask: Attention mask for prompt embeddings
            inp_vid: Input video tensor
            num_input_frames: Number of input frames to use
            seed: Random seed for reproducibility
            sampling_config: Configuration for sampling parameters,
                uses default config if None

        Returns:
            tuple: (
                List of generated video tensors
                List of token index tensors
                Text context tensor
            )
        """
        data_batch = {}
        data_batch["context"], data_batch["context_mask"] = prompt_embedding, prompt_mask
        T, H, W = self.latent_shape

        if sampling_config is None:
            sampling_config = self.sampling_config
        if type(inp_vid) is list:
            batch_size = len(inp_vid)
        elif type(inp_vid) is torch.Tensor:
            batch_size = 1
        data_batch["context"] = data_batch["context"].repeat(batch_size, 1, 1)
        data_batch["context_mask"] = data_batch["context_mask"].repeat(batch_size, 1)
        data_batch["context_mask"] = torch.ones_like(data_batch["context_mask"]).bool()

        latent_context_t_size = 0

        # Choosing the context length from list of available contexts
        context_used = 0
        for _clen in self._supported_context_len:
            if num_input_frames >= _clen:
                context_used = _clen
                latent_context_t_size += 1
        log.info(f"Using context of {context_used} frames")

        num_gen_tokens = int(np.prod([T - latent_context_t_size, H, W]))

        data_batch["video"] = inp_vid
        data_batch["video"] = data_batch["video"].repeat(batch_size, 1, 1, 1, 1)

        data_batch = misc.to(data_batch, "cuda")

        log.debug(f"  num_tokens_to_generate: {num_gen_tokens}")
        log.debug(f"  sampling_config: {sampling_config}")
        log.debug(f"  tokenizer_config: {self.tokenizer_config}")
        log.debug(f"  latent_shape: {self.latent_shape}")
        log.debug(f"  latent_context_t_size: {latent_context_t_size}")
        log.debug(f"  seed: {seed}")

        out_videos_cur_batch, indices_tensor_cur_batch = self.generate_partial_tokens_from_data_batch(
            data_batch=data_batch,
            num_tokens_to_generate=num_gen_tokens,
            sampling_config=sampling_config,
            tokenizer_config=self.tokenizer_config,
            latent_shape=self.latent_shape,
            task_condition="text_and_video",
            seed=seed,
        )
        return out_videos_cur_batch, indices_tensor_cur_batch, data_batch["context"]

    def generate(
        self,
        inp_prompt: str,
        inp_vid: torch.Tensor,
        num_input_frames: int = 9,
        seed: int = 0,
        sampling_config: SamplingConfig = None,
    ) -> np.ndarray | None:
        """Generate a video guided by text prompt and input frames.

        Pipeline steps:
        1. Validates text prompt safety if enabled
        2. Converts text to embeddings
        3. Generates video with text conditioning
        4. Enhances quality via diffusion decoder
        5. Applies video safety checks if enabled

        Args:
            inp_prompt: Text prompt to guide the generation
            inp_vid: Input video tensor with shape (batch_size, time, channels=3, height, width)
            num_input_frames: Number of frames to use as context (default: 9)
            seed: Random seed for reproducibility (default: 0)
            sampling_config: Configuration for sampling parameters,
                uses default config if None

        Returns:
            np.ndarray | None: Generated video as numpy array (time, height, width, channels)
                if generation successful, None if safety checks fail
        """
        log.info("Run guardrail on prompt")
        is_safe = self._run_guardrail_on_prompt_with_offload(inp_prompt)
        if not is_safe:
            log.critical("Input text prompt is not safe")
            return None
        log.info("Pass guardrail on prompt")

        log.info("Run text embedding on prompt")
        prompt_embeddings, prompt_masks = self._run_text_embedding_on_prompt_with_offload([inp_prompt])
        prompt_embedding = prompt_embeddings[0]
        prompt_mask = prompt_masks[0]
        log.info("Finish text embedding on prompt")

        log.info("Run generation")
        out_videos_cur_batch, indices_tensor_cur_batch, prompt_embedding = self._run_model_with_offload(
            prompt_embedding, prompt_mask, inp_vid, num_input_frames, seed, sampling_config
        )
        log.info("Finish AR model generation")

        if not self.disable_diffusion_decoder:
            log.info("Run diffusion decoder on generated tokens")
            out_videos_cur_batch = self._run_diffusion_decoder_with_offload(
                out_videos_cur_batch, indices_tensor_cur_batch, [prompt_embedding]
            )
            log.info("Finish diffusion decoder on generated tokens")
        out_videos_cur_batch = prepare_video_batch_for_saving(out_videos_cur_batch)
        output_video = out_videos_cur_batch[0]

        log.info("Run guardrail on generated video")
        output_video = self._run_guardrail_on_video_with_offload(output_video)
        if output_video is None:
            log.critical("Generated video is not safe")
            return None
        log.info("Finish guardrail on generated video")

        return output_video
