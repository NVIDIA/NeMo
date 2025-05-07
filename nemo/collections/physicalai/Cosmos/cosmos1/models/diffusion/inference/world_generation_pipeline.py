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
from typing import Any, Optional

import einops
import numpy as np
import torch

from cosmos1.models.common.base_world_generation_pipeline import BaseWorldGenerationPipeline
from cosmos1.models.diffusion.inference.inference_utils import (
    generate_world_from_text,
    generate_world_from_video,
    get_condition_latent,
    get_condition_latent_multi_camera,
    get_video_batch,
    get_video_batch_for_multi_camera_model,
    load_model_by_config,
    load_network_model,
    load_tokenizer_model,
)
from cosmos1.models.diffusion.model.model_sample_multiview_driving import (
    DiffusionMultiCameraT2WModel,
    DiffusionMultiCameraV2WModel,
)
from cosmos1.models.diffusion.model.model_t2w import DiffusionT2WModel
from cosmos1.models.diffusion.model.model_v2w import DiffusionV2WModel
from cosmos1.models.diffusion.prompt_upsampler.text2world_prompt_upsampler_inference import (
    create_prompt_upsampler,
    run_chat_completion,
)
from cosmos1.models.diffusion.prompt_upsampler.video2world_prompt_upsampler_inference import (
    create_vlm_prompt_upsampler,
    prepare_dialog,
)
from cosmos1.models.diffusion.prompt_upsampler.video2world_prompt_upsampler_inference import (
    run_chat_completion as run_chat_completion_vlm,
)
from cosmos1.utils import log

MODEL_NAME_DICT = {
    "Cosmos-1.0-Diffusion-7B-Text2World": "Cosmos_1_0_Diffusion_Text2World_7B",
    "Cosmos-1.0-Diffusion-14B-Text2World": "Cosmos_1_0_Diffusion_Text2World_14B",
    "Cosmos-1.0-Diffusion-7B-Video2World": "Cosmos_1_0_Diffusion_Video2World_7B",
    "Cosmos-1.0-Diffusion-14B-Video2World": "Cosmos_1_0_Diffusion_Video2World_14B",
    "Cosmos-1.1-Diffusion-7B-Text2World-Sample-Driving-Multiview": "Cosmos_1_1_Diffusion_Multi_Camera_Text2World_7B",
    "Cosmos-1.1-Diffusion-7B-Video2World-Sample-Driving-Multiview": "Cosmos_1_1_Diffusion_Multi_Camera_Video2World_7B",
}


class DiffusionText2WorldGenerationPipeline(BaseWorldGenerationPipeline):
    def __init__(
        self,
        inference_type: str,
        checkpoint_dir: str,
        checkpoint_name: str,
        prompt_upsampler_dir: Optional[str] = None,
        enable_prompt_upsampler: bool = True,
        has_text_input: bool = True,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
        offload_prompt_upsampler: bool = False,
        offload_guardrail_models: bool = False,
        guidance: float = 7.0,
        num_steps: int = 35,
        height: int = 704,
        width: int = 1280,
        fps: int = 24,
        num_video_frames: int = 121,
        seed: int = 0,
    ):
        """Initialize the diffusion world generation pipeline.

        Args:
            inference_type: Type of world generation ('text2world' or 'video2world')
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the diffusion transformer checkpoint to use
            prompt_upsampler_dir: Directory containing prompt upsampler model weights
            enable_prompt_upsampler: Whether to use prompt upsampling
            has_text_input: Whether the pipeline takes text input for world generation
            offload_network: Whether to offload diffusion transformer after inference
            offload_tokenizer: Whether to offload tokenizer after inference
            offload_text_encoder_model: Whether to offload T5 model after inference
            offload_prompt_upsampler: Whether to offload prompt upsampler
            offload_guardrail_models: Whether to offload guardrail models
            guidance: Classifier-free guidance scale
            num_steps: Number of diffusion sampling steps
            height: Height of output video
            width: Width of output video
            fps: Frames per second of output video
            num_video_frames: Number of frames to generate
            seed: Random seed for sampling
        """
        assert inference_type in [
            "text2world",
            "video2world",
        ], "Invalid inference_type, must be 'text2world' or 'video2world'"

        self.model_name = MODEL_NAME_DICT[checkpoint_name]
        self.guidance = guidance
        self.num_steps = num_steps
        self.height = height
        self.width = width
        self.fps = fps
        self.num_video_frames = num_video_frames
        self.seed = seed

        super().__init__(
            inference_type=inference_type,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            has_text_input=has_text_input,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=offload_text_encoder_model,
            offload_guardrail_models=offload_guardrail_models,
        )
        self.prompt_upsampler_dir = prompt_upsampler_dir
        self.enable_prompt_upsampler = enable_prompt_upsampler
        self.offload_prompt_upsampler = offload_prompt_upsampler

        self.prompt_upsampler = None
        if enable_prompt_upsampler and not offload_prompt_upsampler:
            self._load_prompt_upsampler_model()

    def _load_prompt_upsampler_model(self):
        self.prompt_upsampler = create_prompt_upsampler(
            checkpoint_dir=os.path.join(self.checkpoint_dir, self.prompt_upsampler_dir),
        )

    def _load_model(self):
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos1/models/diffusion/config/config.py",
            model_class=DiffusionT2WModel,
        )

    def _load_network(self):
        load_network_model(self.model, f"{self.checkpoint_dir}/{self.checkpoint_name}/model.pt")

    def _load_tokenizer(self):
        load_tokenizer_model(self.model, f"{self.checkpoint_dir}/Cosmos-1.0-Tokenizer-CV8x8x8")

    def _offload_prompt_upsampler_model(self):
        """Move prompt enhancement model to CPU/disk.

        Offloads prompt upsampling model after processing input
        to reduce GPU memory usage.
        """
        if self.prompt_upsampler:
            del self.prompt_upsampler
            self.prompt_upsampler = None
            gc.collect()
            torch.cuda.empty_cache()

    def _run_prompt_upsampler_on_prompt(self, prompt: str) -> str:
        """Enhance the input prompt using the prompt upsampler model.

        Args:
            prompt: Raw text prompt to be enhanced

        Returns:
            str: Enhanced version of the input prompt with more descriptive details
        """
        upsampled_prompt = run_chat_completion(self.prompt_upsampler, prompt)
        log.info(f"Upsampled prompt: {upsampled_prompt}")
        return upsampled_prompt

    def _run_prompt_upsampler_on_prompt_with_offload(self, *args: Any, **kwargs: Any) -> str:
        """Enhance prompt with prompt upsampler model.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Enhanced prompt string
        """
        if self.offload_prompt_upsampler:
            self._load_prompt_upsampler_model()

        enhanced_prompt = self._run_prompt_upsampler_on_prompt(*args, **kwargs)

        if self.offload_prompt_upsampler:
            self._offload_prompt_upsampler_model()

        return enhanced_prompt

    def _run_tokenizer_decoding(self, sample: torch.Tensor) -> np.ndarray:
        """Decode latent samples to video frames using the tokenizer decoder.

        Args:
            sample: Latent tensor from diffusion model [B, C, T, H, W]

        Returns:
            np.ndarray: Decoded video frames as uint8 numpy array [T, H, W, C]
                        with values in range [0, 255]
        """
        # Decode video
        video = (1.0 + self.model.decode(sample)).clamp(0, 2) / 2  # [B, 3, T, H, W]
        video = (video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()

        return video

    def _run_model(
        self,
        embedding: torch.Tensor,
        negative_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate video latents using the diffusion model.

        Args:
            embedding: Text embedding tensor from text encoder
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            torch.Tensor: Generated video latents before tokenizer decoding

        Note:
            The model and tokenizer are automatically offloaded after inference
            if offloading is enabled in the config.
        """
        # Get video batch and state shape
        data_batch, state_shape = get_video_batch(
            model=self.model,
            prompt_embedding=embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames,
        )

        # Generate video frames
        sample = generate_world_from_text(
            model=self.model,
            state_shape=state_shape,
            is_negative_prompt=True if negative_prompt_embedding is not None else False,
            data_batch=data_batch,
            guidance=self.guidance,
            num_steps=self.num_steps,
            seed=self.seed,
        )

        return sample

    def _run_model_with_offload(
        self, prompt_embedding: torch.Tensor, negative_prompt_embedding: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """Generate world representation with automatic model offloading.

        Wraps the core generation process with model loading/offloading logic
        to minimize GPU memory usage during inference.

        Args:
            *args: Positional arguments passed to _run_model
            **kwargs: Keyword arguments passed to _run_model

        Returns:
            np.ndarray: Generated world representation as numpy array
        """
        if self.offload_network:
            self._load_network()

        if self.offload_tokenizer:
            self._load_tokenizer()

        sample = self._run_model(prompt_embedding, negative_prompt_embedding)

        if self.offload_network:
            self._offload_network()

        if self.offload_tokenizer:
            self._load_tokenizer()

        sample = self._run_tokenizer_decoding(sample)

        if self.offload_tokenizer:
            self._offload_tokenizer()
        return sample

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        word_limit_to_skip_upsampler: Optional[int] = None,
    ) -> tuple[np.ndarray, str] | None:
        """Generate video from text prompt with optional negative prompt guidance.

        Pipeline steps:
        1. Run safety checks on input prompt
        2. Enhance prompt using upsampler if enabled
        3. Run safety checks on upsampled prompt if applicable
        4. Convert prompt to embeddings
        5. Generate video frames using diffusion
        6. Run safety checks and apply face blur on generated video frames

        Args:
            prompt: Text description of desired video
            negative_prompt: Optional text to guide what not to generate
            word_limit_to_skip_upsampler: Skip prompt upsampler for better robustness if the number of words in the prompt is greater than this value
        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """
        log.info(f"Run with prompt: {prompt}")
        log.info(f"Run with negative prompt: {negative_prompt}")
        log.info(f"Run with prompt upsampler: {self.enable_prompt_upsampler}")

        log.info("Run guardrail on prompt")
        is_safe = self._run_guardrail_on_prompt_with_offload(prompt)
        if not is_safe:
            log.critical("Input text prompt is not safe")
            return None
        log.info("Pass guardrail on prompt")

        # Enhance prompt
        if self.enable_prompt_upsampler:
            word_count = len(prompt.split())
            if word_limit_to_skip_upsampler is None or word_count <= word_limit_to_skip_upsampler:
                log.info("Run prompt upsampler on prompt")
                prompt = self._run_prompt_upsampler_on_prompt_with_offload(prompt)
                log.info("Run guardrail on upsampled prompt")
                is_safe = self._run_guardrail_on_prompt_with_offload(prompt=prompt)
                if not is_safe:
                    log.critical("Upsampled text prompt is not safe")
                    return None
                log.info("Pass guardrail on upsampled prompt")
            else:
                log.info(
                    f"Skip prompt upsampler for better robustness because the number of words ({word_count}) in the prompt is greater than {word_limit_to_skip_upsampler}"
                )

        log.info("Run text embedding on prompt")
        if negative_prompt:
            prompts = [prompt, negative_prompt]
        else:
            prompts = [prompt]
        prompt_embeddings, _ = self._run_text_embedding_on_prompt_with_offload(prompts)
        prompt_embedding = prompt_embeddings[0]
        negative_prompt_embedding = prompt_embeddings[1] if negative_prompt else None
        log.info("Finish text embedding on prompt")

        # Generate video
        log.info("Run generation")
        video = self._run_model_with_offload(
            prompt_embedding,
            negative_prompt_embedding=negative_prompt_embedding,
        )
        log.info("Finish generation")

        log.info("Run guardrail on generated video")
        video = self._run_guardrail_on_video_with_offload(video)
        if video is None:
            log.critical("Generated video is not safe")
            return None
        log.info("Pass guardrail on generated video")

        return video, prompt


class DiffusionVideo2WorldGenerationPipeline(DiffusionText2WorldGenerationPipeline):
    def __init__(
        self,
        inference_type: str,
        checkpoint_dir: str,
        checkpoint_name: str,
        prompt_upsampler_dir: Optional[str] = None,
        enable_prompt_upsampler: bool = True,
        has_text_input: bool = True,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
        offload_prompt_upsampler: bool = False,
        offload_guardrail_models: bool = False,
        guidance: float = 7.0,
        num_steps: int = 35,
        height: int = 704,
        width: int = 1280,
        fps: int = 24,
        num_video_frames: int = 121,
        seed: int = 0,
        num_input_frames: int = 1,
    ):
        """Initialize diffusion world generation pipeline.

        Args:
            inference_type: Type of world generation ('text2world' or 'video2world')
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the diffusion transformer checkpoint to use
            prompt_upsampler_dir: Directory containing prompt upsampler model weights
            enable_prompt_upsampler: Whether to use prompt upsampling
            has_text_input: Whether the pipeline takes text input for world generation
            offload_network: Whether to offload diffusion transformer after inference
            offload_tokenizer: Whether to offload tokenizer after inference
            offload_text_encoder_model: Whether to offload T5 model after inference
            offload_prompt_upsampler: Whether to offload prompt upsampler
            offload_guardrail_models: Whether to offload guardrail models
            guidance: Classifier-free guidance scale
            num_steps: Number of diffusion sampling steps
            height: Height of output video
            width: Width of output video
            fps: Frames per second of output video
            num_video_frames: Number of frames to generate
            seed: Random seed for sampling
            num_input_frames: Number of latent conditions
        """
        self.num_input_frames = num_input_frames
        super().__init__(
            inference_type=inference_type,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            prompt_upsampler_dir=prompt_upsampler_dir,
            enable_prompt_upsampler=enable_prompt_upsampler,
            has_text_input=has_text_input,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=offload_text_encoder_model,
            offload_prompt_upsampler=offload_prompt_upsampler,
            offload_guardrail_models=offload_guardrail_models,
            guidance=guidance,
            num_steps=num_steps,
            height=height,
            width=width,
            fps=fps,
            num_video_frames=num_video_frames,
            seed=seed,
        )

    def _run_prompt_upsampler_on_prompt(self, image_or_video_path: str) -> str:
        """Enhance the input prompt using visual context from the conditioning image.

        Args:
            image_or_video_path: Path to conditioning image or video used for visual context

        Returns:
            str: Enhanced prompt incorporating visual details from the image
        """
        dialog = prepare_dialog(image_or_video_path)
        upsampled_prompt = run_chat_completion_vlm(
            self.prompt_upsampler, dialog, max_gen_len=400, temperature=0.01, top_p=0.9, logprobs=False
        )
        log.info(f"Upsampled prompt: {upsampled_prompt}")
        return upsampled_prompt

    def _load_prompt_upsampler_model(self):
        self.prompt_upsampler = create_vlm_prompt_upsampler(
            checkpoint_dir=os.path.join(self.checkpoint_dir, self.prompt_upsampler_dir),
        )

    def _load_model(self):
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos1/models/diffusion/config/config.py",
            model_class=DiffusionV2WModel,
        )

    def _run_model(
        self,
        embedding: torch.Tensor,
        condition_latent: torch.Tensor,
        negative_prompt_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate video frames using the diffusion model.

        Args:
            embedding: Text embedding tensor from T5 encoder
            condition_latent: Latent tensor from conditioning image or video
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            Tensor of generated video frames

        Note:
            Model and tokenizer are automatically offloaded after inference
            if offloading is enabled.
        """
        # Get video batch and state shape
        data_batch, state_shape = get_video_batch(
            model=self.model,
            prompt_embedding=embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames,
        )

        # Generate video frames
        video = generate_world_from_video(
            model=self.model,
            state_shape=self.model.state_shape,
            is_negative_prompt=True,
            data_batch=data_batch,
            guidance=self.guidance,
            num_steps=self.num_steps,
            seed=self.seed,
            condition_latent=condition_latent,
            num_input_frames=self.num_input_frames,
        )

        return video

    def _run_tokenizer_encoding(self, image_or_video_path: str) -> torch.Tensor:
        """
        Encode image to latent space

        Args:
            image_or_video_path: Path to conditioning image

        Returns:
            torch.Tensor: Latent tensor from tokenizer encoding
        """
        condition_latent = get_condition_latent(
            model=self.model,
            input_image_or_video_path=image_or_video_path,
            num_input_frames=self.num_input_frames,
            state_shape=self.model.state_shape,
        )

        return condition_latent

    def _run_model_with_offload(
        self,
        prompt_embedding: torch.Tensor,
        image_or_video_path: str,
        negative_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Generate world representation with automatic model offloading.

        Wraps the core generation process with model loading/offloading logic
        to minimize GPU memory usage during inference.

        Args:
            prompt_embedding: Text embedding tensor from T5 encoder
            image_or_video_path: Path to conditioning image or video
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            np.ndarray: Generated world representation as numpy array
        """
        if self.offload_tokenizer:
            self._load_tokenizer()

        condition_latent = self._run_tokenizer_encoding(image_or_video_path)

        if self.offload_network:
            self._load_network()

        sample = self._run_model(prompt_embedding, condition_latent, negative_prompt_embedding)

        if self.offload_network:
            self._offload_network()

        sample = self._run_tokenizer_decoding(sample)

        if self.offload_tokenizer:
            self._offload_tokenizer()

        return sample

    def generate(
        self,
        prompt: str,
        image_or_video_path: str,
        negative_prompt: Optional[str] = None,
    ) -> tuple[np.ndarray, str] | None:
        """Generate video from text prompt and optional image.

        Pipeline steps:
        1. Run safety checks on input prompt
        2. Enhance prompt using upsampler if enabled
        3. Run safety checks on upsampled prompt if applicable
        4. Convert prompt to embeddings
        5. Generate video frames using diffusion
        6. Run safety checks and apply face blur on generated video frames

        Args:
            prompt: Text description of desired video
            image_or_video_path: Path to conditioning image or video
            negative_prompt: Optional text to guide what not to generate

        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """
        log.info(f"Run with prompt: {prompt}")
        log.info(f"Run with image or video path: {image_or_video_path}")
        log.info(f"Run with negative prompt: {negative_prompt}")
        log.info(f"Run with prompt upsampler: {self.enable_prompt_upsampler}")

        if not self.enable_prompt_upsampler:
            log.info("Run guardrail on prompt")
            is_safe = self._run_guardrail_on_prompt_with_offload(prompt)
            if not is_safe:
                log.critical("Input text prompt is not safe")
                return None
            log.info("Pass guardrail on prompt")
        else:
            log.info("Run prompt upsampler on image or video, input prompt is not used")
            prompt = self._run_prompt_upsampler_on_prompt_with_offload(image_or_video_path=image_or_video_path)
            log.info("Run guardrail on upsampled prompt")
            is_safe = self._run_guardrail_on_prompt_with_offload(prompt)
            if not is_safe:
                log.critical("Upsampled text prompt is not safe")
                return None
            log.info("Pass guardrail on upsampled prompt")

        log.info("Run text embedding on prompt")
        if negative_prompt:
            prompts = [prompt, negative_prompt]
        else:
            prompts = [prompt]
        prompt_embeddings, _ = self._run_text_embedding_on_prompt_with_offload(prompts)
        prompt_embedding = prompt_embeddings[0]
        negative_prompt_embedding = prompt_embeddings[1] if negative_prompt else None
        log.info("Finish text embedding on prompt")

        # Generate video
        log.info("Run generation")
        video = self._run_model_with_offload(
            prompt_embedding,
            negative_prompt_embedding=negative_prompt_embedding,
            image_or_video_path=image_or_video_path,
        )
        log.info("Finish generation")

        log.info("Run guardrail on generated video")
        video = self._run_guardrail_on_video_with_offload(video)
        if video is None:
            log.critical("Generated video is not safe")
            return None
        log.info("Pass guardrail on generated video")

        return video, prompt


class DiffusionText2WorldMultiViewGenerationPipeline(DiffusionText2WorldGenerationPipeline):
    def __init__(
        self,
        inference_type: str,
        checkpoint_dir: str,
        checkpoint_name: str,
        prompt_upsampler_dir: Optional[str] = None,
        has_text_input: bool = True,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
        offload_prompt_upsampler: bool = False,
        offload_guardrail_models: bool = False,
        guidance: float = 7.0,
        num_steps: int = 35,
        height: int = 704,
        width: int = 1280,
        fps: int = 24,
        num_video_frames: int = 121,
        n_cameras: int = 6,
        frame_repeat_negative_condition: int = 10,
        seed: int = 0,
    ):
        """Initialize the diffusion multi-view world generation pipeline.

        Args:
            inference_type: Type of world generation ('text2world' or 'video2world')
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the diffusion transformer checkpoint to use
            prompt_upsampler_dir: Directory containing prompt upsampler model weights
            enable_prompt_upsampler: Whether to use prompt upsampling
            has_text_input: Whether the pipeline takes text input for world generation
            offload_network: Whether to offload diffusion transformer after inference
            offload_tokenizer: Whether to offload tokenizer after inference
            offload_text_encoder_model: Whether to offload T5 model after inference
            offload_prompt_upsampler: Whether to offload prompt upsampler
            offload_guardrail_models: Whether to offload guardrail models
            guidance: Classifier-free guidance scale
            num_steps: Number of diffusion sampling steps
            height: Height of output video
            width: Width of output video
            fps: Frames per second of output video
            num_video_frames: Number of frames to generate
            n_cameras: Number of cameras
            frame_repeat_negative_condition: Number of frames to repeat to be used as negative condition.
            seed: Random seed for sampling
        """
        assert inference_type in [
            "text2world",
            "video2world",
        ], "Invalid inference_type, must be 'text2world' or 'video2world'"

        self.n_cameras = n_cameras
        self.frame_repeat_negative_condition = frame_repeat_negative_condition
        super().__init__(
            inference_type=inference_type,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            prompt_upsampler_dir=prompt_upsampler_dir,
            enable_prompt_upsampler=False,
            has_text_input=has_text_input,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=offload_text_encoder_model,
            offload_prompt_upsampler=offload_prompt_upsampler,
            offload_guardrail_models=offload_guardrail_models,
            guidance=guidance,
            num_steps=num_steps,
            height=height,
            width=width,
            fps=fps,
            num_video_frames=num_video_frames,
            seed=seed,
        )

    def _load_model(self):
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos1/models/diffusion/config/config.py",
            model_class=DiffusionMultiCameraT2WModel,
        )

    def _run_tokenizer_decoding(self, sample: torch.Tensor) -> np.ndarray:
        """Decode latent samples to video frames using the tokenizer decoder.

        Args:
            sample: Latent tensor from diffusion model [B, C, T, H, W]

        Returns:
            np.ndarray: Decoded video frames as uint8 numpy array [T, H, W, C]
                        with values in range [0, 255]
        """
        # Decode video
        video = (1.0 + self.model.decode(sample)).clamp(0, 2) / 2  # [B, 3, T, H, W]
        video_segments = einops.rearrange(video, "b c (v t) h w -> b c v t h w", v=self.n_cameras)
        grid_video = torch.stack(
            [video_segments[:, :, i] for i in [1, 0, 2, 4, 3, 5]],
            dim=2,
        )
        grid_video = einops.rearrange(grid_video, "b c (h w) t h1 w1 -> b c t (h h1) (w w1)", h=2, w=3)
        grid_video = (grid_video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()
        video = (video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()

        return [grid_video, video]

    def _run_model(
        self,
        embedding: torch.Tensor,
        negative_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate video latents using the diffusion model.

        Args:
            embedding: Text embedding tensor from text encoder
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            torch.Tensor: Generated video latents before tokenizer decoding

        Note:
            The model and tokenizer are automatically offloaded after inference
            if offloading is enabled in the config.
        """
        # Get video batch and state shape
        data_batch, state_shape = get_video_batch_for_multi_camera_model(
            model=self.model,
            prompt_embedding=embedding,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames * len(embedding),  # number of cameras
            frame_repeat_negative_condition=self.frame_repeat_negative_condition,
        )

        # Generate video frames
        sample = generate_world_from_text(
            model=self.model,
            state_shape=state_shape,
            is_negative_prompt=False,
            data_batch=data_batch,
            guidance=self.guidance,
            num_steps=self.num_steps,
            seed=self.seed,
        )

        return sample

    def generate(
        self,
        prompt: dict,
    ) -> tuple[np.ndarray, str] | None:
        """Generate video from text prompt with optional negative prompt guidance.

        Pipeline steps:
        1. Convert prompt to embeddings
        2. Generate video frames using diffusion

        Args:
            prompt: A dictionary of text description of desired video.
        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """
        log.info(f"Run with prompt: {prompt}")

        prompts = [
            prompt["prompt"],
            prompt["prompt_left"],
            prompt["prompt_right"],
            prompt["prompt_back"],
            prompt["prompt_back_left"],
            prompt["prompt_back_right"],
        ]
        prompt_embeddings, _ = self._run_text_embedding_on_prompt_with_offload(prompts)
        log.info("Finish text embedding on prompt")

        # Generate video
        log.info("Run generation")
        videos = self._run_model_with_offload(
            prompt_embeddings,
        )
        log.info("Finish generation")

        return videos, prompt


class DiffusionVideo2WorldMultiViewGenerationPipeline(DiffusionText2WorldMultiViewGenerationPipeline):
    def __init__(
        self,
        inference_type: str,
        checkpoint_dir: str,
        checkpoint_name: str,
        prompt_upsampler_dir: Optional[str] = None,
        enable_prompt_upsampler: bool = True,
        has_text_input: bool = True,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
        offload_prompt_upsampler: bool = False,
        offload_guardrail_models: bool = False,
        guidance: float = 7.0,
        num_steps: int = 35,
        height: int = 704,
        width: int = 1280,
        fps: int = 24,
        num_video_frames: int = 121,
        seed: int = 0,
        num_input_frames: int = 1,
        n_cameras: int = 6,
        frame_repeat_negative_condition: int = 10,
    ):
        """Initialize diffusion world multi-view generation pipeline.

        Args:
            inference_type: Type of world generation ('text2world' or 'video2world')
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the diffusion transformer checkpoint to use
            prompt_upsampler_dir: Directory containing prompt upsampler model weights
            enable_prompt_upsampler: Whether to use prompt upsampling
            has_text_input: Whether the pipeline takes text input for world generation
            offload_network: Whether to offload diffusion transformer after inference
            offload_tokenizer: Whether to offload tokenizer after inference
            offload_text_encoder_model: Whether to offload T5 model after inference
            offload_prompt_upsampler: Whether to offload prompt upsampler
            offload_guardrail_models: Whether to offload guardrail models
            guidance: Classifier-free guidance scale
            num_steps: Number of diffusion sampling steps
            height: Height of output video
            width: Width of output video
            fps: Frames per second of output video
            num_video_frames: Number of frames to generate
            seed: Random seed for sampling
            num_input_frames: Number of latent conditions
        """
        self.num_input_frames = num_input_frames
        super().__init__(
            inference_type=inference_type,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            prompt_upsampler_dir=prompt_upsampler_dir,
            has_text_input=has_text_input,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=offload_text_encoder_model,
            offload_prompt_upsampler=offload_prompt_upsampler,
            offload_guardrail_models=offload_guardrail_models,
            guidance=guidance,
            num_steps=num_steps,
            height=height,
            width=width,
            fps=fps,
            num_video_frames=num_video_frames,
            seed=seed,
            n_cameras=n_cameras,
            frame_repeat_negative_condition=frame_repeat_negative_condition,
        )

    def _load_model(self):
        self.model = load_model_by_config(
            config_job_name=self.model_name,
            config_file="cosmos1/models/diffusion/config/config.py",
            model_class=DiffusionMultiCameraV2WModel,
        )

    def _run_model(
        self,
        embedding: torch.Tensor,
        condition_latent: torch.Tensor,
        negative_prompt_embedding: torch.Tensor | None = None,
        data_batch: dict = None,
        state_shape: list = None,
    ) -> torch.Tensor:
        """Generate video frames using the diffusion model.

        Args:
            embedding: Text embedding tensor from T5 encoder
            condition_latent: Latent tensor from conditioning image or video
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            Tensor of generated video frames

        Note:
            Model and tokenizer are automatically offloaded after inference
            if offloading is enabled.
        """
        # Generate video frames
        video = generate_world_from_video(
            model=self.model,
            state_shape=state_shape,
            is_negative_prompt=False,
            data_batch=data_batch,
            guidance=self.guidance,
            num_steps=self.num_steps,
            seed=self.seed,
            condition_latent=condition_latent,
            num_input_frames=self.num_input_frames,
        )

        return video

    def _run_tokenizer_encoding(self, image_or_video_path: str, state_shape: list) -> torch.Tensor:
        """
        Encode image to latent space

        Args:
            image_or_video_path: Path to conditioning image

        Returns:
            torch.Tensor: Latent tensor from tokenizer encoding
        """
        condition_latent, condition_frames = get_condition_latent_multi_camera(
            model=self.model,
            input_image_or_video_path=image_or_video_path,
            num_input_frames=self.num_input_frames,
            state_shape=state_shape,
        )

        return condition_latent, condition_frames

    def _run_model_with_offload(
        self,
        prompt_embedding: torch.Tensor,
        image_or_video_path: str,
        negative_prompt_embedding: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Generate world representation with automatic model offloading.

        Wraps the core generation process with model loading/offloading logic
        to minimize GPU memory usage during inference.

        Args:
            prompt_embedding: Text embedding tensor from T5 encoder
            image_or_video_path: Path to conditioning image or video
            negative_prompt_embedding: Optional embedding for negative prompt guidance

        Returns:
            np.ndarray: Generated world representation as numpy array
        """
        if self.offload_tokenizer:
            self._load_tokenizer()

        data_batch, state_shape = get_video_batch_for_multi_camera_model(
            model=self.model,
            prompt_embedding=prompt_embedding,
            height=self.height,
            width=self.width,
            fps=self.fps,
            num_video_frames=self.num_video_frames * len(prompt_embedding),  # number of cameras
            frame_repeat_negative_condition=self.frame_repeat_negative_condition,
        )

        condition_latent, condition_frames = self._run_tokenizer_encoding(image_or_video_path, state_shape)

        if self.offload_network:
            self._load_network()

        sample = self._run_model(
            prompt_embedding, condition_latent, negative_prompt_embedding, data_batch, state_shape
        )

        if self.offload_network:
            self._offload_network()

        sample = self._run_tokenizer_decoding(sample)

        if self.offload_tokenizer:
            self._offload_tokenizer()

        return sample

    def generate(
        self,
        prompt: dict,
        image_or_video_path: str,
    ) -> tuple[np.ndarray, str] | None:
        """Generate video from text prompt with optional negative prompt guidance.

        Pipeline steps:
        1. Convert prompt to embeddings
        2. Generate video frames using diffusion

        Args:
            prompt: A dictionary of text description of desired video.
        Returns:
            tuple: (
                Generated video frames as uint8 np.ndarray [T, H, W, C],
                Final prompt used for generation (may be enhanced)
            ), or None if content fails guardrail safety checks
        """
        log.info(f"Run with prompt: {prompt}")

        prompts = [
            prompt["prompt"],
            prompt["prompt_left"],
            prompt["prompt_right"],
            prompt["prompt_back"],
            prompt["prompt_back_left"],
            prompt["prompt_back_right"],
        ]
        prompt_embeddings, _ = self._run_text_embedding_on_prompt_with_offload(prompts)
        log.info("Finish text embedding on prompt")

        # Generate video
        log.info("Run generation")
        video = self._run_model_with_offload(
            prompt_embeddings,
            image_or_video_path=image_or_video_path,
        )
        log.info("Finish generation")

        return video, prompt
