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
from abc import ABC
from typing import Any

import numpy as np
import torch

from cosmos1.models.common.t5_text_encoder import CosmosT5TextEncoder
from cosmos1.models.guardrail.common import presets as guardrail_presets


class BaseWorldGenerationPipeline(ABC):
    def __init__(
        self,
        inference_type: str | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_name: str | None = None,
        has_text_input: bool = False,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
        offload_guardrail_models: bool = False,
    ):
        """Initialize base world generation pipeline.

        This abstract base class provides core functionality for world generation models including:
        - Model loading and initialization
        - Text encoding and embedding
        - Safety checks and content filtering
        - Memory management through model offloading

        Args:
            inference_type: The type of inference pipeline ("text2world" or "video2world")
            checkpoint_dir: Root directory containing model checkpoints
            checkpoint_name: Name of the specific checkpoint file to load
            has_text_input: Whether the pipeline takes text input for world generation
            offload_network: If True, moves main model to CPU after inference
            offload_tokenizer: If True, moves tokenizer to CPU after use
            offload_text_encoder_model: If True, moves T5 encoder to CPU after encoding
            offload_guardrail_models: If True, moves safety models to CPU after checks
        """
        self.inference_type = inference_type
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.guardrail_dir = "Cosmos-1.0-Guardrail"
        self.has_text_input = has_text_input

        # Add offloading flags
        self.offload_network = offload_network
        self.offload_tokenizer = offload_tokenizer
        self.offload_text_encoder_model = offload_text_encoder_model
        self.offload_guardrail_models = offload_guardrail_models

        # Initialize model instances
        self.text_guardrail = None
        self.video_guardrail = None
        self.text_encoder = None
        self.model = None

        self._load_model()

        if not self.offload_text_encoder_model:
            self._load_text_encoder_model()
        if not self.offload_guardrail_models:
            if self.has_text_input:
                self._load_text_guardrail()
            self._load_video_guardrail()
        if not self.offload_network:
            self._load_network()
        if not self.offload_tokenizer:
            self._load_tokenizer()

    def _load_tokenizer(self):
        pass

    def _load_network(self):
        pass

    def _load_model(self, checkpoint_name: str) -> Any:
        """Load the world generation model from a checkpoint.

        This abstract method must be implemented by subclasses to load their specific
        model architecture and weights.

        Args:
            checkpoint_name: Path to the model checkpoint file

        Returns:
            The loaded model instance

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    def _load_text_encoder_model(self):
        """Load the T5 text encoder model.

        Initializes and loads the T5 encoder model used for converting text prompts
        into embeddings that condition the world generation model.

        Returns:
            Loaded T5 text encoder model instance
        """
        self.text_encoder = CosmosT5TextEncoder(cache_dir=self.checkpoint_dir)

    def _load_text_guardrail(self):
        """Load text safety classifier models.

        Initializes models used for checking input prompts against safety policies.
        Models are loaded from the specified guardrail directory.
        """
        self.text_guardrail = guardrail_presets.create_text_guardrail_runner(
            checkpoint_dir=os.path.join(self.checkpoint_dir, self.guardrail_dir)
        )

    def _load_video_guardrail(self):
        """Load video safety classifier models.

        Initializes models used for validating generated video content against
        safety policies. Models are loaded from the specified guardrail directory.
        """
        self.video_guardrail = guardrail_presets.create_video_guardrail_runner(
            checkpoint_dir=os.path.join(self.checkpoint_dir, self.guardrail_dir)
        )

    def _offload_network(self):
        if self.model.model:
            del self.model.model
            self.model.model = None
            gc.collect()
            torch.cuda.empty_cache()

    def _offload_tokenizer(self):
        if self.model.tokenizer:
            del self.model.tokenizer
            self.model.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()

    def _offload_guardrail_models(self):
        """Offload safety classifier models to reduce memory usage.

        Moves safety models to CPU and clears GPU memory if they are no longer needed.
        This helps manage memory when processing multiple inputs sequentially.
        """
        if self.text_guardrail:
            del self.text_guardrail
            self.text_guardrail = None
        if self.video_guardrail:
            del self.video_guardrail
            self.video_guardrail = None
        gc.collect()
        torch.cuda.empty_cache()

    def _offload_text_encoder_model(self):
        """Offload T5 text encoder to reduce memory usage.

        Moves the T5 encoder to CPU and clears GPU memory after text encoding is complete.
        This helps manage memory when processing multiple inputs sequentially.
        """
        if self.text_encoder:
            del self.text_encoder
            self.text_encoder = None
            gc.collect()
            torch.cuda.empty_cache()

    def _run_model(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Generate world latents using the model.

        This abstract method must be implemented by subclasses to define their specific
        generation process.

        Args:
            *args: Variable positional arguments for model inference
            **kwargs: Variable keyword arguments for model inference

        Returns:
            torch.Tensor: Generated world representation tensor
        """
        pass

    def _run_model_with_offload(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Generate world representation with memory management.

        Handles loading the model before inference and offloading afterward if enabled.
        This helps minimize GPU memory usage during inference.

        Args:
            *args: Arguments passed to _run_model
            **kwargs: Keyword arguments passed to _run_model

        Returns:
            np.ndarray: Generated world representation as numpy array
        """
        pass

    def _run_guardrail_on_prompt(self, prompt: str) -> bool:
        """Check if prompt meets safety requirements.

        Validates the input prompt against safety policies using loaded guardrail models.

        Args:
            prompt: Raw text prompt to validate

        Returns:
            bool: True if prompt passes all safety checks, False otherwise
        """
        return guardrail_presets.run_text_guardrail(prompt, self.text_guardrail)

    def _run_guardrail_on_prompt_with_offload(self, prompt: str) -> bool:
        """Check prompt safety with memory management.

        Validates prompt safety while handling model loading/offloading to manage memory.

        Args:
            prompt: Raw text prompt to validate

        Returns:
            bool: True if prompt passes all safety checks, False otherwise
        """
        if self.offload_guardrail_models:
            self._load_text_guardrail()

        is_safe = self._run_guardrail_on_prompt(prompt)

        if self.offload_guardrail_models:
            self._offload_guardrail_models()

        return is_safe

    def _run_guardrail_on_video(self, video: np.ndarray) -> np.ndarray | None:
        """Check if video meets safety requirements.

        Validates generated video content against safety policies using guardrail models.

        Args:
            video: Video frames to validate

        Returns:
            np.ndarray: Processed video if safe, None if unsafe
        """
        return guardrail_presets.run_video_guardrail(video, self.video_guardrail)

    def _run_guardrail_on_video_with_offload(self, video: np.ndarray) -> np.ndarray | None:
        """Check if generated video meets safety requirements.

        Args:
            video: Video frames to validate

        Returns:
            np.ndarray: Processed video frames if safe, None otherwise

        Note:
            Guardrail models are offloaded after checks if enabled.
        """
        if self.offload_guardrail_models:
            self._load_video_guardrail()

        video = self._run_guardrail_on_video(video)

        if self.offload_guardrail_models:
            self._offload_guardrail_models()
        return video

    def _run_text_embedding_on_prompt(
        self, prompts: list[str], **kwargs: Any
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Convert text prompts to embeddings.

        Processes text prompts into embedding tensors that condition the generation model.

        Args:
            prompts: List of text prompts to encode
            **kwargs: Additional arguments for text encoding

        Returns:
            tuple containing:
                - List of text embedding tensors for each prompt
                - List of attention masks for each embedding
        """

        embeddings = []
        masks = []
        for prompt in prompts:
            embedding, mask = self.text_encoder.encode_prompts(
                [prompt],
                **kwargs,
            )
            embeddings.append(embedding)
            masks.append(mask)

        return embeddings, masks

    def _run_text_embedding_on_prompt_with_offload(
        self, prompts: list[str], **kwargs: Any
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Convert text prompt into embeddings using T5 encoder.

        Args:
            prompt: Processed and validated text prompt

        Returns:
            Text embedding tensor to condition diffusion model

        Note:
            T5 model is offloaded after encoding if enabled.
        """
        if self.offload_text_encoder_model:
            self._load_text_encoder_model()

        embeddings, masks = self._run_text_embedding_on_prompt(prompts, **kwargs)

        if self.offload_text_encoder_model:
            self._offload_text_encoder_model()
        return embeddings, masks

    def _run_tokenizer_decoding(self, samples: torch.Tensor) -> np.ndarray:
        """Decode model outputs into final world representation.

        This abstract method must be implemented by subclasses to convert raw model
        outputs into their specific world representation format.

        Args:
            samples: Raw output tensor from the generation model

        Returns:
            np.ndarray: Decoded world representation
        """
        pass

    def generate(self, *args: Any, **kwargs: Any):
        """Generate world representation.

        This abstract method must be implemented by subclasses to convert raw model
        outputs into their specific world representation format.

        Args:
            *args: Variable positional arguments for model inference
            **kwargs: Variable keyword arguments for model inference
        """
        pass
