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

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
from safetensors.torch import load_file
from torch.nn.modules.module import _IncompatibleKeys

from cosmos1.models.autoregressive.configs.base.model import ModelConfig
from cosmos1.models.autoregressive.configs.base.tokenizer import TokenizerConfig
from cosmos1.models.autoregressive.modules.mm_projector import MultimodalProjector
from cosmos1.models.autoregressive.networks.transformer import Transformer
from cosmos1.models.autoregressive.networks.vit import VisionTransformer, get_vit_config
from cosmos1.models.autoregressive.tokenizer.tokenizer import DiscreteMultimodalTokenizer, update_vocab_size
from cosmos1.models.autoregressive.utils.checkpoint import (
    get_partial_state_dict,
    process_state_dict,
    substrings_to_ignore,
)
from cosmos1.models.autoregressive.utils.sampling import decode_n_tokens, decode_one_token, prefill
from cosmos1.utils import log, misc


class AutoRegressiveModel(torch.nn.Module):
    """
    A class to build and use a AutoRegressiveModel model for text generation.

    Methods:
        build: Build a AutoRegressiveModel instance by initializing and loading a model checkpoint.
        generate: Generate text sequences based on provided prompts using the language generation model.
    """

    def __init__(
        self,
        model: Transformer = None,
        tokenizer: DiscreteMultimodalTokenizer = None,
        config: ModelConfig = None,
        vision_encoder: VisionTransformer = None,
        mm_projector: MultimodalProjector = None,
    ):
        """
        Initialize the AutoRegressiveModel instance with a model and tokenizer.

        Args:
            model (Transformer): The Transformer model for text generation.
            tokenizer (Tokenizer): The tokenizer for encoding and decoding text.
            config (Config): The configuration for the AutoRegressiveModel model.
            vision_encoder (VisionTransformer): The vision encoder for the AutoRegressiveModel model.
            mm_projector (MultimodalProjector): The multi-modal projector for the AutoRegressiveModel model.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.vision_encoder = vision_encoder
        self.mm_projector = mm_projector

    @property
    def precision(self):
        return self.model.precision

    def get_num_params(
        self,
    ) -> int:
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def load_ar_model(
        self,
        tokenizer_config,
    ):
        """
        Load the AR model.
        """
        model_config = self.config
        ckpt_path = model_config.ckpt_path
        with misc.timer(f"loading checkpoint from {ckpt_path}"):
            if ckpt_path.endswith("safetensors"):
                # Load with safetensors API
                checkpoint = load_file(ckpt_path, device="cpu")
            else:
                # The pytorch version
                checkpoint = torch.load(
                    ckpt_path,
                    map_location="cpu",
                    mmap=True,  # load the checkpoint in memory-mapped mode
                    weights_only=True,
                )
        llm_checkpoint = checkpoint["model"] if "model" in checkpoint else checkpoint
        orig_precision = torch.get_default_dtype()
        precision = getattr(torch, model_config.precision)
        torch.set_default_dtype(precision)
        log.debug(f"Setting torch default dtype to {precision}")

        model = Transformer(
            params=model_config,
            tokenizer_config=tokenizer_config,
        )
        log.debug(
            f"tokenizer tokenizer_config.video_tokenizer.vocab_size {tokenizer_config.video_tokenizer.vocab_size}"
        )
        vocab_size = update_vocab_size(
            existing_vocab_size=0,
            to_be_added_vocab_size=tokenizer_config.video_tokenizer.vocab_size,
            training_type=tokenizer_config.training_type,
            add_special_tokens=False,
        )
        log.debug(
            f"tokenizer tokenizer_config.video_tokenizer.vocab_size {tokenizer_config.video_tokenizer.vocab_size}  vocab_size {vocab_size}"
        )
        # Perform vocab expansion
        if vocab_size > model.vocab_size:
            log.debug(f"Expanding vocab size to {vocab_size}")
            # For text-to-video training, we only expand the embedding layer but not the output (unembedding) layer,
            expand_output_layer = not (tokenizer_config.training_type == "text_to_video")
            model.expand_vocab(
                vocab_size,
                init_method="gaussian",
                expand_output_layer=expand_output_layer,
            )
        # Remove the "model." prefix in the state_dict
        llm_checkpoint = process_state_dict(llm_checkpoint, prefix_to_remove="model.")
        with misc.timer("loading state_dict into model"):
            missing_keys, _ = model.load_state_dict(llm_checkpoint, strict=True)
        # Remove keys with "_extra_state" suffix in missing_keys (defined by TransformerEngine for FP8 usage)
        missing_keys = [k for k in missing_keys if not k.endswith("_extra_state")]
        assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"

        self.model = model.to(precision).to("cuda")
        torch.set_default_dtype(orig_precision)  # Reset the default dtype to the original value

    def load_tokenizer(self, tokenizer_config):
        """
        Load the tokenizer.
        """
        self.tokenizer = DiscreteMultimodalTokenizer(tokenizer_config)

    @staticmethod
    def build(
        model_config: ModelConfig = ModelConfig(),
        tokenizer_config: TokenizerConfig = None,
    ) -> "AutoRegressiveModel":
        """
        Build a AutoRegressiveModel instance by initializing and loading a model checkpoint.

        Args:
            model_config (ModelConfig, optional): The model configuration for the AutoRegressiveModel instance. Defaults to ModelConfig().
            tokenizer_config (TokenizerConfig, optional): The tokenizer configuration for the AutoRegressiveModel instance. Defaults to None.
            download_rank_sync (bool, optional): Whether to download the checkpoint in a rank-synchronized manner. Defaults to True.
        Returns:
            AutoRegressiveModel: An instance of the AutoRegressiveModel class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory.

        Note:
            This method sets the device to CUDA and loads the pre-trained model and tokenizer.
        """
        # Initialize model configuration parameters
        config_params = {}

        # Load checkpoint and model parameters

        if model_config.ckpt_path is None:
            # If ckpt_path is not provided, we assume the model checkpoint is saved in the ckpt_dir
            ckpt_dir = model_config.ckpt_dir

            # We prioritize safetensors version over the pytorch version, since the former is
            # much faster for checkpoint loading.
            checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))
            if len(checkpoints) == 0:
                checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

            assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
            assert (
                len(checkpoints) == 1
            ), f"multiple checkpoint files found in {ckpt_dir} (currently only one is supported)"
            ckpt_path = str(checkpoints[0])  # Assuming single checkpoint for non-parallel case

            if os.path.exists(Path(ckpt_dir) / "config.json"):
                with open(Path(ckpt_dir) / "config.json", "r") as f:
                    config_params = json.loads(f.read())
            else:
                log.info(
                    f"No params.json found in the checkpoint directory ({ckpt_dir}). " f"Using default model config."
                )

        else:
            # If ckpt_path is provided, we load the model from the specified path,
            # and use the default model configuration
            ckpt_path = model_config.ckpt_path

        for key, value in config_params.items():
            if hasattr(model_config, key):
                # Override the default model configuration with the parameters from the checkpoint
                setattr(model_config, key, value)

        with misc.timer(f"loading checkpoint from {ckpt_path}"):
            if ckpt_path.endswith("safetensors"):
                # Load with safetensors API
                checkpoint = load_file(ckpt_path, device="cpu")
            else:
                # The pytorch version
                checkpoint = torch.load(
                    ckpt_path,
                    map_location="cpu",
                    mmap=True,  # load the checkpoint in memory-mapped mode
                    weights_only=True,
                )
        llm_checkpoint = checkpoint["model"] if "model" in checkpoint else checkpoint

        if model_config.vision_encoder is not None:
            # Take the LLM weights (starting with "model.") from the VLM checkpoint
            llm_checkpoint = get_partial_state_dict(llm_checkpoint, prefix="model.")
        if model_config.vision_encoder is not None:
            # For vanilla VLM ckpt before fine-tuning, `checkpoint['model']` only contains LLM weights, and `checkpoint['vision_encoder']`
            #   and `checkpoint['mm_projector']` are both for those weights
            # For fine-tuned VLM ckpt, `checkpoint['model']` contains all LLM, mm_projector and vision_encoder weights
            if "vision_encoder" in checkpoint:
                log.debug("Using pretrained vision_encoder")
                vit_checkpoint = checkpoint["vision_encoder"]
            else:
                log.debug("Using fine-tuned vision_encoder")
                vit_checkpoint = get_partial_state_dict(llm_checkpoint, prefix="vision_encoder.")
                vit_checkpoint = process_state_dict(vit_checkpoint, prefix_to_remove="vision_encoder.")
            if "mm_projector" in checkpoint:
                log.debug("Using pretrained mm_projector")
                projector_checkpoint = checkpoint["mm_projector"]
            else:
                log.debug("Using fine-tuned mm_projector")
                projector_checkpoint = get_partial_state_dict(llm_checkpoint, prefix="mm_projector.")
                projector_checkpoint = process_state_dict(projector_checkpoint, prefix_to_remove="mm_projector.")
            assert (
                len(vit_checkpoint) > 0 and len(projector_checkpoint) > 0
            ), "vit_checkpoint and projector_checkpoint cannot be empty. We do not support random initialization for vision_encoder and mm_projector."

        tokenizer = DiscreteMultimodalTokenizer(tokenizer_config)
        orig_precision = torch.get_default_dtype()
        precision = getattr(torch, model_config.precision)
        torch.set_default_dtype(precision)
        log.debug(f"Setting torch default dtype to {precision}")

        model = Transformer(
            params=model_config,
            tokenizer_config=tokenizer_config,
        )
        model_kwargs = {}

        if model_config.vision_encoder is not None:
            assert model_config.mm_projector is not None, "mm_projector must be provided if vision_encoder is provided."
            vit_config = get_vit_config(model_config.vision_encoder)
            vision_encoder = VisionTransformer.build(
                vit_config,
            )

            mm_projector = MultimodalProjector(
                mm_projector_type=model_config.mm_projector, in_dim=vit_config["dim"], out_dim=model_config["dim"]
            )
            model_kwargs.update({"vision_encoder": vision_encoder, "mm_projector": mm_projector})

        # Perform vocab expansion
        if tokenizer.vocab_size > model.vocab_size:
            log.debug(f"Expanding vocab size to {tokenizer.vocab_size}")
            # For text-to-video training, we only expand the embedding layer but not the output (unembedding) layer,
            expand_output_layer = not (tokenizer.training_type == "text_to_video")
            model.expand_vocab(
                tokenizer.vocab_size,
                init_method="gaussian",
                expand_output_layer=expand_output_layer,
            )

        # Remove the "model." prefix in the state_dict
        llm_checkpoint = process_state_dict(llm_checkpoint, prefix_to_remove="model.")
        with misc.timer("loading state_dict into model"):
            missing_keys, unexpected_keys = model.load_state_dict(llm_checkpoint, strict=True)
        # Remove keys with "_extra_state" suffix in missing_keys (defined by TransformerEngine for FP8 usage)
        missing_keys = [k for k in missing_keys if not k.endswith("_extra_state")]
        assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"

        if model_config.vision_encoder is not None:
            vision_encoder.load_state_dict(vit_checkpoint)
            mm_projector.load_state_dict(projector_checkpoint)
            if model_config.vision_encoder_in_channels != 3:
                vision_encoder.expand_in_channels(model_config.vision_encoder_in_channels)

        model = model.to(precision)  # ensure model parameters are in the correct precision
        log.debug(f"Model config: {model_config}")

        model_class = AutoRegressiveModel

        torch.set_default_dtype(orig_precision)  # Reset the default dtype to the original value

        return model_class(model, tokenizer, model_config, **model_kwargs)

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: List[List[int]] | torch.Tensor,
        max_gen_len: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_gen_seq: int = 1,
        logprobs: bool = False,
        echo: bool = False,
        seed: int = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        compile_sampling: bool = True,
        compile_prefill: bool = False,
        verbose: bool = True,
        stop_tokens: Optional[Set[int]] = None,
        images: Optional[torch.Tensor] = None,
    ):
        """
        Autoregressive generation built upon the gpt-fast implementation (https://github.com/pytorch-labs/gpt-fast).

        Args:
            prompt_tokens (List[List[int]] | torch.Tensor): A single prompt of shape (1, seq_len).
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_k (int, optional): Top-k value for top-k sampling. Defaults to None.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to None.
            num_gen_seq (int, optional): Number of outputs to generate given the same prompt. Defaults to 1. When temperature == 0, num_gen_seq must be 1 because the generation is deterministic.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
            logit_clipping_range (list, optional): Range of logits to clip. Defaults to [].
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            compile_sampling (bool, optional): Flag indicating whether to compile the decoding function. Defaults to True.
            compile_prefill (bool, optional): Flag indicating whether to compile the prefill function. Defaults to False.
            verbose (bool, optional): Flag indicating whether to print the the time. Defaults to False.
        """
        assert top_k is None or top_p is None, f"Only one of top_k ({top_k} or top_p ({top_p} should be specified."
        if temperature == 0:
            top_p, top_k = None, None
            log.debug("Setting top_p and top_k to None because temperature is 0")
        if top_p is not None:
            log.debug(f"Using top-p sampling with p={top_p} and temperature={temperature}")
        elif top_k is not None:
            log.debug(f"Using top-k sampling with k={top_k} and temperature={temperature}")
        else:
            log.debug("Not applying top-k or top-p sampling. Will use top-k sampling with k=None")

        orig_precision = torch.get_default_dtype()
        torch.set_default_dtype(self.precision)

        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        # Experimental features to reduce compilation times, will be on by default in future
        torch._inductor.config.fx_graph_cache = True

        if seed is not None:
            misc.set_random_seed(seed)

        assert not logprobs, "logprobs are not supported for fast_generate yet"
        # Examine if the function prefil and decode_one_token functions are compiled yet. If not, compile them based on the flags
        if compile_sampling and not getattr(self, "inference_decode_compiled", False):
            self.decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
            self.inference_decode_compiled = True
            log.info("Compiled AR sampling function. Note: the first run will be slower due to compilation")
        if compile_prefill and not getattr(self, "inference_prefill_compiled", False):
            self.prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
            self.inference_prefill_compiled = True
            log.info("Compiled prefill function. Note: the first run will be slower due to compilation")

        if not hasattr(self, "decode_one_token"):
            self.decode_one_token = decode_one_token
        if not hasattr(self, "prefill"):
            self.prefill = prefill

        # Initialization and Assertions
        if isinstance(self.model.params, list):
            # During training, model.params is a list
            log.debug(
                f"Find self.model.params is a list, use self.config instead. Get max_batch_size={self.config.max_batch_size}, max_seq_len={self.config.max_seq_len}"
            )
            params = self.config
        else:
            params = self.model.params
        if isinstance(prompt_tokens, list):
            prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device="cuda")
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens.view(1, -1)
        else:
            assert prompt_tokens.ndim == 2, f"prompt_tokens has shape {prompt_tokens.shape}"
        batch_size, prompt_len = prompt_tokens.shape
        total_len = min(params.max_seq_len, max_gen_len + prompt_len)
        if max_gen_len + prompt_len > params.max_seq_len:
            log.warning(
                f"max_gen_len + prompt_len={max_gen_len + prompt_len} exceeds max_seq_len={params.max_seq_len}, truncate max_gen_len to {params.max_seq_len - prompt_len}"
            )
            max_gen_len = params.max_seq_len - prompt_len

        if context_mask is not None:
            context_mask = context_mask.to(dtype=torch.bool)
            if context_mask.ndim == 2:
                assert (
                    context_mask.shape[0] == batch_size
                ), f"batch_size mismatch: {context_mask.shape[0]} != {batch_size}"
                # Unsqueeze it to make it of shape [batch_size, 1, 1, context_seq_len]
                context_mask = context_mask.view(batch_size, 1, 1, -1)

        if num_gen_seq > 1:
            assert (
                batch_size == 1
            ), f"num_gen_seq > 1 is only supported for a single prompt, got {len(prompt_tokens)} prompts"
            log.debug(f"Generating {num_gen_seq} sequences with the same prompt")
            assert (
                num_gen_seq <= params.max_batch_size
            ), f"num_gen_seq={num_gen_seq} exceeds max_batch_size={params.max_batch_size}"
            # repeat the prompt tokens for num_gen_seq times
            prompt_tokens = prompt_tokens.repeat(num_gen_seq, 1)
            assert prompt_tokens.shape == (
                num_gen_seq,
                prompt_len,
            ), f"prompt_tokens must be of shape (num_gen_seq, seq_len), got {prompt_tokens.shape}"
            batch_size = len(prompt_tokens)

        # create an empty tensor of the expected final shape and fill in the current tokens
        empty = torch.empty(batch_size, total_len, dtype=prompt_tokens.dtype, device=prompt_tokens.device)
        empty[:, :prompt_len] = prompt_tokens
        seq = empty
        input_pos = torch.arange(0, prompt_len, device="cuda")

        if verbose:
            prefill_start = time.time()

        if images is not None:
            images = images.to(device=prompt_tokens.device, dtype=torch.bfloat16)
            prompt_token_embeddings = self.embed_vision_language_features(prompt_tokens, images)
        else:
            prompt_token_embeddings = None

        if context is not None:
            context = context.to(device=prompt_tokens.device, dtype=self.precision)

        # Prefill stage
        next_token = self.prefill(
            self.model,
            input_pos=input_pos,
            tokens=prompt_tokens if prompt_token_embeddings is None else None,
            token_embeddings=prompt_token_embeddings,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            context=context,
            context_mask=context_mask,
        )
        if verbose:
            prefill_time = time.time() - prefill_start

        seq[:, [prompt_len]] = next_token.to(dtype=seq.dtype)
        input_pos = torch.tensor([prompt_len], dtype=torch.long, device="cuda")
        stop_tokens = self.tokenizer.stop_tokens if stop_tokens is None else stop_tokens
        stop_tokens = torch.tensor(list(stop_tokens), dtype=torch.long, device="cuda")

        if verbose:
            decode_start = time.time()
        # Decode stage
        generated_tokens = decode_n_tokens(
            self.model,
            next_token.view(batch_size, -1),
            input_pos,
            max_gen_len - 1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_tokens=stop_tokens,
            decode_one_token_function=self.decode_one_token,
            context=context,
            context_mask=context_mask,
        )
        gen_len = len(generated_tokens)
        if verbose:
            decode_time = time.time() - decode_start
            prefill_throughput = prompt_len / prefill_time
            decode_throughput = gen_len / decode_time
            log.debug(f"[Prefill] Time: {prefill_time:.2f}s; Throughput: {prefill_throughput:.2f} tokens/s")
            log.debug(f"[Decode] Time: {decode_time:.2f}s; Throughput: {decode_throughput:.2f} tokens/s")

        generated_tokens = torch.cat(generated_tokens, dim=1)

        log.debug(f"generated_tokens: {generated_tokens.shape}")
        seq = seq[:, : prompt_len + 1 + gen_len]
        seq[:, prompt_len + 1 :] = generated_tokens
        if not echo:
            seq = seq[:, prompt_len:]

        torch.set_default_dtype(orig_precision)  # Reset the default dtype to the original value

        return seq, None

    def embed_vision_language_features(self, input_ids: torch.Tensor, images: torch.tensor) -> torch.Tensor:
        """
        Embed vision and language features into a combined representation.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            images (torch.tensor): Input images.

        Returns:
            torch.Tensor: Combined vision-language features.

        Raises:
            AssertionError: If vision encoder or mm projector is not initialized,
                            or if dimensions mismatch.
        """
        # Ensure vision encoder and mm projector are initialized
        assert self.vision_encoder is not None
        assert self.mm_projector is not None

        # Get image token ID and validate it
        image_token_id = self.vision_encoder.image_token_id
        assert isinstance(image_token_id, int) and image_token_id >= 0, f"Invalid image_token_id: {image_token_id}"

        # Identify text and image locations in the input
        text_locations = input_ids != image_token_id
        image_locations = input_ids == image_token_id

        # Process text features
        text_features = self.model.tok_embeddings(input_ids[text_locations])

        # Process image features
        images = images.to(device=text_features.device, dtype=text_features.dtype)
        vit_outputs = self.vision_encoder(images)
        image_features = self.mm_projector(vit_outputs)

        # Get dimensions
        B, seq_len = input_ids.shape
        N_total = B * seq_len
        N_txt, D_txt = text_features.shape
        N_img, N_patch, D_img = image_features.shape

        # Reshape image features
        image_features = image_features.reshape(N_img * N_patch, D_img)

        # Validate dimensions
        assert D_txt == D_img, f"Text features dim {D_txt} should be equal to image features dim {D_img}"
        assert (
            N_total == N_txt + N_img * N_patch
        ), f"seq_len {seq_len} should be equal to N_txt + N_img*N_Patch {(N_txt, N_img * N_patch, image_locations.sum().item())}"

        # Combine text and image features
        combined_features = torch.empty(
            (B, seq_len, D_txt),
            dtype=text_features.dtype,
            device=text_features.device,
        )
        combined_features[text_locations, :] = text_features
        combined_features[image_locations, :] = image_features

        return combined_features

    def state_dict(self, *args, **kwargs):
        """
        Process the state dict (e.g., remove "_extra_state" keys imposed by TransformerEngine for FP8).
        """
        state_dict = super().state_dict(*args, **kwargs)
        return process_state_dict(state_dict)

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False):
        """
        Ignore the missing keys with substrings matching `substring_to_ignore` (e.g., "_extra_state" keys imposed by
        TransformerEngine for FP8).
        """
        state_dict = process_state_dict(state_dict)
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False, assign=assign)
        actual_missing_keys = []
        for key in missing_keys:
            if not any(substring in key for substring in substrings_to_ignore):
                actual_missing_keys.append(key)
        if strict:
            if len(actual_missing_keys) > 0 or len(unexpected_keys) > 0:
                raise ValueError(f"Missing keys: {actual_missing_keys}\n\nUnexpected keys: {unexpected_keys}")
        return _IncompatibleKeys(actual_missing_keys, unexpected_keys)
