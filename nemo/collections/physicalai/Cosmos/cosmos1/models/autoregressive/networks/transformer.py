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

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys

from cosmos1.models.autoregressive.modules.attention import Attention
from cosmos1.models.autoregressive.modules.embedding import (
    RotaryPositionEmbeddingPytorchV1,
    RotaryPositionEmbeddingPytorchV2,
    SinCosPosEmbAxisTE,
)
from cosmos1.models.autoregressive.modules.mlp import MLP
from cosmos1.models.autoregressive.modules.normalization import create_norm
from cosmos1.models.autoregressive.utils.checkpoint import process_state_dict, substrings_to_ignore
from cosmos1.models.autoregressive.utils.misc import maybe_convert_to_namespace
from cosmos1.utils import log


class TransformerBlock(nn.Module):
    """
    A single transformer block consisting of an attention layer and a feed-forward layer.
    """

    def __init__(self, layer_id: int, args=None):
        """
        Initializes the TransformerBlock module.

        Args:
            layer_id: The ID of the transformer block.
            args: The model arguments containing hyperparameters.
        """
        super().__init__()
        args = maybe_convert_to_namespace(args)
        attention_args = {
            "n_heads": args["n_heads"],
            "n_kv_heads": args["n_kv_heads"],
            "dim": args["dim"],
            "context_dim": None,
            "max_batch_size": args["max_batch_size"],
            "max_seq_len": args["max_seq_len"],
            "use_qk_normalization": args["use_qk_normalization"],
            "causal_mask": args["causal_mask"],
            "head_dim": args["head_dim"],
            "fuse_qkv": getattr(args, "fuse_qkv", False),
            "precision": getattr(args, "precision", "bfloat16"),
            "attn_type": getattr(args, "attn_type", "self"),
        }
        self.attention = Attention(**attention_args)

        self.has_cross_attention = False
        self.cross_attention, self.cross_attention_norm = None, None

        if args["insert_cross_attn"] and layer_id % args["insert_cross_attn_every_k_layers"] == 0:
            self.has_cross_attention = True
            cross_attention_args = attention_args.copy()
            cross_attention_args.update({"context_dim": args["context_dim"], "fuse_qkv": False, "attn_type": "cross"})
            self.cross_attention = Attention(**cross_attention_args)
            self.cross_attention_norm = create_norm(args["norm_type"], dim=args["dim"], eps=args["norm_eps"])

        self.feed_forward = MLP(
            dim=args["dim"],
            hidden_dim=args["ffn_hidden_size"],
        )
        self.layer_id = layer_id
        self.attention_norm = create_norm(args["norm_type"], dim=args["dim"], eps=args["norm_eps"])
        self.ffn_norm = create_norm(args["norm_type"], dim=args["dim"], eps=args["norm_eps"])

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionEmbeddingPytorchV2,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs the forward pass of the TransformerBlock module.

        Args:
            x: The input tensor.
            input_pos: The position of the current sequence. Used in inference (with KV cache) only.
            freqs_cis: The precomputed frequency values for rotary position embeddings.
            mask: The attention mask tensor.
            context (Optional[torch.Tensor]): The context tensor added via cross-attn.
            context_mask (Optional[torch.Tensor]): The context cross-attn mask tensor.

        Returns:
            The output tensor after applying the transformer block.
        """
        # Apply attention and residual connection
        h = x + self.attention(self.attention_norm(x), rope=rope, input_pos=input_pos, mask=mask)

        # If insert cross-attention, apply CA and residual connection
        if self.has_cross_attention:
            h = h + self.cross_attention(
                self.cross_attention_norm(h), rope=rope, input_pos=input_pos, mask=context_mask, context=context
            )

        # Apply feed-forward network and residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        """
        Initializes the weights of the transformer block.
        """
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)

        if self.has_cross_attention:
            self.cross_attention_norm.reset_parameters()
            self.cross_attention.init_weights(self.weight_init_std)
            # zero-init the final output layer of cross-attention
            # nn.init.zeros_(self.cross_attention.wo.weight)


class Transformer(nn.Module):
    """
    The Transformer network consisting of transformer blocks.
    """

    def __init__(self, params, tokenizer_config=None, init_weights: bool = True):
        """
        Initializes the Transformer module.

        Args:
            params: The model parameters containing hyperparameters.
            tokenizer_config: The model tokenizer configuration.
            init_weights (bool): Whether to initialize the weights of the transformer following
                TorchTitan's Llama3 initialization scheme.
        """
        super().__init__()
        # Check if self.params is an OmegaConf DictConfig instance
        self.params = maybe_convert_to_namespace(params)
        self.vocab_size = params["vocab_size"]
        self.n_layers = params["n_layers"]
        self.precision = getattr(torch, params["precision"])
        self.tokenizer_config = tokenizer_config
        self.num_video_frames = params["num_video_frames"]

        # Token embeddings
        self.tok_embeddings = self._create_token_embeddings()
        self.rope_config = self._create_rope_config()

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(layer_id, self.params).to(self.precision) for layer_id in range(self.n_layers)]
        )

        # Final layer normalization
        self.norm = create_norm(self.params["norm_type"], dim=self.params["dim"], eps=self.params["norm_eps"]).to(
            self.precision
        )
        if self.params["pytorch_rope_version"] == "v1":
            self.rope = RotaryPositionEmbeddingPytorchV1(**self.rope_config)
        elif self.params["pytorch_rope_version"] == "v2":
            # Rotary position embeddings
            training_type = self.tokenizer_config.training_type if self.tokenizer_config is not None else None
            self.rope = RotaryPositionEmbeddingPytorchV2(
                seq_len=self.params["max_seq_len"], training_type=training_type, **self.rope_config
            )
        else:
            raise ValueError(f"Invalid PyTorch RoPE version: {self.params['pytorch_rope_version']}")
        # Causal mask
        self.causal_mask = torch.tril(
            torch.ones(self.params["max_seq_len"], self.params["max_seq_len"], dtype=torch.bool)
        ).cuda()

        # Output projection
        self.output = self._create_output_projection()

        # Freeze network parameters for finetuning w/ cross-attention
        self.has_cross_attention = getattr(params, "insert_cross_attn", False)

        # Absolute position embeddings
        if self.params["apply_abs_pos_emb"]:
            self.pos_emb_config = self._create_abs_pos_emb_config()
            self.pos_emb, self.abs_pos_emb = self._initialize_abs_pos_emb()

    def _create_rope_config(self) -> Dict:
        shape_map = {
            "3D": self.params["video_latent_shape"],
            "1D": None,
        }
        latent_shape = shape_map.get(self.params["rope_dim"], None)
        head_dim = self.params["head_dim"]
        if head_dim is None:
            head_dim = self.params["dim"] // self.params["n_heads"]
        return {
            "dim": head_dim,
            "max_position_embeddings": self.params["max_seq_len"],
            "original_max_position_embeddings": self.params["original_seq_len"],
            "rope_theta": self.params["rope_theta"],
            "apply_yarn": self.params["apply_yarn"],
            "scale": self.params["yarn_scale"],
            "beta_fast": self.params["yarn_beta_fast"],
            "beta_slow": self.params["yarn_beta_slow"],
            "rope_dim": self.params["rope_dim"],
            "latent_shape": latent_shape,
            "original_latent_shape": self.params["original_latent_shape"],
            "pad_to_multiple_of": self.params["pad_to_multiple_of"],
        }

    def _create_abs_pos_emb_config(self):
        shape_map = {
            "3D": self.params["video_latent_shape"],
            "1D": None,
        }
        latent_shape = shape_map.get(self.params["rope_dim"], None)
        return {
            "dim": self.params["dim"],
            "latent_shape": latent_shape,
            "pad_to_multiple_of": self.params["pad_to_multiple_of"],
        }

    def _create_token_embeddings(self, vocab_size: int = None):
        """
        Create token embeddings.

        Returns:
            nn.Module: Token embeddings module.
        """
        if vocab_size is None:
            vocab_size = self.params["vocab_size"]
        return nn.Embedding(vocab_size, self.params["dim"]).to(self.precision)

    def _create_output_projection(self, vocab_size: int = None):
        """
        Create the output projection layer.

        Args:
            vocab_size (int): Vocabulary size (to override the default vocab size).
        Returns:
            LinearTE: Output projection layer.
        """
        if vocab_size is None:
            vocab_size = self.params["vocab_size"]
        return nn.Linear(self.params["dim"], vocab_size, bias=False).to(self.precision)

    def _initialize_abs_pos_emb(self):
        pos_emb = SinCosPosEmbAxisTE(**self.pos_emb_config)
        training_type = self.tokenizer_config.training_type if self.tokenizer_config is not None else None
        abs_pos_emb = pos_emb.forward(training_type=training_type)
        return pos_emb, abs_pos_emb

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs the forward pass of the Transformer module.

        Args:
            tokens (torch.Tensor, optional): The input tensor of token IDs.
            input_pos (Optional[torch.Tensor]): The position of the current sequence. Used in inference with KV cache.
            token_embeddings (torch.Tensor, optional): Precomputed token embeddings. If provided, tokens should be None.
            context (Optional[torch.Tensor]): The context tensor added via cross-attn.
            context_mask (Optional[torch.Tensor]): The context cross-attn mask tensor.
        Returns:
            The output tensor after applying the transformer layers.
        """
        # Token embeddings
        assert (
            tokens is None or token_embeddings is None
        ), "Either tokens or token_embeddings should be provided, not both."

        if token_embeddings is None:
            seq_len = tokens.shape[1]
            h = self.tok_embeddings(tokens)
        else:
            seq_len = token_embeddings.shape[1]
            h = token_embeddings

        # Create attention mask
        mask = self._create_attention_mask(input_pos=input_pos)

        # Prepare layer arguments
        layer_kwargs = self._prepare_layer_kwargs(
            input_pos=input_pos,
            mask=mask,
            context=context,
            context_mask=context_mask,
        )

        # Apply transformer layers
        for layer in self.layers:
            if self.params["apply_abs_pos_emb"]:
                h = self.apply_abs_pos_emb(h, input_pos=input_pos)
            h = layer(h, **layer_kwargs)

        # Apply final layer normalization
        h = self.norm(h)

        # Output linear projection
        output = self.output(h)
        return output

    def _create_attention_mask(self, input_pos: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Creates an attention mask for the transformer layers.

        Args:
            input_pos[torch.Tensor]: The position of input sequence (used for inference only).

        Returns:
            Optional[torch.Tensor]: The attention mask, or None for causal mask.
        """

        assert input_pos is not None, "input_pos must be provided for inference"
        mask = self.causal_mask[input_pos]
        return mask

    def _prepare_layer_kwargs(
        self,
        input_pos: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Prepares the keyword arguments for transformer layers.

        Args:
            input_pos (Optional[torch.Tensor]): The position of the current sequence.
            mask (Optional[torch.Tensor]): The attention mask.
            context (Optional[torch.Tensor]): The context tensor added via cross-attn.
            context_mask (Optional[torch.Tensor]): The context cross-attn mask tensor.

        Returns:
            Dict[str, Any]: A dictionary of keyword arguments for the transformer layers.
        """
        if context is not None:
            context = context.to(self.precision)

        if isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask[None, None, :, :]
        if isinstance(context_mask, torch.Tensor) and context_mask.ndim == 2:
            context_mask = context_mask[None, None, :, :]

        layer_kwargs = {
            "mask": mask,
            "context": context,
            "context_mask": context_mask,
        }

        layer_kwargs["input_pos"] = input_pos
        layer_kwargs["rope"] = self.rope

        return layer_kwargs

    def apply_abs_pos_emb(self, x: torch.Tensor, input_pos: int = None) -> torch.Tensor:
        """
        Applies the absolute position embeddings to the input tensor.
        """
        abs_pos_emb = self.abs_pos_emb
        abs_pos_emb = abs_pos_emb[:, input_pos, :] if input_pos is not None else abs_pos_emb
        return x + abs_pos_emb

    @torch.no_grad()
    def expand_vocab(
        self, new_vocab_size: int, init_method: str = "gaussian", multiple_of=64, expand_output_layer=True
    ):
        """
        Expands the vocabulary of the model to the new size.

        Args:
            new_vocab_size (int): The new vocabulary size.
            init_method (str): The initialization method for new embeddings.
                               Can be "zero" or "gaussian". Default is "gaussian".
            multiple_of (int): The new vocabulary size must be a multiple of this value. Defaults to 64 to fully
                leverage the power of NVIDIA TensorCore (source 1: https://x.com/karpathy/status/1621578354024677377,
                source 2: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc)
            expand_output_layer (bool): Whether to also expand the output layer. Defaults to True.

        Returns:
            None
        """
        if new_vocab_size <= self.vocab_size:
            raise ValueError(
                f"New vocabulary size ({new_vocab_size}) must be " f"larger than current size ({self.vocab_size})"
            )
        if new_vocab_size % multiple_of != 0:
            log.debug(f"New vocabulary size must be a multiple of {multiple_of}. Obtained {new_vocab_size}.")
            new_vocab_size = (new_vocab_size // multiple_of + 1) * multiple_of
            log.debug(f"Rounded vocabulary size to {new_vocab_size}.")
        # Resize token embeddings
        old_embeddings = self.tok_embeddings
        tensor_kwargs = {"device": old_embeddings.weight.device, "dtype": old_embeddings.weight.dtype}
        self.tok_embeddings = self._create_token_embeddings(vocab_size=new_vocab_size).to(**tensor_kwargs)
        # Initialize new embeddings
        if init_method not in ["zero", "gaussian"]:
            raise ValueError(f"Unknown initialization method: {init_method}")
        # The default initialization of nn.Embedding is Gaussian, so we don't need to do anything
        # if init_method == "gaussian". Only if init_method == "zero", we need to zero out the new embeddings.
        if init_method == "zero":
            self.tok_embeddings.weight.data[self.vocab_size :].zero_()

        # Copy old embeddings
        log.debug(
            f"old_embeddings: {old_embeddings.weight.data.shape}, new_embeddings: {self.tok_embeddings.weight.data.shape}, vocab_size: {self.vocab_size}"
        )
        self.tok_embeddings.weight.data[: self.vocab_size] = old_embeddings.weight.data
        # Resize output layer
        old_output = self.output
        self.output = self._create_output_projection(vocab_size=new_vocab_size if expand_output_layer else None)

        # Initialize new output weights
        self.output.weight.data[self.vocab_size :].zero_()
        # Copy old output weights
        self.output.weight.data[: self.vocab_size] = old_output.weight.data

        # Update vocab size
        self.vocab_size = new_vocab_size
        log.debug(f"Expanded vocabulary size to {new_vocab_size}")

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
        if strict:
            actual_missing_keys = []
            for key in missing_keys:
                if not any(substring in key for substring in substrings_to_ignore):
                    actual_missing_keys.append(key)
            if len(actual_missing_keys) > 0 or len(unexpected_keys) > 0:
                raise ValueError(f"Missing keys: {actual_missing_keys}\n\nUnexpected keys: {unexpected_keys}")
            missing_keys = actual_missing_keys
        return _IncompatibleKeys(missing_keys, unexpected_keys)
