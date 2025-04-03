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

from typing import Optional

import attrs

from cosmos1.models.autoregressive.configs.base.tokenizer import TokenizerConfig


@attrs.define
class ModelConfig:
    """
    A class to hold model configuration arguments.

    Args:
        dim (int): The dimensionality of the input and output of each transformer block.
        n_layers (int): Number of layers in the transformer.
        n_heads (int): Number of attention heads.
        n_kv_heads (Optional[int]): Number of key-value heads. If None, defaults to n_heads. Note: this is equivalent to
            `num_gqa_groups` in TransformerEngine, where GQA means Grouped Query Attention.
        head_dim (Optional[int]): Dimensionality of each head. If None, defaults to dim // n_heads.
        vocab_size (int): Vocabulary size.
        ffn_hidden_size (int): Hidden size for feedforward network.
        norm_eps (float): Epsilon value for normalization.
        rope_theta (float): Theta value for rotary positional embeddings.
        apply_abs_pos_emb (bool): Whether to apply absolute position embeddings.
        max_batch_size (int): Maximum batch size for inference.
        max_seq_len (int): Maximum sequence length for input text.
        fuse_qkv (bool): Whether to fuse QKV in attention. Defaults to True.
        causal_mask (bool): Whether to use causal mask. Defaults to True.
        norm_type (str): Type of normalization layer. Choices: "rmsnorm", "fused_rmsnorm", "layernorm", "np_layernorm".
        precision (str): Data type for the model.
        use_qk_normalization (bool): Whether to enable QK normalization.
        ckpt_dir (str): Checkpoint directory.
        ckpt_path (str): Checkpoint path.
        apply_yarn (Optional[bool]): Whether to apply YaRN (long-context extension).
        yarn_scale (Optional[float]): Scale factor for YaRN.
        yarn_beta_fast (Optional[int]): Beta fast variable for YaRN (i.e., low_freq_factor in Llama 3.1 RoPE scaling code)
        yarn_beta_slow (Optional[int]): Beta slow variable for YaRN (i.e., high_freq_factor in Llama 3.1 RoPE scaling code)
        original_seq_len (Optional[int]): Original sequence length.
        vision_encoder (Optional[str]): Vision encoder name.
        mm_projector (Optional[str]): Multi-modal projector name.
        vision_encoder_in_channels (Optional[int]): Number of channels in the input image for the vision encoder. Default is 3, you can specify to int larger than 3. E.g. if you have 4-channel images with the last channel as the alpha channel, set this to 4.
        rope_dim (Optional[str]): Dimensionality of the RoPE. Choices: "1D", "3D".
        pytorch_rope_version (Optional[str]): Version of the PyTorch RoPE implementation. Choices: "v1", "v2".
        original_latent_shape (Optional[list]): Original shape of the latent tensor needed for rope extension.
        pad_to_multiple_of (Optional[int]): Pad the position embedding to a multiple of this value.
        vision_encoder_in_channels (Optional[int]): Number of channels in the input image for the vision encoder. Default is 3.
        insert_cross_attn (bool): Whether to insert the cross-attention layers after each multi-head self-attention (MSA) layer.
        insert_cross_attn_every_k_layers (int): Insert cross-attention layers every k TransformerLayers.
        context_dim (Optional[int]): The dimensionality of cross-attention embedding, e.g., T5 embed feature dim.
        num_video_frames (Optional[int]): Number of video frames.
        video_height (Optional[int]): Raw video pixel height dimension.
        video_width (Optional[int]): Raw video pixel width dimension.
        video_latent_shape (Optional[list]): Video tokenizer output dimension, in (T,H,W).
    """

    dim: int = attrs.field(default=4096)
    n_layers: int = attrs.field(default=32)
    n_heads: int = attrs.field(default=32)
    n_kv_heads: Optional[int] = attrs.field(default=8)
    head_dim: Optional[int] = attrs.field(default=None)
    vocab_size: int = attrs.field(default=128256)
    ffn_hidden_size: int = attrs.field(default=14336)
    norm_eps: float = attrs.field(default=1e-5)
    rope_theta: float = attrs.field(default=500000)
    apply_abs_pos_emb: bool = attrs.field(default=False)
    max_batch_size: int = attrs.field(default=1)
    max_seq_len: int = attrs.field(default=8192)
    fuse_qkv: bool = attrs.field(default=False)
    causal_mask: bool = attrs.field(default=True)
    norm_type: str = attrs.field(default="rmsnorm")
    precision: str = attrs.field(default="bfloat16")
    use_qk_normalization: bool = False
    tokenizer: Optional[TokenizerConfig] = None
    ckpt_dir: Optional[str] = attrs.field(default=None)
    ckpt_path: Optional[str] = attrs.field(
        default=None
    )  # If not None, load the model from this path instead of ckpt_dir
    apply_yarn: Optional[bool] = attrs.field(default=False)
    yarn_scale: Optional[float] = attrs.field(default=None)
    yarn_beta_fast: Optional[int] = attrs.field(default=None)
    yarn_beta_slow: Optional[int] = attrs.field(default=None)
    original_seq_len: Optional[int] = attrs.field(default=None)
    vision_encoder: Optional[str] = attrs.field(default=None)
    vision_encoder_in_channels: Optional[int] = attrs.field(default=3)
    mm_projector: Optional[str] = attrs.field(default=None)
    rope_dim: Optional[str] = attrs.field(default="1D")
    pytorch_rope_version: Optional[str] = attrs.field(default="v2")
    original_latent_shape: Optional[list] = None
    pad_to_multiple_of: Optional[int] = None
    vision_encoder_in_channels: Optional[int] = attrs.field(default=3)
    insert_cross_attn: bool = False
    insert_cross_attn_every_k_layers: int = 1
    context_dim: Optional[int] = attrs.field(default=1024)
    # For video training
    num_video_frames: Optional[int] = None
    # Raw video pixel dimension
    video_height: Optional[int] = None
    video_width: Optional[int] = None
    # Video tokenizer output dimension, in (T,H,W), it's computed by num_video_frames/temporal_compress_factor, video_height/spatial_compression_fact, video_width/spatial_compression_fact
    video_latent_shape: Optional[list] = None

    def __getitem__(self, item):
        return getattr(self, item)
