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

from cosmos1.models.autoregressive.tokenizer.discrete_video import DiscreteVideoFSQStateDictTokenizer
from cosmos1.models.autoregressive.tokenizer.networks import CausalDiscreteVideoTokenizer
from cosmos1.utils.lazy_config import LazyCall as L
from cosmos1.utils.lazy_config import LazyDict


def create_discrete_video_fsq_tokenizer_state_dict_config(
    ckpt_path, pixel_chunk_duration=33, compression_ratio=[8, 16, 16]
) -> LazyDict:
    CausalDiscreteFactorizedVideoTokenizerConfig: LazyDict = L(CausalDiscreteVideoTokenizer)(
        # The new causal discrete tokenizer, that is at least 2x more efficient in memory and runtime.
        # - It relies on fully 3D discrete wavelet transform
        # - Uses a layer norm instead of a group norm
        # - Factorizes full convolutions into spatial and temporal convolutions
        # - Factorizes full attention into spatial and temporal attention
        # - Strictly causal, with flexible temporal length at inference.
        attn_resolutions=[32],
        channels=128,
        channels_mult=[2, 4, 4],
        dropout=0.0,
        in_channels=3,
        num_res_blocks=2,
        out_channels=3,
        resolution=1024,
        patch_size=4,
        patch_method="haar",
        z_channels=16,
        z_factor=1,
        num_groups=1,
        legacy_mode=False,
        spatial_compression=16,
        temporal_compression=8,
        embedding_dim=6,
        levels=[8, 8, 8, 5, 5, 5],
        name="CausalDiscreteFactorizedVideoTokenizer",
    )

    return L(DiscreteVideoFSQStateDictTokenizer)(
        enc_fp=ckpt_path.replace("ema.jit", "encoder.jit"),
        dec_fp=ckpt_path.replace("ema.jit", "decoder.jit"),
        tokenizer_module=CausalDiscreteFactorizedVideoTokenizerConfig,
        name="discrete_video_fsq",
        latent_ch=6,
        is_bf16=True,
        pixel_chunk_duration=pixel_chunk_duration,
        latent_chunk_duration=1 + (pixel_chunk_duration - 1) // compression_ratio[0],
        max_enc_batch_size=8,
        max_dec_batch_size=4,
        levels=[8, 8, 8, 5, 5, 5],
        compression_ratio=compression_ratio,
    )


@attrs.define(slots=False)
class TextTokenizerConfig:
    """
    Text tokenizer config

    Args:
        config: Config file to define the text tokenizer class.
        data_key (str): The input key from data_dict that will be passed to the text tokenizer.
        tokenize_here (bool): Whether to use the tokenizer to perform online tokenization.
        tokenizer_offset (int): Offset that is added to the tokens.
        vocab_size (int): Vocabulary size of the tokenizer.
    """

    config: LazyDict
    data_key: str = ""
    tokenize_here: bool = False
    tokenizer_offset: int = 0
    vocab_size: int = 0


@attrs.define(slots=False)
class VideoTokenizerConfig:
    """
    Video tokenizer config

    Args:
        config: Config file to define the video tokenizer class.
        data_key (str): The input key from data_dict that will be passed to the video tokenizer.
        tokenize_here (bool): Whether to use the tokenizer to perform online tokenization.
        tokenizer_offset (int): Offset that is added to the tokens. In case of joint text-video tokenizers, we
            add an offset to make sure that video tokens and text tokens don't overlap.
        vocab_size (int): Vocabulary size of the tokenizer.
        max_seq_len (int): Maximum token length for an input video.
    """

    config: LazyDict
    data_key: str = ""
    tokenize_here: bool = True
    tokenizer_offset: int = 0
    vocab_size: int = 0
    max_seq_len: int = -1


@attrs.define(slots=False)
class TokenizerConfig:
    """
    Joint tokenizer config

    Args:
        text_tokenizer (TextTokenizerConfig): Text tokenizer config file
        class_tokenizer (ClassTokenizerConfig): Class tokenizer config file
        video_tokenizer (VideoTokenizerConfig): Video tokenizer config file
        image_tokenizer (ImageTokenizerConfig): Image tokenizer config file
        seq_len (int): Final token sequence length
        training_type (str): Type of training we use. Supports ["text_only", "text_to_video", "class_to_image", "image_text_interleaved"]
        add_special_tokens (bool): Whether to add special tokens to the output tokens
        pad_to_multiple_of (int): Pad the token sequence length to the nearest multiple of this number. Defaults to 64.
    """

    text_tokenizer: Optional[TextTokenizerConfig] = None
    video_tokenizer: Optional[VideoTokenizerConfig] = None
    seq_len: int = 4096
    training_type: str = None
    add_special_tokens: bool = True
    pad_to_multiple_of: Optional[int] = 64
