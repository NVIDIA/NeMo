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

import copy
from typing import Callable, List, Optional

from cosmos1.models.autoregressive.configs.base.model import ModelConfig
from cosmos1.models.autoregressive.configs.base.tokenizer import (
    TextTokenizerConfig,
    TokenizerConfig,
    VideoTokenizerConfig,
    create_discrete_video_fsq_tokenizer_state_dict_config,
)
from cosmos1.models.autoregressive.tokenizer.image_text_tokenizer import ImageTextTokenizer
from cosmos1.models.autoregressive.tokenizer.text_tokenizer import TextTokenizer
from cosmos1.utils import log
from cosmos1.utils.lazy_config import LazyCall as L

# Common architecture specifications
BASE_CONFIG = {"n_kv_heads": 8, "norm_type": "rmsnorm", "norm_eps": 1e-5, "ffn_hidden_size": 14336}
COSMOS_ARCHITECTURES = {
    "4b": {
        "n_layers": 16,
        "dim": 4096,
        "n_heads": 32,
    },
    "12b": {
        "n_layers": 40,
        "dim": 5120,
        "n_heads": 32,
        "head_dim": 128,
    },
}

COSMOS_YARN_CONFIG = {
    "original_latent_shape": [3, 40, 64],
    "apply_yarn": True,
    "yarn_beta_fast": 4,
    "yarn_beta_slow": 1,
    "yarn_scale": 2,
}

# Llama3 architecture specifications for different model sizes
LLAMA3_ARCHITECTURES = {
    "8b": {
        "n_layers": 32,
        "dim": 4096,
        "n_heads": 32,
        "ffn_hidden_size": 14336,
    },
}
# Llama3.1 uses YaRN for long context support (context of 128k tokens)
LLAMA_YARN_CONFIG = {
    "apply_yarn": True,
    "yarn_scale": 8,
    "yarn_beta_fast": 4,
    "yarn_beta_slow": 1,
}

# Mistral architecture specifications for different model sizes
MISTRAL_ARCHITECTURES = {
    "12b": {
        "n_layers": 40,
        "dim": 5120,
        "n_heads": 32,
        "ffn_hidden_size": 14336,
        "head_dim": 128,
    },
}

PIXTRAL_VISION_ARCHITECTURES = {
    "12b": {"vision_encoder": "pixtral-12b-vit", "mm_projector": "mlp"},
}


def get_model_arch_specs(model_size: str, model_family: str = "mistral", pretrained: bool = False) -> dict:
    """
    Get the model architecture specifications for the given model size, model family and pretrained status.

    Args:
        model_size (str): Model size. Choices: "1b", "3b", "4b", "7b", etc.
        model_family (str): Model family. Choices: "llama", "llama3", "llama3.1", "mistral"
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        dict: A dictionary containing the model architecture specifications.
    """
    arch_specs = copy.deepcopy(BASE_CONFIG)
    model_size = model_size.lower()
    if model_family.startswith("cosmos"):
        arch_specs.update(COSMOS_ARCHITECTURES[model_size])
    elif model_family.startswith("llama"):
        arch_specs.update(LLAMA3_ARCHITECTURES[model_size])
    elif model_family in ["mistral", "pixtral"]:
        arch_specs.update(MISTRAL_ARCHITECTURES[model_size])
        if model_family == "pixtral":
            arch_specs.update(PIXTRAL_VISION_ARCHITECTURES[model_size])
    else:
        raise ValueError(f"Model family {model_family} is not supported.")

    if pretrained:
        if model_family == "cosmos":
            if model_size == "12b":
                arch_specs.update(COSMOS_YARN_CONFIG)
                log.debug(f"Using YaRN for RoPE extension with config: {COSMOS_YARN_CONFIG}")
            else:
                pass
        elif model_family in ["llama", "llama3"]:
            pretrained_specs = {
                "rope_theta": 500000,
                "max_seq_len": 8192,
                "vocab_size": 128256,
            }
            arch_specs.update(pretrained_specs)
        elif model_family == "llama3.1":
            pretrained_specs = {
                "rope_theta": 500000,
                "max_seq_len": 131072,
                "original_seq_len": 8192,
                "vocab_size": 128256,
                **LLAMA_YARN_CONFIG,
            }
            arch_specs.update(pretrained_specs)
        elif model_family == "mistral":
            assert model_size == "12b", "We only support Mistral-Nemo-12B model."
            pretrained_specs = {
                "rope_theta": 1000000,
                "max_seq_len": 128000,
                "vocab_size": 131072,
            }
            arch_specs.update(pretrained_specs)
        elif model_family == "pixtral":
            assert model_size == "12b", "We only support Pixtral 12B model."
            pretrained_specs = {"rope_theta": 1000000000, "max_seq_len": 128000, "vocab_size": 131072}
            arch_specs.update(pretrained_specs)
        else:
            raise ValueError(f"Model family {model_family} doesn't have a pretrained config.")

    return arch_specs


def create_text_model_config(
    model_ckpt_path: str,
    tokenizer_path: str,
    model_family: str = "mistral",
    model_size: str = "12b",
    is_instruct_model: bool = True,
    max_seq_len: int = None,
    max_batch_size: int = 1,
    rope_dim: str = "1D",
    add_special_tokens: bool = True,
    pytorch_rope_version: str = None,
) -> dict:
    """Create a text model for training or inference.
    Args:
        model_ckpt_path (str): Path to the model checkpoint.
        tokenizer_path (str): Path to the tokenizer folder.
        model_family (str): Model family. Choices: "llama", "llama3", "llama3.1", "mistral".
        model_size (str): Model size. Choices: "1b", "3b", "4b", "7b", "8b", "72b", etc.
        is_instruct_model (bool): Whether the model is an instruct model.
        inference (bool): Whether to create the model for inference.
        max_seq_len (int): Maximum sequence length.
        max_batch_size (int): Maximum batch size.
        rope_dim (str): RoPE dimension. Choices: "1D", "3D".
        add_special_tokens (bool): Whether to add special tokens.
    Returns:
        dict: A dictionary containing the model configuration, which can be used to instantiate the model object.
    """
    # Model size specific parameters
    model_arch_specs = get_model_arch_specs(model_family=model_family, model_size=model_size, pretrained=True)
    if max_seq_len is not None:
        # Override the max_seq_len if provided
        model_arch_specs["max_seq_len"] = max_seq_len
    if pytorch_rope_version is not None:
        model_arch_specs["pytorch_rope_version"] = pytorch_rope_version
    model_config = ModelConfig(
        max_batch_size=max_batch_size,
        precision="bfloat16",
        ckpt_path=model_ckpt_path,
        use_qk_normalization=False,
        rope_dim=rope_dim,
        **model_arch_specs,
    )

    tokenizer_config = TokenizerConfig(
        text_tokenizer=TextTokenizerConfig(
            config=L(TextTokenizer)(
                model_family=model_family,
                is_instruct_model=is_instruct_model,
                local_path=tokenizer_path,
            ),
            data_key="text",
            tokenizer_offset=model_config.vocab_size,
            tokenize_here=False,
            vocab_size=model_config.vocab_size,
        ),
        seq_len=model_config.max_seq_len,
        training_type="text_only",
        add_special_tokens=add_special_tokens,
    )
    return model_config, tokenizer_config


def create_vision_language_model_config(
    model_ckpt_path: str,
    tokenizer_ckpt_path: str,
    model_family: str = "pixtral",
    model_size: str = "12b",
    is_instruct_model: bool = True,
    max_batch_size: int = 1,
    rope_dim: str = "1D",
    add_special_tokens: bool = True,
    max_seq_len: int = None,
    vision_encoder_in_channels: int = 3,
    fuse_qkv: bool = False,
    pytorch_rope_version: str = None,
) -> dict:
    """Create a vision-language model for training or inference.
    Args:
        model_ckpt_path (str): Path to the model checkpoint.
        tokenizer_ckpt_path (str): Path to the tokenizer checkpoint.
        model_family (str): Model family. Choices: "pixtral".
        model_size (str): Model size. Choices: "12b".
        is_instruct_model (bool): Whether the model is an instruct model.
        rope_dim (str): RoPE dimension. Choices: "1D".
        add_special_tokens (bool): Whether to add special tokens.
        max_seq_len (int): Maximum sequence length.
        vision_encoder_in_channels (int): Number of channels in the input image for the vision encoder. Default is 3, you can specify to int larger than 3. E.g. if you have 4 channel images where last channel is binary mask, set this to 4.
        fuse_qkv (bool): Whether to fuse the QKV linear layers.
    Returns:
        dict: A dictionary containing the model configuration, which can be used to instantiate the model object.
    """
    # Model size specific parameters
    model_arch_specs = get_model_arch_specs(model_family=model_family, model_size=model_size, pretrained=True)
    if max_seq_len is not None:
        # Override the max_seq_len if provided
        model_arch_specs["max_seq_len"] = max_seq_len
    if pytorch_rope_version is not None:
        model_arch_specs["pytorch_rope_version"] = pytorch_rope_version

    model_config = ModelConfig(
        max_batch_size=max_batch_size,
        precision="bfloat16",
        ckpt_path=model_ckpt_path,
        use_qk_normalization=False,
        rope_dim=rope_dim,
        vision_encoder_in_channels=vision_encoder_in_channels,
        fuse_qkv=fuse_qkv,
        **model_arch_specs,
    )
    # Vision-language tokenizer
    tokenizer_config = TokenizerConfig(
        text_tokenizer=TextTokenizerConfig(
            config=L(ImageTextTokenizer)(
                model_family=model_family,
                is_instruct_model=is_instruct_model,
                image_processor_path=tokenizer_ckpt_path,
                tokenizer_path=tokenizer_ckpt_path,
            ),
            data_key="image_text_interleaved",
            tokenizer_offset=model_config.vocab_size,
            tokenize_here=False,
            vocab_size=model_config.vocab_size,
        ),
        seq_len=model_config.max_seq_len,
        training_type="image_text_interleaved",
        add_special_tokens=add_special_tokens,
    )
    return model_config, tokenizer_config


def create_video2world_model_config(
    model_ckpt_path: str,
    tokenizer_ckpt_path: str,
    model_family: str = "cosmos",
    model_size: str = "4b",
    pixel_chunk_duration: int = 9,
    num_video_frames: int = 36,
    compression_ratio: List[int] = [8, 16, 16],
    original_seq_len: int = 8192,
    num_condition_latents_t: int = 1,
    num_tokens_to_ignore: int = -1,
    batch_size: int = 2,
    video_tokenizer_config_creator: Callable = create_discrete_video_fsq_tokenizer_state_dict_config,
    rope_dim: str = "3D",
    add_special_tokens: bool = True,
    video_height: int = 384,
    video_width: int = 640,
    use_qk_normalization: bool = True,
    insert_cross_attn: bool = False,
    insert_cross_attn_every_k_layers: int = 1,
    context_dim: int = 1024,
    training_type: str = "video_to_video",
    pad_to_multiple_of: Optional[int] = 64,
    vocab_size: int = 64000,
    apply_abs_pos_emb: bool = False,
) -> dict:
    """Create a video-to-world model config.
    Args:
        model_family (str): Model family. Choices: "llama", "llama3", "llama3.1", "mistral".
        model_size (str): Model size. Choices: "1b", "8b", "3b".
        pixel_chunk_duration (int): Number of frames in each chunk.
        num_video_frames (int): Number of video frames.
        compression_ratio (List[int]): Compression ratio for the video frames. Choices: [8, 16, 16] or [4, 8, 8].
        original_seq_len (int): Original sequence length.
        apply_yarn (bool): Whether to apply YaRN for long context scaling.
        yarn_beta_fast (Optional[int]): Fast beta for YaRN.
        yarn_beta_slow (Optional[int]): Slow beta for YaRN.
        yarn_scale (Optional[int]): Scale factor for ctx extension.
        use_qk_normalization (bool): Whether to use Query-Key normalization.
        training_type (str): Type of training task.
        batch_size (int): Batch size.
        video_tokenizer_config_creator (Callable): Method that takes "pixel_chunk_duration: int" and "version: str" as arguments and returns video tokenizer config
        video_tokenizer_version (str): Version of the video tokenizer.
        num_condition_latents_t (int): Number of conditioning latent channels
        num_tokens_to_ignore (int) = Number of tokens to ignore. This takes the precedence
        video_height (int): Height of the video frame. Defaults to 384.
        video_width (int): Width of the video frame. Defaults to 640.
        rope_dim (str): RoPE dimension. Choices: "1D", "3D".
        add_special_tokens (bool): Whether to add special tokens, use False for 2D/3D RoPE.
        pad_to_multiple_of (int): Pad the token sequence length to the nearest multiple of this number. Defaults to 64.
        vocab_size (int): Vocabulary size.
        apply_abs_pos_emb (bool): Whether to apply absolute positional embeddings.
    Returns:
        dict: A dictionary containing the model configuration representing the model object, can be instantiated.
    """
    assert (
        pixel_chunk_duration % compression_ratio[0] == 1
    ), f"pixel_chunk_duration({pixel_chunk_duration}) should be k*n + 1 (k={compression_ratio[0]})"
    latent_chunk_duration = (pixel_chunk_duration - 1) // compression_ratio[0] + 1
    latent_height = video_height // compression_ratio[1]
    latent_width = video_width // compression_ratio[2]
    # Do some math to compute the video latent shape and sequence length
    assert (
        num_video_frames % pixel_chunk_duration == 0
    ), f"num_video_frames {num_video_frames} should be divisible by pixel_chunk_duration {pixel_chunk_duration}"
    video_latent_shape = [
        num_video_frames // pixel_chunk_duration * latent_chunk_duration,
        latent_height,
        latent_width,
    ]
    # product of video_latent_shape
    num_token_video_latent = video_latent_shape[0] * video_latent_shape[1] * video_latent_shape[2]
    if add_special_tokens:
        seq_len = num_token_video_latent + 3  # Sequence length per batch, max_seq_len + 3
        seq_len = (seq_len + 63) // 64 * 64  # Round up to multiple of 64
    # for text to video, we need to add <bov> token to indicate the start of the video
    elif training_type == "text_to_video":
        seq_len = num_token_video_latent + 1
    else:
        seq_len = num_token_video_latent

    if seq_len % pad_to_multiple_of != 0:
        # Round up to the nearest multiple of pad_to_multiple_of
        seq_len = ((seq_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    # Model size specific parameters
    model_arch_specs = get_model_arch_specs(model_family=model_family, model_size=model_size, pretrained=True)

    # Whether skip the loss for first chunk or not, note the first token is already skipped when computing the loss
    # If num_tokens_to_ignore is specified, use it.
    # Else compute it from num_condition_latents_t
    if num_tokens_to_ignore < 0:
        num_tokens_to_ignore = latent_height * latent_width * num_condition_latents_t
        if not add_special_tokens and num_condition_latents_t > 0:
            # If there are no special tokens (bov), do a -1 so that you can compute the loss
            # from the first token of the next chunk
            num_tokens_to_ignore -= 1

    model_config = ModelConfig(
        video_height=video_height,
        video_width=video_width,
        max_seq_len=seq_len,
        max_batch_size=batch_size,
        precision="bfloat16",
        ckpt_path=model_ckpt_path,
        use_qk_normalization=use_qk_normalization,
        vocab_size=64000,
        original_seq_len=original_seq_len,
        video_latent_shape=video_latent_shape,
        num_video_frames=num_video_frames,
        rope_dim=rope_dim,
        pad_to_multiple_of=pad_to_multiple_of,
        insert_cross_attn=insert_cross_attn,
        insert_cross_attn_every_k_layers=insert_cross_attn_every_k_layers,
        context_dim=context_dim,
        apply_abs_pos_emb=apply_abs_pos_emb,
        **model_arch_specs,
    )

    video_tokenizer_config = video_tokenizer_config_creator(
        tokenizer_ckpt_path, pixel_chunk_duration, compression_ratio
    )
    tokenizer_config = TokenizerConfig(
        text_tokenizer=None,
        video_tokenizer=VideoTokenizerConfig(
            config=video_tokenizer_config,
            data_key="video",
            tokenizer_offset=0,  # Since there is no text embeddings in the model. Note this only apply when the model is trained from scratch. If we use text pretrained model, the offset will be vocab_size of text token.
            tokenize_here=True,
            max_seq_len=num_token_video_latent,
            vocab_size=vocab_size,
        ),
        seq_len=seq_len,
        training_type=training_type,
        add_special_tokens=add_special_tokens,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    return model_config, tokenizer_config
