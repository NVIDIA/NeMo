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

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.models.vision.clip_vit_model import CLIPViTModel as MCoreCLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector as MCoreMultimodalProjector
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.collections.llm import Llama2Config7B
from nemo.collections.llm.gpt.model.base import get_batch_on_this_context_parallel_rank
from nemo.collections.multimodal.mimo.model.base import BaseMimoModel
from nemo.collections.multimodal.mimo.model.layer_spec import get_image_output_projection_layer_spec
from nemo.collections.multimodal.mimo.model.projection import ImageOutputProjectionModule
from nemo.lightning import get_vocab_size, io
from nemo.lightning.io.pl import ckpt_to_weights_subdir


def mimo_forward_step(model, batch) -> torch.Tensor:
    forward_args = {
        "images": batch["images"],
        "output_images": batch["output_images"],
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
        "loss_mask": batch.get("loss_mask", None),
        "labels": batch.get("labels", None),
        "input_text": batch.get("input_text", None),
        "num_image_tiles": batch.get("num_image_tiles", None),
        "image_token_mask": batch.get("image_token_mask", None),
    }

    output_dict = model(**forward_args)
    return output_dict


def mimo_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
    from megatron.core import parallel_state

    batch = next(dataloader_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_keys = set()
    required_keys.update(
        (
            "attention_mask",
            "images",
            "tokens",
            "num_image_tiles",
            "output_images",
            "input_text",
            "image_token_mask",
        )
    )
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(("position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask"))

    _batch = {
        key: (
            (val.cuda(non_blocking=True) if hasattr(val, "cuda") else val)
            if key in required_keys and val is not None
            else None
        )
        for key, val in _batch.items()
    }
    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch)

    return output


@dataclass
class ImageInputProjectionConfig(TransformerConfig, io.IOMixin):
    projector_type: str = "mlp"
    input_size: Optional[int] = 1024
    hidden_size: int = 4096
    ffn_hidden_size: int = 4096
    activation_func: Callable = F.gelu
    bias: bool = True
    bias_activation_fusion: bool = True
    add_bias_linear: bool = True
    layer_spec: ModuleSpec = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    ).submodules
    num_layers: int = 1  # placeholder, NOT used!
    num_attention_heads: int = 8  # placeholder, NOT used!'

    def configure_model(self) -> "MCoreMultimodalProjector":
        return MCoreMultimodalProjector(
            self,
            self.layer_spec,
            self.projector_type,
            input_size=self.input_size,  # input size to the projection.
        )


@dataclass
class ImageOutputProjectionConfig(TransformerConfig, io.IOMixin):
    num_layers: int = 4
    num_attention_heads: int = 16
    num_query_token: int = 77
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    hidden_size: int = 4096
    hidden_dropout: int = 0.0
    attention_dropout: float = 0.0
    ffn_hidden_size: int = 4096
    gated_linear_unit: bool = False
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False  # TODO: Yash Check this
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization = 'LayerNorm'

    def configure_model(self):

        enc_layer_spec, dec_layer_spec, output_linear_projection = get_image_output_projection_layer_spec()

        output_projection = ImageOutputProjectionModule(
            config=self,
            encoder_config=self,
            transformer_encoder_layer_spec=enc_layer_spec,
            transformer_decoder_layer_spec=dec_layer_spec,
            output_linear_projection=output_linear_projection,
        )
        return output_projection


@dataclass
class ImageEncoderTransformerConfig(TransformerConfig, io.IOMixin):

    from nemo.collections.vlm.layer_specs import get_layer_spec_te

    vision_model_type = 'clip'
    drop_vision_class_token: bool = True
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    hidden_dropout: int = 0.0
    attention_dropout: float = 0.0
    gated_linear_unit: bool = False
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False  # TODO: Yash Check this
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization = 'LayerNorm'
    apply_rope_fusion = False
    layer_spec: ModuleSpec = get_layer_spec_te(is_vit=True)
    img_h: int = 336
    img_w: int = 336
    patch_dim: int = 14

    ln_pre_impl: Union[ModuleSpec, type] = TENorm
    ln_post_impl: Union[ModuleSpec, type] = TENorm

    def __post_init__(self):
        if self.vision_model_type == "siglip":
            self.num_layers = 27
            self.num_attention_heads = 16
            self.hidden_size = 1152
            self.ffn_hidden_size = 4304
            self.activation_func = torch.nn.functional.gelu  # Mcore uses fast_gelu
            self.kv_channels = 72
            self.num_query_groups = 16
            self.qk_layernorm = False
            self.layernorm_epsilon = 1e-6
            self.add_class_token: bool = False
            self.class_token_len: int = 0
        elif self.vision_model_type == "clip":
            self.num_layers: int = 24
            self.num_attention_heads: int = 16
            self.hidden_size: int = 1024
            self.ffn_hidden_size: int = 4096
            self.kv_channels: int = 64
            self.num_query_groups: int = 16
            self.activation_func = torch.nn.functional.gelu  # Mcore uses fast_gelu
            self.add_class_token: bool = True
            self.class_token_len: int = 1
        else:
            raise NotImplementedError(f"Vision model type {self.vision_model_type} not implemented")
        # TODO: Yash add config for internvit

    def configure_model(self) -> "MCoreCLIPViTModel":

        return MCoreCLIPViTModel(
            self,
            self.layer_spec,
            ln_pre_impl=self.ln_pre_impl,
            ln_post_impl=self.ln_post_impl,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
            patch_dim=self.patch_dim,
            img_h=self.img_h,
            img_w=self.img_w,
            model_subtype=self.vision_model_type,
        )


class ImageDecoderTransformerConfig(io.IOMixin):
    image_decoder_name = "stabilityai/stable-diffusion-2"

    def configure_model(self):
        from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

        scheduler = EulerDiscreteScheduler.from_pretrained(self.image_decoder_name, subfolder="scheduler")
        image_decoder = StableDiffusionPipeline.from_pretrained(self.image_decoder_name, scheduler=scheduler)

        image_decoder.vae.requires_grad_(False)
        image_decoder.unet.requires_grad_(False)
        image_decoder.text_encoder.requires_grad_(False)
        return image_decoder


@dataclass
class MimoConfig(TransformerConfig, io.IOMixin):

    language_transformer_config: Optional[TransformerConfig] = field(default_factory=lambda: Llama2Config7B())

    image_encoder_transformer_config: Optional[TransformerConfig] = field(
        default_factory=lambda: ImageEncoderTransformerConfig()
    )
    image_decoder_transformer_config: ImageDecoderTransformerConfig = field(
        default_factory=lambda: ImageDecoderTransformerConfig()
    )

    image_input_projection_config: Optional[TransformerConfig] = field(
        default_factory=lambda: ImageInputProjectionConfig()
    )
    image_output_projection_config: Optional[TransformerConfig] = field(
        default_factory=lambda: ImageOutputProjectionConfig()
    )

    freeze_language_model: bool = False
    freeze_image_encoder: bool = True
    freeze_image_input_projection: bool = False
    freeze_image_output_projection: bool = False

    forward_step_fn: Callable = mimo_forward_step
    data_step_fn: Callable = mimo_data_step

    vocab_size: Optional[int] = None
    image_special_tokens: Optional[List[str]] = None
    image_special_token_indices: Optional[List[int]] = None
    make_vocab_size_divisible_by: int = 128

    load_vision_mlp_language_model_path: Optional[str] = None
    language_model_path: Optional[str] = None
    vision_model_path: Optional[str] = None
    mlp_projector_path: Optional[str] = None

    # These below variables are placeholders, needed for TransformerConfig assertions. not used anywhere
    seq_length: int = 2048
    max_position_embeddings: int = 2048
    num_layers: int = 1
    hidden_size: int = 2048
    num_attention_heads: int = 8
    ffn_hidden_size: int = 8
    # TODO:Yash make this enum
    stage: str = "encoder_alignment"  # [encoder_alignment, decoder_alignment, 'interleaved_pretrain']
    # denoising loss for stable diffusion decoder, only used for decoder_alignment stages
    generation_loss: bool = False

    def configure_model(self, tokenizer) -> "BaseMimoModel":

        original_vocab_size = tokenizer.vocab_size - len(self.image_special_token_indices)
        self.vocab_size = get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by)

        logging.info(f"padded vocab size to {self.vocab_size}")
        model = BaseMimoModel(config=self)

        if self.language_model_path:
            sharded_state_dict = dict(state_dict=model.language_model.sharded_state_dict(prefix="module."))
            strict = StrictHandling.LOG_UNEXPECTED
            loaded_state_dict = dist_checkpointing.load(
                sharded_state_dict=sharded_state_dict,
                checkpoint_dir=ckpt_to_weights_subdir(self.language_model_path, is_saving=False),
                strict=strict,
            )
            loaded_state_dict = {k.removeprefix("module."): v for k, v in loaded_state_dict["state_dict"].items()}

            model.language_model.load_state_dict(loaded_state_dict)
            logging.info(f"Loaded language model from {self.language_model_path}")
            print(f"Loaded language model from {self.language_model_path}")

        if self.vision_model_path:
            sharded_state_dict = dict(state_dict=model.vision_model.sharded_state_dict(prefix="module."))
            strict = StrictHandling.LOG_UNEXPECTED

            loaded_state_dict = dist_checkpointing.load(
                sharded_state_dict=sharded_state_dict,
                checkpoint_dir=ckpt_to_weights_subdir(self.vision_model_path, is_saving=False),
                strict=strict,
            )
            loaded_state_dict = {k.removeprefix("module."): v for k, v in loaded_state_dict["state_dict"].items()}

            model.vision_model.load_state_dict(loaded_state_dict, strict=False)
            logging.info(f"Loaded vision model from {self.vision_model_path}")
            print(f"Loaded vision model from {self.vision_model_path}")
        # initializing the special token embedings to be the average of the original embeddings
        # TODO:Yash have to handle TP below. Have to properly gather across TP ranks
        # average_embedding = model.language_model.embedding.word_embeddings.weight.data[:original_vocab_size].mean(
        #     dim=0, keepdim=True
        # )
        # model.language_model.embedding.word_embeddings.weight.data[
        #     original_vocab_size : original_vocab_size + len(self.image_special_token_indices)
        # ] = average_embedding.repeat(len(self.image_special_token_indices), 1)
        model.freeze(
            freeze_language_model=self.freeze_language_model,
            freeze_vision_model=self.freeze_image_encoder,
            freeze_vision_projection=self.freeze_image_input_projection,
        )

        return model
