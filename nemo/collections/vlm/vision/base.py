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


import os
from dataclasses import dataclass
from typing import Optional, Union

import torch.distributed
from megatron.core.models.vision.clip_vit_model import CLIPViTModel as MCoreCLIPViTModel
from megatron.core.transformer.custom_layers.transformer_engine import (
    TENorm,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import CLIPVisionConfig, CLIPVisionModel

from nemo.lightning import io


def set_input_tensor(self, tensor):
    pass


def get_image_sequence_length(img_h, img_w, patch_dim, add_class_token, class_token_len):
    """Get image sequence length given image size, patch size, and class token."""
    num_patches_per_dim_h = img_h // patch_dim
    num_patches_per_dim_w = img_w // patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    return num_patches + (class_token_len if add_class_token else 0)


@dataclass
class HFCLIPVisionConfig(CLIPVisionConfig, io.IOMixin):
    """
    https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/models/clip/configuration_clip.py#L261
    """

    hidden_size: int = 1024
    add_class_token: bool = False
    class_token_len: int = 1
    num_image_embeddings_per_tile: Optional[int] = None
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None

    def __post_init__(self, *args, **kwargs) -> None:
        CLIPVisionConfig.__init__(self, *args, **kwargs, hidden_size=self.hidden_size)
        if self.pretrained_model_name_or_path is not None:
            config = CLIPVisionConfig.from_pretrained(self.pretrained_model_name_or_path)
            for key, value in config.to_dict().items():
                setattr(self, key, value)
        self.num_image_embeddings_per_tile = get_image_sequence_length(
            img_h=self.image_size,
            img_w=self.image_size,
            patch_dim=self.patch_size,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
        )

    def configure_model(self) -> "CLIPVisionModel":
        # Monkey patch the method to the vision encoder
        CLIPVisionModel.set_input_tensor = set_input_tensor

        if self.pretrained_model_name_or_path is None:
            model = CLIPVisionModel(self)
        else:
            model = CLIPVisionModel.from_pretrained(self.pretrained_model_name_or_path)
        return model


@dataclass
class CLIPViTConfig(TransformerConfig, io.IOMixin):
    ln_pre_impl: Union[ModuleSpec, type] = TENorm
    ln_post_impl: Union[ModuleSpec, type] = TENorm
    add_class_token: bool = True
    class_token_len: int = 1
    patch_dim: int = 14
    img_h: int = 336
    img_w: int = 336
    vision_model_type: str = "clip"  # ["clip", "siglip", "internvit"]
    num_image_embeddings_per_tile: Optional[int] = None
    transformer_layer_spec: ModuleSpec = None

    num_layers: int = 1  # Placeholder, NOT used!
    num_attention_heads: int = 8  # Placeholder, NOT used!

    def __post_init__(self):
        if self.vision_model_type == "siglip":
            self.add_class_token = False
            self.class_token_len = 0
        self.num_image_embeddings_per_tile = get_image_sequence_length(
            img_h=self.img_h,
            img_w=self.img_w,
            patch_dim=self.patch_dim,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
        )

    def configure_model(self) -> "CLIPViTModel":
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            from nemo.collections.vlm.layer_specs import get_layer_spec_te
            transformer_layer_spec = get_layer_spec_te(is_vit=True)
        return CLIPViTModel(
            self,
            transformer_layer_spec,
            ln_pre_impl=self.ln_pre_impl,
            ln_post_impl=self.ln_post_impl,
            add_class_token=self.add_class_token,
            class_token_len=self.class_token_len,
            patch_dim=self.patch_dim,
            img_h=self.img_h,
            img_w=self.img_w,
            model_subtype=self.vision_model_type,
        )


class CLIPViTModel(MCoreCLIPViTModel):
    """CLIP ViT vision model."""

    def forward(
            self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, num_unused_layers: int = 0
    ) -> torch.Tensor:
        if num_unused_layers > 0:
            unused_layers = self.decoder.layers[-num_unused_layers:]
            self.decoder.layers = self.decoder.layers[:-num_unused_layers]
            x = super().forward(x, attention_mask)
            self.decoder.layers.append(unused_layers)
            return x

        return super().forward(x, attention_mask)
