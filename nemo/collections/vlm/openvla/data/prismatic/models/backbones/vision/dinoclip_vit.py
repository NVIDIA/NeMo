"""
dinoclip_vit.py

Vision backbone that returns concatenated features from both DINOv2 and CLIP.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Tuple

import timm
import torch
from PIL import Image
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize

from nemo.collections.vlm.openvla.data.prismatic.models.backbones.vision.base_vision import (
    ImageTransform,
    LetterboxPad,
    VisionBackbone,
    unpack_tuple,
)

# Registry =>> Supported DinoCLIP Pairs (as TIMM identifiers)
DINOCLIP_VISION_BACKBONES = {
    "dinoclip-vit-l-336px": {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "clip": "vit_large_patch14_clip_336.openai",
    },
}


@dataclass
class DinoCLIPImageTransform:
    dino_image_transform: ImageTransform
    clip_image_transform: ImageTransform
    is_prismatic: bool = True

    def __call__(self, img: Image, **kwargs: str) -> Dict[str, torch.Tensor]:
        return {"dino": self.dino_image_transform(img, **kwargs), "clip": self.clip_image_transform(img, **kwargs)}


class DinoCLIPViTBackbone(VisionBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        self.dino_timm_path_or_url = DINOCLIP_VISION_BACKBONES[vision_backbone_id]["dino"]
        self.clip_timm_path_or_url = DINOCLIP_VISION_BACKBONES[vision_backbone_id]["clip"]

        # Initialize both Featurizers (ViTs) by downloading from HF / TIMM Hub if necessary
        self.dino_featurizer: VisionTransformer = timm.create_model(
            self.dino_timm_path_or_url, pretrained=True, num_classes=0, img_size=self.default_image_size
        )
        self.dino_featurizer.eval()

        self.clip_featurizer: VisionTransformer = timm.create_model(
            self.clip_timm_path_or_url, pretrained=True, num_classes=0, img_size=self.default_image_size
        )
        self.clip_featurizer.eval()

        # Monkey-Patch the `forward()` function of the featurizers to ensure FSDP-compatibility
        #   => Note: By default set `get_intermediate_layers` to return the *SECOND-TO-LAST* layer patches!
        #   => TODO (siddk) Remove after resolution of https://github.com/pytorch/pytorch/issues/109385
        self.dino_featurizer.forward = unpack_tuple(
            partial(self.dino_featurizer.get_intermediate_layers, n={len(self.dino_featurizer.blocks) - 2})
        )
        self.clip_featurizer.forward = unpack_tuple(
            partial(self.clip_featurizer.get_intermediate_layers, n={len(self.clip_featurizer.blocks) - 2})
        )

        # Get Configs for _both_ Featurizers =>> Note :: Override default image size for larger resolution models
        self.dino_data_cfg = timm.data.resolve_model_data_config(self.dino_featurizer)
        self.dino_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        self.clip_data_cfg = timm.data.resolve_model_data_config(self.clip_featurizer)
        self.clip_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # Initialize *both* Transforms
        default_dino_transform = timm.data.create_transform(**self.dino_data_cfg, is_training=False)
        default_clip_transform = timm.data.create_transform(**self.clip_data_cfg, is_training=False)
        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_dino_transform, Compose), "Unexpected `default_dino_image_transform`!"
            assert isinstance(default_clip_transform, Compose), "Unexpected `default_clip_image_transform`!"
            assert isinstance(default_dino_transform.transforms[0], Resize)
            assert isinstance(default_clip_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            dino_transform = Compose(
                [
                    Resize(target_size, interpolation=default_dino_transform.transforms[0].interpolation),
                    *default_dino_transform.transforms[1:],
                ]
            )
            clip_transform = Compose(
                [
                    Resize(target_size, interpolation=default_clip_transform.transforms[0].interpolation),
                    *default_clip_transform.transforms[1:],
                ]
            )

            self.image_transform = DinoCLIPImageTransform(dino_transform, clip_transform)

        elif self.image_resize_strategy == "resize-crop":
            self.image_transform = DinoCLIPImageTransform(default_dino_transform, default_clip_transform)

        elif self.image_resize_strategy == "letterbox":
            assert isinstance(default_dino_transform, Compose), "Unexpected `default_dino_transform`!"
            assert isinstance(default_clip_transform, Compose), "Unexpected `default_clip_transform`!"
            assert "mean" in self.dino_data_cfg and "mean" in self.clip_data_cfg, "DinoCLIP `data_cfg` missing `mean`!"

            # Compute Padding Fill Value(s) (rescaled normalization mean if applicable)
            dino_fill = tuple([int(x * 255) for x in self.dino_data_cfg["mean"]])
            clip_fill = tuple([int(x * 255) for x in self.clip_data_cfg["mean"]])

            # Build New Transform
            self.image_transform = DinoCLIPImageTransform(
                Compose([LetterboxPad(dino_fill), *default_dino_transform.transforms]),
                Compose([LetterboxPad(clip_fill), *default_clip_transform.transforms]),
            )

        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then both of the _entire_ featurizers."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Runs the transformed image/pixel tensors through each vision backbone, returning concatenated patches."""
        dino_patches = self.dino_featurizer(pixel_values["dino"])
        clip_patches = self.clip_featurizer(pixel_values["clip"])

        return torch.cat([dino_patches, clip_patches], dim=2)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.dino_data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.dino_featurizer.embed_dim + self.clip_featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        assert self.dino_featurizer.patch_embed.num_patches == self.clip_featurizer.patch_embed.num_patches
        return self.dino_featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
