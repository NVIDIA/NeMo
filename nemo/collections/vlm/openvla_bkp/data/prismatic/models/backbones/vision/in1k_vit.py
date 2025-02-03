"""
in1k_vit.py

Vision Transformers trained / finetuned on ImageNet (ImageNet-21K =>> ImageNet-1K)
"""

from nemo.collections.vlm.openvla_bkp.data.prismatic.models.backbones.vision.base_vision import TimmViTBackbone

# Registry =>> Supported Vision Backbones (from TIMM)
IN1K_VISION_BACKBONES = {
    "in1k-vit-l": "vit_large_patch16_224.augreg_in21k_ft_in1k",
}


class IN1KViTBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(
            vision_backbone_id,
            IN1K_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
        )
