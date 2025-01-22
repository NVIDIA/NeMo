"""
siglip_vit.py
"""

from nemo.collections.vlm.openvla.data.prismatic.models.backbones.vision.base_vision import TimmViTBackbone

# Registry =>> Supported SigLIP Vision Backbones (from TIMM) =>> Note:: Using SigLIP w/ Patch = 14 (but SO400M Arch)
SIGLIP_VISION_BACKBONES = {
    "siglip-vit-b16-224px": "vit_base_patch16_siglip_224",
    "siglip-vit-b16-256px": "vit_base_patch16_siglip_256",
    "siglip-vit-b16-384px": "vit_base_patch16_siglip_384",
    "siglip-vit-so400m": "vit_so400m_patch14_siglip_224",
    "siglip-vit-so400m-384px": "vit_so400m_patch14_siglip_384",
}


class SigLIPViTBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(
            vision_backbone_id,
            SIGLIP_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
        )
