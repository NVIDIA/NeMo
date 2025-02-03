"""
clip_vit.py
"""

from nemo.collections.vlm.openvla_bkp.data.prismatic.models.backbones.vision.base_vision import TimmViTBackbone

# Registry =>> Supported CLIP Vision Backbones (from TIMM)
CLIP_VISION_BACKBONES = {
    "clip-vit-b": "vit_base_patch16_clip_224.openai",
    "clip-vit-l": "vit_large_patch14_clip_224.openai",
    "clip-vit-l-336px": "vit_large_patch14_clip_336.openai",
}


# [IMPORTANT] By Default, TIMM initialized OpenAI CLIP models with the standard GELU activation from PyTorch.
#             HOWEVER =>> Original OpenAI models were trained with the quick_gelu *approximation* -- while it's
#                         a decent approximation, the resulting features are *worse*; this was a super tricky bug
#                         to identify, but luckily there's an easy fix (`override_act_layer`)
class CLIPViTBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(
            vision_backbone_id,
            CLIP_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
            override_act_layer="quick_gelu" if CLIP_VISION_BACKBONES[vision_backbone_id].endswith(".openai") else None,
        )
