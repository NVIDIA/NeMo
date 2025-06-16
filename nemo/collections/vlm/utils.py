# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Except portions as noted which are Copyright (c) 2023 OpenGVLab and licensed under the MIT license found in LICENSE.
from torchvision import transforms as T
from torchvision.transforms import Compose
from torchvision.transforms.functional import InterpolationMode


IMAGENET_PIXEL_MEAN = [0.485, 0.456, 0.406]
IMAGENET_PIXEL_STD = [0.229, 0.224, 0.225]
SIGLIP_PIXEL_MEAN = [0.5, 0.5, 0.5]
SIGLIP_PIXEL_STD = [0.5, 0.5, 0.5]
CLIP_PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
RADIO_G_PIXEL_MEAN = [0.4850, 0.4560, 0.4060]
RADIO_G_PIXEL_STD = [0.2230, 0.2240, 0.2250]


pixel_statistics = {
    "clip": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "siglip": (SIGLIP_PIXEL_MEAN, SIGLIP_PIXEL_STD),
    "internvit": (IMAGENET_PIXEL_MEAN, IMAGENET_PIXEL_STD),
    "radio": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "radio-g": (RADIO_G_PIXEL_MEAN, RADIO_G_PIXEL_STD),
    "internvit300M": (IMAGENET_PIXEL_MEAN, IMAGENET_PIXEL_STD),
    "huggingface": (SIGLIP_PIXEL_MEAN, SIGLIP_PIXEL_STD),
}


# pylint: disable=C0301
# From https://github.com/OpenGVLab/InternVL/blob/c62fa4f7c850165d7386bdc48ac6bc5a6fab0864/internvl_chat/internvl/train/dataset.py#L685
# Copyright (c) 2023 OpenGVLab.
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio to target_ratio"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


class ImageTransform:
    """Image transformation."""

    def __init__(self, input_size, vision_model_type):
        self._transform = _build_transform(input_size, vision_model_type)
        self._vision_model_type = vision_model_type

    def __call__(
        self,
        img,
        img_h,
        img_w,
        use_tiling=False,
        max_num_tiles=1,
        use_thumbnail=False,
        augment=False,
        find_closest_aspect_ratio_fn=find_closest_aspect_ratio,
    ):
        assert not augment, "Image augmentation not implemented."
        if use_tiling:
            assert img_h == img_w, "dynamic tiling expects equal tile height and width"
            imgs = dynamic_preprocess(
                img,
                min_num=1,
                max_num=max_num_tiles,
                image_size=img_h,
                use_thumbnail=use_thumbnail,
                find_closest_aspect_ratio_fn=find_closest_aspect_ratio_fn,
            )
            imgs = [self._transform(img) for img in imgs]
        else:
            imgs = [self._transform(img)]

        return imgs


# pylint: disable=C0301
# From https://github.com/OpenGVLab/InternVL/blob/c62fa4f7c850165d7386bdc48ac6bc5a6fab0864/internvl_chat/internvl/train/dataset.py#L702
# Copyright (c) 2023 OpenGVLab.
def dynamic_preprocess(
    image,
    min_num=1,
    max_num=6,
    image_size=448,
    use_thumbnail=False,
    find_closest_aspect_ratio_fn=find_closest_aspect_ratio,
):
    """ """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio_fn(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


# pylint: disable=C0301
# Based on https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L79
# and https://github.com/OpenGVLab/InternVL/blob/aa521e6eb1df4cf153aa4118fcf13e673c055d46/internvl_chat/internvl/train/dataset.py#L276
def _build_transform(input_size, vision_model_type):
    if vision_model_type in ("siglip", "internvit", "internvit300M", "radio", "radio-g"):
        pixel_mean, pixel_std = pixel_statistics[vision_model_type]

        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=pixel_mean, std=pixel_std),
            ]
        )
    elif vision_model_type == "clip":
        pixel_mean, pixel_std = pixel_statistics[vision_model_type]

        transform = Compose(
            [
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.ToTensor(),
                T.Normalize(mean=pixel_mean, std=pixel_std),
            ]
        )
    elif vision_model_type.startswith("hf://"):
        from megatron.core.models.huggingface.module import get_hf_model_type

        model_type = get_hf_model_type(vision_model_type)
        if "siglip" in model_type:
            from transformers.models.siglip.image_processing_siglip import SiglipImageProcessor

            processor = SiglipImageProcessor(size={"height": input_size, "width": input_size})

            def transform(x):
                x = x.convert("RGB") if x.mode != "RGB" else x
                x = processor(x, return_tensors="pt")
                return x["pixel_values"][0]

        else:
            raise NotImplementedError(f"image processing not defined for huggingface model {vision_model_type}")
    else:
        raise NotImplementedError(f"image processing not defined for vision model {vision_model_type}")

    return transform
