# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.

# pylint: disable=C0115,C0301

from typing import Any, List

import attrs

from nemo.collections.diffusion.sampler.conditioner import (
    BooleanFlag,
    ReMapkey,
    TextAttr,
    VideoConditioner,
    VideoExtendConditioner,
)


@attrs.define(slots=False)
class TextConfig:

    obj: Any = TextAttr()  # No arguments
    dropout_rate: float = 0.2
    input_keys: List[str] = attrs.field(factory=lambda: ["t5_text_embeddings", "t5_text_mask"])


@attrs.define(slots=False)
class ActionControlConfig:
    """
    Action control configuration for V2W model conditioning.
    """

    # default embedding model for action control for reference/readability, overwrite using DiT model config!
    obj: Any = ReMapkey(output_key="action_control_condition", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "action"


@attrs.define(slots=False)
class FPSConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `fps`.
    """

    obj: Any = ReMapkey(output_key="fps", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "fps"


@attrs.define(slots=False)
class PaddingMaskConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `fps`.
    """

    obj: Any = ReMapkey(output_key="padding_mask", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "padding_mask"


@attrs.define(slots=False)
class ImageSizeConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `fps`.
    """

    obj: Any = ReMapkey(output_key="image_size", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "image_size"


@attrs.define(slots=False)
class NumFramesConfig:
    """
    Remap the key from the input dictionary to the output dictionary. For `num_frames`.
    """

    obj: Any = ReMapkey(output_key="num_frames", dtype=None)
    dropout_rate: float = 0.0
    input_key: str = "num_frames"


@attrs.define(slots=False)
class VideoCondBoolConfig:

    obj: Any = BooleanFlag(output_key="video_cond_bool")

    dropout_rate: float = 0.0
    input_key: str = "fps"  # This is a placeholder, we never use this value
    # Config below are for long video generation only
    compute_loss_for_condition_region: bool = False  # Compute loss for condition region

    # How to sample condition region during training. "first_random_n" set the first n frames to be condition region, n is random, "random" set the condition region to be random,
    condition_location: str = "first_n"
    random_conditon_rate: float = 0.5  # The rate to sample the condition region randomly
    first_random_n_num_condition_t_max: int = (
        4  # The maximum number of frames to sample as condition region, used when condition_location is "first_random_n"
    )
    first_random_n_num_condition_t_min: int = (
        0  # The minimum number of frames to sample as condition region, used when condition_location is "first_random_n"
    )

    # How to dropout value of the conditional input frames
    cfg_unconditional_type: str = (
        "zero_condition_region_condition_mask"  # Unconditional type. "zero_condition_region_condition_mask" set the input to zero for condition region, "noise_x_condition_region" set the input to x_t, same as the base model
    )

    # How to corrupt the condition region
    apply_corruption_to_condition_region: str = (
        "noise_with_sigma_fixed"  # Apply corruption to condition region, option: "gaussian_blur", "noise_with_sigma", "clean" (inference), "noise_with_sigma_fixed" (inference)
    )
    # Inference only option: list of sigma value for the corruption at different chunk id, used when apply_corruption_to_condition_region is "noise_with_sigma" or "noise_with_sigma_fixed"
    apply_corruption_to_condition_region_sigma_value: list[float] = [0.001, 0.2] + [
        0.5
    ] * 10  # Sigma value for the corruption, used when apply_corruption_to_condition_region is "noise_with_sigma_fixed"

    # Add augment_sigma condition to the network
    condition_on_augment_sigma: bool = False
    # The following arguments is to match with previous implementation where we use train sde to sample augment sigma (with adjust video noise turn on)
    augment_sigma_sample_p_mean: float = 0.0  # Mean of the augment sigma
    augment_sigma_sample_p_std: float = 1.0  # Std of the augment sigma
    augment_sigma_sample_multiplier: float = 4.0  # Multipler of augment sigma

    # Add pose condition to the network
    add_pose_condition: bool = False

    # Sample PPP... from IPPP... sequence
    sample_tokens_start_from_p_or_i: bool = False

    # Normalize the input condition latent
    normalize_condition_latent: bool = False


BaseVideoConditionerConfig: Any = VideoConditioner(
    text=TextConfig(),
)

VideoConditionerFPSConfig: Any = VideoConditioner(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
)

VideoConditionerFpsSizePaddingConfig: Any = VideoConditioner(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
)

VideoExtendConditionerConfig: Any = VideoExtendConditioner(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
    video_cond_bool=VideoCondBoolConfig(),
)

VideoActionExtendConditionerConfig: Any = VideoExtendConditioner(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
    video_cond_bool=VideoCondBoolConfig(),
    action_ctrl=ActionControlConfig(),
)
