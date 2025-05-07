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

# pylint: disable=C0115,C0116,C0301

import random
from functools import partial
from typing import Any, Optional

import cv2
import matplotlib.colors as mcolors
import numpy as np
import pycocotools
import torch
import torchvision.transforms.functional as transforms_F

from cosmos1.models.diffusion.config.ctrl.blurs import (
    AnisotropicDiffusionConfig,
    BilateralFilterConfig,
    BlurAugmentorConfig,
    GaussianBlurConfig,
    GuidedFilterConfig,
    LaplacianOfGaussianConfig,
    MedianBlurConfig,
)
from cosmos1.models.diffusion.datasets.augmentors.guided_filter import FastGuidedFilter
from cosmos1.models.diffusion.datasets.augmentors.human_keypoint_utils import coco_wholebody_133_skeleton
from cosmos1.utils import log

IMAGE_RES_SIZE_INFO: dict[str, tuple[int, int]] = {
    "1080": {  # the image format does not support 1080, but here we match it with video resolution
        "1,1": (1024, 1024),
        "4,3": (1440, 1056),
        "3,4": (1056, 1440),
        "16,9": (1920, 1056),
        "9,16": (1056, 1920),
    },
    "1024": {"1,1": (1024, 1024), "4,3": (1280, 1024), "3,4": (1024, 1280), "16,9": (1280, 768), "9,16": (768, 1280)},
    # 720; mainly for make sure it matches video resolution conventions
    "720": {"1,1": (960, 960), "4,3": (960, 704), "3,4": (704, 960), "16,9": (1280, 704), "9,16": (704, 1280)},
    "512": {"1,1": (512, 512), "4,3": (640, 512), "3,4": (512, 640), "16,9": (640, 384), "9,16": (384, 640)},
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
}


VIDEO_RES_SIZE_INFO: dict[str, tuple[int, int]] = {
    "1080": {  # 1080p doesn't have 1:1
        "1,1": (1024, 1024),
        "4,3": (1440, 1056),
        "3,4": (1056, 1440),
        "16,9": (1920, 1056),
        "9,16": (1056, 1920),
    },
    # 1024; the video format does not support it, but here we match it with image resolution
    "1024": {"1,1": (1024, 1024), "4,3": (1280, 1024), "3,4": (1024, 1280), "16,9": (1280, 768), "9,16": (768, 1280)},
    "720": {"1,1": (960, 960), "4,3": (960, 704), "3,4": (704, 960), "16,9": (1280, 704), "9,16": (704, 1280)},
    "512": {"1,1": (512, 512), "4,3": (640, 512), "3,4": (512, 640), "16,9": (640, 384), "9,16": (384, 640)},
    "480": {"1,1": (480, 480), "4,3": (640, 480), "3,4": (480, 640), "16,9": (768, 432), "9,16": (432, 768)},
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
}


class Augmentor:
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        r"""Base augmentor class

        Args:
            input_keys (list): List of input keys
            output_keys (list): List of output keys
            args (dict): Arguments associated with the augmentation
        """
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.args = args

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise ValueError("Augmentor not implemented")


def resize_frames(frames, is_image, data_dict):
    # Resize the frames to target size before computing control signals to save compute.
    need_reshape = len(frames.shape) < 4
    if need_reshape:  # HWC -> CTHW
        frames = frames.transpose((2, 0, 1))[:, None]
    H, W = frames.shape[2], frames.shape[3]

    if "__url__" in data_dict and "aspect_ratio" in data_dict["__url__"].meta.opts:
        aspect_ratio = data_dict["__url__"].meta.opts["aspect_ratio"]
    elif "aspect_ratio" in data_dict:  # Non-webdataset format
        aspect_ratio = data_dict["aspect_ratio"]
    else:
        aspect_ratio = "16,9"
    RES_SIZE_INFO = IMAGE_RES_SIZE_INFO if is_image else VIDEO_RES_SIZE_INFO
    new_W, new_H = RES_SIZE_INFO["720"][aspect_ratio]
    scaling_ratio = min((new_W / W), (new_H / H))
    if scaling_ratio < 1:
        W, H = int(scaling_ratio * W + 0.5), int(scaling_ratio * H + 0.5)
        frames = [
            cv2.resize(_image_np, (W, H), interpolation=cv2.INTER_AREA) for _image_np in frames.transpose((1, 2, 3, 0))
        ]
        frames = np.stack(frames).transpose((3, 0, 1, 2))
    if need_reshape:  # CTHW -> HWC
        frames = frames[:, 0].transpose((1, 2, 0))
    return frames


# frames CTHW
def apply_gaussian_blur(frames: np.ndarray, ksize: int = 5, sigmaX: float = 1.0) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1  # ksize must be odd
    blurred_image = [
        cv2.GaussianBlur(_image_np, (ksize, ksize), sigmaX=sigmaX) for _image_np in frames.transpose((1, 2, 3, 0))
    ]
    blurred_image = np.stack(blurred_image).transpose((3, 0, 1, 2))
    return blurred_image


class GaussianBlur:
    def __init__(self, config: GaussianBlurConfig) -> None:
        self.use_random = config.use_random
        self.config = config

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if self.use_random:
            ksize = np.random.randint(self.config.ksize_min, self.config.ksize_max + 1)
            sigmaX = np.random.uniform(self.config.sigmaX_min, self.config.sigmaX_max)
        else:
            ksize = self.config.ksize
            sigmaX = self.config.sigmaX
        return apply_gaussian_blur(frames, ksize, sigmaX)


def apply_guided_filter(frames: np.ndarray, radius: int, eps: float, scale: float) -> np.ndarray:
    blurred_image = [
        FastGuidedFilter(_image_np, radius, eps, scale).filter(_image_np)
        for _image_np in frames.transpose((1, 2, 3, 0))
    ]
    blurred_image = np.stack(blurred_image).transpose((3, 0, 1, 2))
    return blurred_image


class GuidedFilter:
    def __init__(self, config: GuidedFilterConfig) -> None:
        self.use_random = config.use_random
        self.config = config

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if self.use_random:
            radius = np.random.randint(self.config.radius_min, self.config.radius_max + 1)
            eps = np.random.uniform(self.config.eps_min, self.config.eps_max)
            scale = np.random.randint(self.config.scale_min, self.config.scale_max + 1)
        else:
            radius = self.config.radius
            eps = self.config.eps
            scale = self.config.scale
        return apply_guided_filter(frames, radius, eps, scale)


def apply_bilateral_filter(
    frames: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
    iter: int = 1,
) -> np.ndarray:
    blurred_image = []
    for _image_np in frames.transpose((1, 2, 3, 0)):
        for _ in range(iter):
            _image_np = cv2.bilateralFilter(_image_np, d, sigma_color, sigma_space)
        blurred_image += [_image_np]

    blurred_image = np.stack(blurred_image).transpose((3, 0, 1, 2))
    return blurred_image


class BilateralFilter:
    def __init__(self, config: BilateralFilterConfig) -> None:
        self.use_random = config.use_random
        self.config = config

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        config = self.config
        if self.use_random:
            d = np.random.randint(config.d_min, config.d_max)
            sigma_color = np.random.randint(config.sigma_color_min, config.sigma_color_max)
            sigma_space = np.random.randint(config.sigma_space_min, config.sigma_space_max)
            iter = np.random.randint(config.iter_min, config.iter_max)
        else:
            d = config.d
            sigma_color = config.sigma_color
            sigma_space = config.sigma_space
            iter = config.iter
        return apply_bilateral_filter(frames, d, sigma_color, sigma_space, iter)


def apply_median_blur(frames: np.ndarray, ksize=5) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1  # ksize must be odd
    blurred_image = [cv2.medianBlur(_image_np, ksize) for _image_np in frames.transpose((1, 2, 3, 0))]
    blurred_image = np.stack(blurred_image).transpose((3, 0, 1, 2))
    return blurred_image


class MedianBlur:
    def __init__(self, config: MedianBlurConfig) -> None:
        self.use_random = config.use_random
        self.config = config

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if self.use_random:
            ksize = np.random.randint(self.config.ksize_min, self.config.ksize_max + 1)
        else:
            ksize = self.config.ksize
        return apply_median_blur(frames, ksize)


def apply_laplacian_of_gaussian(
    frames: np.ndarray, ksize: int = 5, sigma: float = 1.4, binarize: bool = False, threshold: float = 0.0
) -> np.ndarray:
    """
    Apply Laplacian of Gaussian edge detection to a set of frames.

    Args:
    frames (np.ndarray): Input frames with shape (C, T, H, W)
    ksize (int): Size of the Gaussian kernel. Must be odd and positive.
    sigma (float): Standard deviation of the Gaussian distribution.
    binarize (bool): Whether to binarize the output edge map.
    threshold (float): Threshold for binarization (if binarize is True).

    Returns:
    np.ndarray: Edge-detected frames with shape (C, T, H, W).
    """
    # Ensure ksize is odd
    if ksize % 2 == 0:
        ksize += 1  # ksize must be odd

    edge_frames = []
    for frame in frames.transpose((1, 2, 3, 0)):  # (T, H, W, C)
        # Convert to grayscale if the image is in color
        if frame.shape[-1] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame.squeeze()

        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        normalized = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if binarize:
            _, edge_map = cv2.threshold(normalized, threshold, 255, cv2.THRESH_BINARY)
        else:
            edge_map = normalized

        # Expand dimensions to match input shape
        edge_map = np.repeat(edge_map[..., np.newaxis], frames.shape[0], axis=-1)
        edge_frames.append(edge_map)
    return np.stack(edge_frames).transpose((3, 0, 1, 2))  # (C, T, H, W)


class LaplacianOfGaussian:
    """
    Applies Laplacian of Gaussian edge detection to images or video frames.
    """

    def __init__(self, config: LaplacianOfGaussianConfig) -> None:
        self.use_random = config.use_random
        self.config = config

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """Apply LoG to input frames."""
        if self.use_random:
            ksize = np.random.randint(self.config.ksize_min, self.config.ksize_max + 1)
            sigma = np.random.uniform(self.config.sigma_min, self.config.sigma_max)
            binarize = np.random.choice([True, False]) if not self.config.binarize else self.config.binarize
            threshold = np.random.uniform(self.config.threshold_min, self.config.threshold_max) if binarize else 0
        else:
            ksize = self.config.ksize
            sigma = self.config.sigma
            binarize = self.config.binarize
            threshold = self.config.threshold
        return apply_laplacian_of_gaussian(frames, ksize, sigma, binarize, threshold)


def apply_anisotropic_diffusion(frames: np.ndarray, alpha: float, K: float, niters: int) -> np.ndarray:
    """
    Apply Anisotropic Diffusion to a set of frames.

    Args:
    frames (np.ndarray): Input frames with shape (C, T, H, W)
    alpha (float): The amount of time to step forward on each iteration (between 0 and 1)
    K (float): Sensitivity to edges
    niters (int): Number of iterations

    Returns:
    np.ndarray: Anisotropic-diffused frames with shape (C, T, H, W).
    """
    blurred_image = [
        cv2.ximgproc.anisotropicDiffusion(_image_np, alpha, K, niters) for _image_np in frames.transpose((1, 2, 3, 0))
    ]
    blurred_image = np.stack(blurred_image).transpose((3, 0, 1, 2))

    return blurred_image


class AnisotropicDiffusion:
    """
    Applies Anisotropic Diffusion to images or video frames.
    """

    def __init__(self, config: AnisotropicDiffusionConfig) -> None:
        self.use_random = config.use_random
        self.config = config

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if self.use_random:
            alpha = np.random.uniform(self.config.alpha_min, self.config.alpha_max)
            K = np.random.uniform(self.config.K_min, self.config.K_max)
            niters = np.random.randint(self.config.niters_min, self.config.niters_max + 1)
        else:
            alpha = self.config.alpha
            K = self.config.K
            niters = self.config.niters
        return apply_anisotropic_diffusion(frames, alpha, K, niters)


class Blur:
    def __init__(self, config: BlurAugmentorConfig, output_key: str = "") -> None:
        self.output_key = output_key if output_key else None

        probabilities = [combo.probability for combo in config.blur_combinations]
        total_prob = sum(probabilities)
        assert abs(total_prob - 1.0) < 1e-6, f"Probabilities must sum to 1.0, got {total_prob}"

        self.blur_combinations = config.blur_combinations
        self.downscale_factor = config.downscale_factor
        self.probabilities = probabilities
        self._set_blur_instances()

    def _set_blur_instances(self):
        if not self.blur_combinations:
            return
        self.blur_combinations_instances = []

        for blur_combination in self.blur_combinations:
            blur_mapping = {
                "gaussian": (GaussianBlur, blur_combination.gaussian_blur),
                "guided": (GuidedFilter, blur_combination.guided_filter),
                "bilateral": (BilateralFilter, blur_combination.bilateral_filter),
                "median": (MedianBlur, blur_combination.median_blur),
                "log": (LaplacianOfGaussian, blur_combination.log),
                "anisotropic": (AnisotropicDiffusion, blur_combination.anisotropic_diffusion),
            }

            cur_instances = []
            for blur_type in blur_combination.blur_types:
                assert blur_type in blur_mapping, f"Unknown {blur_type}. Needs to correct blur_type or blur_mapping."

                blur_class, blur_config = blur_mapping[blur_type]
                cur_instances.append(blur_class(blur_config))

            self.blur_combinations_instances.append(cur_instances)

        assert len(self.blur_combinations_instances) == len(
            self.blur_combinations
        ), "Number of blur_combinations_instances needs to match number of blur_combinations."

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        blur_instances = random.choices(self.blur_combinations_instances, weights=self.probabilities, k=1)[0]

        H, W = frames.shape[2], frames.shape[3]
        downscale_factor = random.choice(self.downscale_factor)
        if downscale_factor > 1:
            frames = [
                cv2.resize(_image_np, (W // downscale_factor, H // downscale_factor), interpolation=cv2.INTER_AREA)
                for _image_np in frames.transpose((1, 2, 3, 0))
            ]
            frames = np.stack(frames).transpose((3, 0, 1, 2))

        for ins in blur_instances:
            frames = ins(frames)

        if downscale_factor > 1:
            frames = [
                cv2.resize(_image_np, (W, H), interpolation=cv2.INTER_LINEAR)
                for _image_np in frames.transpose((1, 2, 3, 0))
            ]
            frames = np.stack(frames).transpose((3, 0, 1, 2))
        return frames


class AddControlInputBlurDownUp(Augmentor):
    """
    Main class for adding blurred input to the data dictionary.
    self.output_keys[0] indicates the types of blur added to the input.
    For example, control_input_gaussian_guided indicates that both Gaussian and Guided filters are applied
    """

    def __init__(
        self,
        input_keys: list,  # [key_load, key_img]
        output_keys: Optional[list] = [
            "control_input_gaussian_guided_bilateral_median_log"
        ],  # eg ["control_input_gaussian_guided"]
        args: Optional[dict] = None,  # not used
        use_random: bool = True,  # whether to use random parameters
        blur_config: BlurAugmentorConfig = BlurAugmentorConfig(),
        downup_preset: str | int = "medium",  # preset strength for downup factor
        min_downup_factor: int = 4,  # minimum downup factor
        max_downup_factor: int = 16,  # maximum downup factor
        downsize_before_blur: bool = False,  # whether to downsize before applying blur and then upsize or downup after blur
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        self.use_random = use_random
        downup_preset_values = {
            "none": 1,
            "very_low": min_downup_factor,
            "low": min_downup_factor,
            "medium": (min_downup_factor + max_downup_factor) // 2,
            "high": max_downup_factor,
            "very_high": max_downup_factor,
        }

        self.blur = Blur(config=blur_config, output_key=self.output_keys[0])

        self.downup_preset = downup_preset if isinstance(downup_preset, int) else downup_preset_values[downup_preset]
        self.downsize_before_blur = downsize_before_blur
        self.min_downup_factor = min_downup_factor
        self.max_downup_factor = max_downup_factor

    def _load_frame(self, data_dict: dict) -> tuple[np.ndarray, bool]:
        key_img = self.input_keys[1]
        frames = data_dict[key_img]
        frames = np.array(frames)
        is_image = False
        if len(frames.shape) < 4:
            frames = frames.transpose((2, 0, 1))[:, None]
            is_image = True
        return frames, is_image

    def __call__(self, data_dict: dict) -> dict:
        key_out = self.output_keys[0]
        frames, is_image = self._load_frame(data_dict)
        # H, W = frames.shape[2], frames.shape[3]

        # Resize the frames to target size before blurring.
        frames = resize_frames(frames, is_image, data_dict)
        H, W = frames.shape[2], frames.shape[3]
        # if is_image:
        #     data_dict[key_img] = torch.from_numpy(frames)[:, 0]
        # else:
        #     data_dict[key_img] = torch.from_numpy(frames)

        if self.use_random:
            scale_factor = random.randint(self.min_downup_factor, self.max_downup_factor + 1)
        else:
            scale_factor = self.downup_preset
        if self.downsize_before_blur:
            frames = [
                cv2.resize(_image_np, (W // scale_factor, H // scale_factor), interpolation=cv2.INTER_AREA)
                for _image_np in frames.transpose((1, 2, 3, 0))
            ]
            frames = np.stack(frames).transpose((3, 0, 1, 2))
        frames = self.blur(frames)
        if self.downsize_before_blur:
            frames = [
                cv2.resize(_image_np, (W, H), interpolation=cv2.INTER_LINEAR)
                for _image_np in frames.transpose((1, 2, 3, 0))
            ]
            frames = np.stack(frames).transpose((3, 0, 1, 2))
        if is_image:
            frames = frames[:, 0]
        # turn into tensor
        controlnet_img = torch.from_numpy(frames)
        if not self.downsize_before_blur:
            # Resize image
            controlnet_img = transforms_F.resize(
                controlnet_img,
                size=(int(H / scale_factor), int(W / scale_factor)),
                interpolation=transforms_F.InterpolationMode.BICUBIC,
                antialias=True,
            )
            controlnet_img = transforms_F.resize(
                controlnet_img,
                size=(H, W),
                interpolation=transforms_F.InterpolationMode.BICUBIC,
                antialias=True,
            )
        data_dict[key_out] = controlnet_img
        return data_dict


class AddControlInputCanny(Augmentor):
    """
    Add control input to the data dictionary. control input are expanded to 3-channels
    steps to add new items: modify this file, configs/conditioner.py, conditioner.py
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = ["control_input_canny"],
        args: Optional[dict] = None,
        use_random: bool = True,
        preset_strength="medium",
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        self.use_random = use_random
        self.preset_strength = preset_strength

    def __call__(self, data_dict: dict) -> dict:
        key_img = self.input_keys[1]
        key_out = self.output_keys[0]
        frames = data_dict[key_img]
        # Get lower and upper threshold for canny edge detection.
        if self.use_random:  # always on for training, always off for inference
            t_lower = np.random.randint(20, 100)  # Get a random lower thre within [0, 255]
            t_diff = np.random.randint(50, 150)  # Get a random diff between lower and upper
            t_upper = min(255, t_lower + t_diff)  # The upper thre is lower added by the diff
        else:
            if self.preset_strength == "none" or self.preset_strength == "very_low":
                t_lower, t_upper = 20, 50
            elif self.preset_strength == "low":
                t_lower, t_upper = 50, 100
            elif self.preset_strength == "medium":
                t_lower, t_upper = 100, 200
            elif self.preset_strength == "high":
                t_lower, t_upper = 200, 300
            elif self.preset_strength == "very_high":
                t_lower, t_upper = 300, 400
            else:
                raise ValueError(f"Preset {self.preset_strength} not recognized.")
        frames = np.array(frames)
        is_image = len(frames.shape) < 4

        # Resize the frames to target size before computing canny edges.
        frames = resize_frames(frames, is_image, data_dict)

        # Compute the canny edge map by the two thresholds.
        if is_image:
            edge_maps = cv2.Canny(frames, t_lower, t_upper)[None, None]
        else:
            edge_maps = [cv2.Canny(img, t_lower, t_upper) for img in frames.transpose((1, 2, 3, 0))]
            edge_maps = np.stack(edge_maps)[None]
        edge_maps = torch.from_numpy(edge_maps).expand(3, -1, -1, -1)
        if is_image:
            edge_maps = edge_maps[:, 0]
        data_dict[key_out] = edge_maps
        return data_dict


class AddControlInputIdentity(Augmentor):
    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = ["control_input_identity"],
        args: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        key_img = self.input_keys[1]
        key_out = self.output_keys[0]
        frames = np.array(data_dict[key_img])  # CTHW for video, HWC for image
        is_image = len(frames.shape) < 4
        if is_image:
            frames = frames.transpose((2, 0, 1))
        data_dict[key_out] = torch.from_numpy(frames).clone()  # CTHW for video, CHW for image
        return data_dict


class AddControlInput(Augmentor):
    """
    For backward compatibility. The previously trained models use legacy_process
    """

    def __init__(
        self,
        input_keys: list,
        output_keys=["control_input_gaussian_guided_bilateral_median_log_anisotropic"],
        args=None,
        blur_config: BlurAugmentorConfig = BlurAugmentorConfig(),
        use_random=True,
        preset_strength="medium",
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)

        self.process = AddControlInputBlurDownUp(
            input_keys,
            output_keys,
            args,
            blur_config=blur_config,
            downup_preset=preset_strength,  # preset strength for downup factor
            use_random=use_random,
        )

    def __call__(self, data_dict: dict) -> dict:
        return self.process(data_dict)


class AddControlInputComb(Augmentor):
    """
    Add control input to the data dictionary. control input are expanded to 3-channels
    steps to add new items: modify this file, configs/conditioner.py, conditioner.py
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = None,
        blur_config: BlurAugmentorConfig = None,
        args: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        assert "comb" in args
        self.comb = {}
        for class_name in args["comb"]:
            if class_name in [AddControlInput, AddControlInputBlurDownUp]:
                aug = class_name(input_keys=input_keys, args=args, blur_config=blur_config, **kwargs)
            else:
                aug = class_name(input_keys=input_keys, args=args, **kwargs)

            key = aug.output_keys[0]
            self.comb[key] = aug

    def __call__(self, data_dict: dict) -> dict:
        all_comb = []
        for k, v in self.comb.items():
            data_dict = v(data_dict)
            all_comb.append(data_dict.pop(k))
            if all_comb[-1].dim() == 4:
                all_comb[-1] = all_comb[-1].squeeze(1)
        all_comb = torch.cat(all_comb, dim=0)
        data_dict[self.output_keys[0]] = all_comb
        return data_dict


def get_augmentor_for_eval(
    input_key: str,
    output_key: str,
    blur_config: BlurAugmentorConfig = BlurAugmentorConfig(),
    preset_strength: str = "medium",
    blur_type: str = "gaussian,guided,bilateral,median,log,anisotropic",  # do we still need this value?
) -> AddControlInputComb:
    comb = []
    if "canny" in output_key:
        comb.append(partial(AddControlInputCanny, output_keys=["control_input_canny"]))
    if "upscale" in output_key:
        comb.append(partial(AddControlInputUpscale, output_keys=["control_input_upscale"]))
    if "identity" in output_key:
        comb.append(partial(AddControlInputIdentity, output_keys=["control_input_identity"]))
    if "depth" in output_key:
        comb.append(partial(AddControlInputDepth, output_keys=["control_input_depth"]))
    if "segmentation" in output_key:
        comb.append(partial(AddControlInputSeg, output_keys=["control_input_segmentation"]))
    if "blur" in output_key:
        comb.append(AddControlInput)
    process = AddControlInputComb(
        input_keys=["", input_key],
        output_keys=[output_key],
        args={"comb": comb},
        blur_config=blur_config,
        use_random=False,
        preset_strength=preset_strength,
    )
    return process


class AddControlInputDepth(Augmentor):
    """
    Add control input to the data dictionary. control input are expanded to 3-channels
    steps to add new items: modify this file, configs/conditioner.py, conditioner.py
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = ["control_input_depth"],
        args: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        if "control_input_depth" in data_dict:
            # already processed
            return data_dict
        if "video" not in data_dict:
            image = np.array(data_dict[self.input_keys[1]])
            h, w, _ = image.shape
            data_dict[self.output_keys[0]] = torch.from_numpy(np.zeros((3, h, w)).astype(np.uint8))
            return data_dict

        assert data_dict["chunk_index"] == data_dict["depth"]["chunk_index"]
        key_out = self.output_keys[0]
        depth = data_dict["depth"]["video"]
        # depth = resize_frames(depth, False, data_dict)
        data_dict[key_out] = depth
        return data_dict


# Array of 23 highly distinguishable colors in RGB format
PREDEFINED_COLORS_SEGMENTATION = np.array(
    [
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [255, 255, 0],  # Yellow
        [0, 255, 255],  # Cyan
        [255, 0, 255],  # Magenta
        [255, 140, 0],  # Dark Orange
        [255, 105, 180],  # Hot Pink
        [0, 0, 139],  # Dark Blue
        [0, 128, 128],  # Teal
        [75, 0, 130],  # Indigo
        [128, 0, 128],  # Purple
        [255, 69, 0],  # Red-Orange
        [34, 139, 34],  # Forest Green
        [128, 128, 0],  # Olive
        [70, 130, 180],  # Steel Blue
        [255, 215, 0],  # Gold
        [255, 222, 173],  # Navajo White
        [144, 238, 144],  # Light Green
        [255, 99, 71],  # Tomato
        [221, 160, 221],  # Plum
        [0, 255, 127],  # Spring Green
        [255, 255, 255],  # White
    ]
)


def generate_distinct_colors():
    """
    Generate `n` visually distinguishable and randomized colors.

    Returns:
        np.ndarray, (3)
    """
    # Randomize hue, saturation, and lightness within a range
    hue = random.uniform(0, 1)  # Full spectrum of hues
    saturation = random.uniform(0.1, 1)  # Vibrant colors
    lightness = random.uniform(0.2, 1.0)  # Avoid too dark

    r, g, b = mcolors.hsv_to_rgb((hue, saturation, lightness))
    return (np.array([r, g, b]) * 255).astype(np.uint8)


def segmentation_color_mask(segmentation_mask: np.ndarray, use_fixed_color_list: bool = False) -> np.ndarray:
    """
    Convert segmentation mask to color mask
    Args:
        segmentation_mask: np.ndarray, shape (num_masks, T, H, W)
    Returns:
        np.ndarray, shape (3, T, H, W), with each mask converted to a color mask, value [0,255]
    """

    num_masks, T, H, W = segmentation_mask.shape
    segmentation_mask_sorted = [segmentation_mask[i] for i in range(num_masks)]
    # Sort the segmentation mask by the number of non-zero pixels, from most to least
    segmentation_mask_sorted = sorted(segmentation_mask_sorted, key=lambda x: np.count_nonzero(x), reverse=True)

    output = np.zeros((3, T, H, W), dtype=np.uint8)
    if use_fixed_color_list:
        predefined_colors_permuted = PREDEFINED_COLORS_SEGMENTATION[
            np.random.permutation(len(PREDEFINED_COLORS_SEGMENTATION))
        ]
    else:
        predefined_colors_permuted = [generate_distinct_colors() for _ in range(num_masks)]
    # index the segmentation mask from last channel to first channel, i start from num_masks-1 to 0
    for i in range(num_masks):
        mask = segmentation_mask_sorted[i]
        color = predefined_colors_permuted[i % len(predefined_colors_permuted)]

        # Create boolean mask and use it for assignment
        bool_mask = mask > 0
        for c in range(3):
            output[c][bool_mask] = color[c]

    return output


def decode_partial_rle_width1(rle_obj, start_row, end_row):
    """
    Decode a partial RLE encoded mask with width = 1. In SAM2 output, the video mask (num_frame, height, width) are reshaped to (total_size, 1).
    Sometimes the video mask could be large, e.g. 1001x1080x1092 shape and it takes >1GB memory if using pycocotools, resulting in segmentation faults when training with multiple GPUs and data workers.
    This function is used to decode the mask for a subset of frames to reduce memory usage.

    Args:
        rle_obj (dict): RLE object containing:
            - 'size': A list [height, width=1] indicating the dimensions of the mask.
            - 'counts': A bytes or string object containing the RLE encoded data.
        start_row (int): The starting row (inclusive). It's computed from frame_start * height * width.
        end_row (int): The ending row (exclusive). It's computed from frame_end * height * width.

    Returns:
        numpy.ndarray: Decoded binary mask for the specified rows as a 1D numpy array.
    """
    height, width = rle_obj["size"]

    # Validate row range
    if width != 1:
        raise ValueError("This function is optimized for width=1.")
    if start_row < 0 or end_row > height or start_row >= end_row:
        raise ValueError("Invalid row range specified.")

    # Decode the RLE counts
    counts = rle_obj["counts"]
    if isinstance(counts, str):
        counts = np.frombuffer(counts.encode("ascii"), dtype=np.uint8)
    elif isinstance(counts, bytes):
        counts = np.frombuffer(counts, dtype=np.uint8)
    else:
        raise ValueError("Unsupported format for counts. Must be str or bytes.")

    # Interpret counts as a sequence of run lengths
    run_lengths = []
    current_val = 0
    i = 0
    while i < len(counts):
        x = 0
        k = 0
        more = True
        while more:
            c = counts[i] - 48
            x |= (c & 0x1F) << (5 * k)
            more = (c & 0x20) != 0
            i += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(run_lengths) > 2:
            x += run_lengths[-2]

        run_lengths.append(x)
        current_val += x
        if current_val > end_row:
            break
    # Initialize the partial mask
    idx_start = start_row
    idx_end = end_row
    partial_mask = np.zeros(idx_end - idx_start, dtype=np.uint8)
    partial_height = end_row - start_row
    idx = 0  # Current global index
    for i, run in enumerate(run_lengths):
        run_start = idx
        run_end = idx + run
        if run_end <= idx_start:
            # Skip runs entirely before the region
            idx = run_end
            continue
        if run_start >= idx_end:
            # Stop decoding once we pass the region
            break

        # Calculate overlap with the target region
        start = max(run_start, idx_start)
        end = min(run_end, idx_end)
        if start < end:
            partial_start = start - idx_start
            partial_end = end - idx_start
            partial_mask[partial_start:partial_end] = i % 2

        idx = run_end
    return partial_mask.reshape((partial_height, 1), order="F")


class AddControlInputSeg(Augmentor):
    """
    Add control input to the data dictionary. control input are expanded to 3-channels
    steps to add new items: modify this file, configs/conditioner.py, conditioner.py
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = ["control_input_segmentation"],
        thres_mb_python_decode: Optional[int] = 256,  # required: <= 512 for 7b
        use_fixed_color_list: bool = False,
        num_masks_max: int = 100,
        random_sample_num_masks: bool = True,
        args: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            thres_mb_python_decode: int, threshold of memory usage for python decode, in MB
            use_fixed_color_list: bool, if True, use predefined colors for segmentation masks. If False, generate random colors for segmentation masks.
            num_masks_max: int, maximum number of masks to sample
            random_sample_num_masks: bool, if True, sample number of masks randomly. If False, sample all masks in the data.

        """
        super().__init__(input_keys, output_keys, args)
        self.use_fixed_color_list = use_fixed_color_list
        self.num_masks_max = num_masks_max
        self.thres_mb_python_decode = thres_mb_python_decode
        self.random_sample_num_masks = random_sample_num_masks

    def __call__(self, data_dict: dict) -> dict:
        if "control_input_segmentation" in data_dict:
            # already processed
            log.info(
                f"control_input_segmentation already processed, shape={data_dict['control_input_segmentation'].shape}, dtype={data_dict['control_input_segmentation'].dtype}, value range: {data_dict['control_input_segmentation'].min()}, {data_dict['control_input_segmentation'].max()}"
            )
            return data_dict
        if "video" not in data_dict:
            image = np.array(data_dict[self.input_keys[1]])
            h, w, _ = image.shape
            data_dict[self.output_keys[0]] = torch.from_numpy(np.zeros((3, h, w)).astype(np.uint8))
            return data_dict
        frames = data_dict["video"]
        _, T, H, W = frames.shape

        all_masks = []
        # sample number of masks
        if self.random_sample_num_masks:
            num_masks = np.random.randint(0, min(self.num_masks_max + 1, len(data_dict["segmentation"]) + 1))
        else:
            num_masks = len(data_dict["segmentation"])
        mask_ids = np.arange(len(data_dict["segmentation"])).tolist()
        mask_ids_select = np.random.choice(mask_ids, num_masks, replace=False)
        # concat phrases
        segmentation_phrase_all = [data_dict["segmentation"][mid]["phrase"] for mid in mask_ids_select]
        segmentation_phrase_all = ";".join(segmentation_phrase_all)
        data_dict["segmentation_phrase_all"] = segmentation_phrase_all
        # obtrain frame indices
        frame_start = data_dict["frame_start"]
        frame_end = data_dict["frame_end"]
        frame_indices = np.arange(frame_start, frame_end).tolist()
        assert (
            len(frame_indices) == T
        ), f"frame_indices length {len(frame_indices)} != T {T}, likely due to video decoder using different fps, i.e. sample with stride. Need to return frame indices from video decoder."
        all_masks = np.zeros((num_masks, T, H, W)).astype(np.uint8)
        for idx, mid in enumerate(mask_ids_select):
            mask = data_dict["segmentation"][mid]
            shape = mask["segmentation_mask_rle"]["mask_shape"]
            num_byte_per_mb = 1024 * 1024
            # total number of elements in uint8 (1 byte) / num_byte_per_mb
            if shape[0] * shape[1] * shape[2] / num_byte_per_mb > self.thres_mb_python_decode:
                # Switch to python decode if the mask is too large to avoid out of shared memory

                rle = decode_partial_rle_width1(
                    mask["segmentation_mask_rle"]["data"],
                    frame_start * shape[1] * shape[2],
                    frame_end * shape[1] * shape[2],
                )
                partial_shape = (frame_end - frame_start, shape[1], shape[2])
                rle = rle.reshape(partial_shape) * 255
            else:
                rle = pycocotools.mask.decode(mask["segmentation_mask_rle"]["data"])
                rle = rle.reshape(shape) * 255
                # Select the frames that are in the video
                rle = np.stack([rle[i] for i in frame_indices])
            all_masks[idx] = rle
            del rle

        key_out = self.output_keys[0]
        # both value in [0,255]
        # control_input_segmentation is the colored segmentation mask, value in [0,255], shape (3, T, H, W)
        data_dict[key_out] = torch.from_numpy(segmentation_color_mask(all_masks, self.use_fixed_color_list))
        del all_masks  # free memory
        # Note: return this in trainnig will cause out of shared memory, uncomment only for data visualization
        # control_input_segmentation_raw is the raw segmentation mask, value in {0,255}, shape (num_masks, T, H, W)
        # all_masks_pad = np.zeros((self.num_masks, T, H, W))
        # all_masks_pad[: len(all_masks)] = all_masks
        # data_dict[key_out + "_raw"] = torch.from_numpy(all_masks_pad)
        return data_dict


class AddControlInputMask(Augmentor):
    """Add control input masks for video inpainting (replace) and outpainting operations."""

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = ["control_input_mask"],
        thres_mb_python_decode: Optional[int] = 256,
        args: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        self.masking_params = args.get("masking_params", {})
        self.thres_mb_python_decode = thres_mb_python_decode

    def generate_video_outpaint_mask(self, T: int, H: int, W: int) -> torch.Tensor:
        """Generate outpainting mask for video."""
        mask = np.ones((T, H, W), dtype=np.uint8)
        mask_sub_type = self.masking_params.get("mask_sub_type")

        if mask_sub_type == "one_side":
            region_range = self.masking_params["outpaint_region_range"]["one_side"]
            n = int(H * np.random.uniform(region_range.min, region_range.max))
            direction = np.random.choice(["top", "bottom", "left", "right"])

            for t in range(T):
                if direction == "top":
                    mask[t, :n] = 0
                elif direction == "bottom":
                    mask[t, -n:] = 0
                elif direction == "left":
                    mask[t, :, :n] = 0
                else:
                    mask[t, :, -n:] = 0

        elif mask_sub_type == "two_sides":
            region_range = self.masking_params["outpaint_region_range"]["two_sides"]
            n = int(H * np.random.uniform(region_range.min, region_range.max))
            direction = np.random.choice(["vertical", "horizontal"])

            for t in range(T):
                if direction == "vertical":
                    mask[t, :n] = mask[t, -n:] = 0
                else:
                    mask[t, :, :n] = mask[t, :, -n:] = 0

        else:
            region_range = self.masking_params["outpaint_region_range"]["all_sides"]
            nw = int(W * np.random.uniform(region_range.min, region_range.max))
            nh = int(H * np.random.uniform(region_range.min, region_range.max))

            mask_pos = mask_sub_type
            # Calculate positions once before the frame loop
            box_w = W - nw * 2
            box_h = H - nh * 2
            offset_h, offset_w = None, None
            if mask_pos == "corner":
                offset_h = np.random.choice([0, (H - box_h)])
                offset_w = np.random.choice([0, (W - box_w)])
            elif mask_pos == "random":
                offset_h = np.random.randint(0, H - box_h)
                offset_w = np.random.randint(0, W - box_w)

            # Apply same mask position to all frames
            for t in range(T):
                if mask_pos == "center":
                    mask[t, :nh] = mask[t, -nh:] = 0
                    mask[t, :, :nw] = mask[t, :, -nw:] = 0
                else:
                    mask[t] = 0
                    mask[t, offset_h : offset_h + box_h, offset_w : offset_w + box_w] = 1

        return torch.from_numpy(mask)

    def generate_video_replace_mask(
        self, frames: torch.Tensor, segmentation: list, frame_start: int, frame_end: int
    ) -> torch.Tensor:
        """Generate replacement mask for video."""
        _, T, H, W = frames.shape
        obj_idx = None
        mask = np.zeros((T, H, W), dtype=np.uint8)

        if segmentation:
            obj_idx, obj_mask = self.get_object_mask(segmentation, frame_start, frame_end, T, H, W)
            if obj_idx is not None:
                mask = 1 - obj_mask

        return obj_idx, torch.from_numpy(mask)

    def decode_sam_masks(self, mask_data, frame_start, frame_end, shape) -> np.ndarray:
        """Decode SAM RLE masks for video frames."""
        num_byte_per_mb = 1024 * 1024
        if shape[0] * shape[1] * shape[2] / num_byte_per_mb > self.thres_mb_python_decode:
            rle = decode_partial_rle_width1(
                mask_data,
                frame_start * shape[1] * shape[2],
                frame_end * shape[1] * shape[2],
            )
            partial_shape = (frame_end - frame_start, shape[1], shape[2])
            rle = rle.reshape(partial_shape)
        else:
            rle = pycocotools.mask.decode(mask_data)
            rle = rle.reshape(shape)
            frame_indices = np.arange(frame_start, frame_end).tolist()
            rle = np.stack([rle[i] for i in frame_indices])
        return rle

    def get_object_mask(
        self, segmentation, frame_start, frame_end, T, H, W
    ) -> tuple[Optional[int], Optional[np.ndarray]]:
        """Get a valid object mask from video segmentation data using quality criteria.

        Args:
            segmentation: List of dicts containing phrase and RLE mask data
            frame_start: Starting frame index
            frame_end: Ending frame index
            T: Number of frames
            H: Frame height
            W: Frame width

        Returns:
            tuple: (selected_idx, mask) where mask is a binary numpy array
        """
        if not segmentation:
            return None, None

        candidate_obj_indices = []
        candidate_masks = []
        total_frame_area = H * W

        temporal_presence_thre = self.masking_params["inpaint"].get("temporal_presence_thre", 0.2)
        frame_quality_score_thre = self.masking_params["inpaint"].get("frame_quality_score_thre", 0.2)

        for obj_idx, seg in enumerate(segmentation):
            if "segmentation_mask_rle" not in seg:
                continue

            # Decode mask
            shape = seg["segmentation_mask_rle"]["mask_shape"]
            mask = self.decode_sam_masks(seg["segmentation_mask_rle"]["data"], frame_start, frame_end, shape)
            mask = (mask > 0).astype(np.uint8)
            if mask is None:
                continue

            # Calculate temporal consistency
            temporal_presence = np.mean([np.any(mask[t]) for t in range(T)])
            if temporal_presence < temporal_presence_thre:  # Skip objects that aren't consistently present
                continue

            # Evaluate mask quality for each frame
            frame_qualities = []

            for t in range(T):
                cur_obj_mask = mask[t]
                if not np.any(cur_obj_mask):
                    continue

                mask_area = cur_obj_mask.sum()
                area_fraction_in_image = mask_area / total_frame_area

                # Get bounding box for the mask
                y_indices, x_indices = np.where(cur_obj_mask)
                if len(y_indices) == 0:
                    continue
                box = [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                area_fraction_in_bbox = mask_area / box_area if box_area > 0 else 0

                # Check quality criteria
                enough_box_area = area_fraction_in_bbox > self.masking_params["inpaint"]["area_fraction_in_bbox_thre"]
                enough_image_area = (
                    area_fraction_in_image > self.masking_params["inpaint"]["area_fraction_in_image_thre"]
                )

                if "area_fraction_in_image_ceil" in self.masking_params["inpaint"]:
                    not_too_much_image_area = (
                        area_fraction_in_image < self.masking_params["inpaint"]["area_fraction_in_image_ceil"]
                    )
                else:
                    not_too_much_image_area = True

                # Check border conditions
                if "touch_border_thre" in self.masking_params["inpaint"]:
                    is_foreground, _ = check_if_foreground(
                        cur_obj_mask, border_threshold=self.masking_params["inpaint"]["touch_border_thre"]
                    )
                else:
                    is_foreground = True

                if "close_to_border_thre" in self.masking_params["inpaint"]:
                    not_close_to_border = check_within_border_thres(
                        cur_obj_mask, border_threshold=self.masking_params["inpaint"]["close_to_border_thre"]
                    )
                else:
                    not_close_to_border = True

                # All criteria must be met
                if (
                    enough_box_area
                    and is_foreground
                    and enough_image_area
                    and not_close_to_border
                    and not_too_much_image_area
                ):
                    frame_qualities.append(1)
                else:
                    frame_qualities.append(0)

            # Object is a candidate if it meets criteria in more than frame_quality_score_thre proportion of frames
            frame_quality_score = np.mean(frame_qualities) if frame_qualities else 0
            if frame_quality_score > frame_quality_score_thre:
                candidate_obj_indices.append(obj_idx)
                candidate_masks.append(mask)

        if not candidate_obj_indices:
            return None, None
        # Randomly select from candidates that passed all criteria
        selected_idx = np.random.randint(len(candidate_obj_indices))
        return candidate_obj_indices[selected_idx], candidate_masks[selected_idx]

    def sample_mask_type(self) -> tuple[str, str]:
        """Sample mask type and subtype based on configured probabilities."""
        ratios = self.masking_params["task_ratio"]

        # Flatten probabilities into list of (type, subtype, prob)
        choices = []
        for mask_type, subtypes in ratios.items():
            if mask_type == "outpaint":
                for subtype, prob in subtypes.items():
                    if subtype == "all_sides":
                        for pos, pos_prob in prob.items():
                            choices.append((mask_type, pos, pos_prob))
                    else:
                        choices.append((mask_type, subtype, prob))
            else:
                for subtype, prob in subtypes.items():
                    choices.append((mask_type, subtype, prob))

        # Normalize probabilities
        probs = np.array([c[2] for c in choices])
        probs = probs / probs.sum()

        # Sample type and subtype
        idx = np.random.choice(len(choices), p=probs)
        return choices[idx][0], choices[idx][1]

    def __call__(self, data_dict: dict) -> dict:
        if "video" not in data_dict:
            image = np.array(data_dict[self.input_keys[1]])
            h, w, _ = image.shape
            data_dict[self.output_keys[0]] = torch.from_numpy(np.zeros((3, h, w)).astype(np.uint8))
            return data_dict

        frames = data_dict["video"]
        _, T, H, W = frames.shape
        frame_start = data_dict["frame_start"]
        frame_end = data_dict["frame_end"]

        mask_type, mask_sub_type = self.sample_mask_type()
        self.masking_params["mask_type"] = mask_type
        self.masking_params["mask_sub_type"] = mask_sub_type

        if mask_type == "outpaint":
            mask = self.generate_video_outpaint_mask(T, H, W)
            data_dict["mask_phrase"] = ""
        else:  # replace or inpaint
            selected_idx, mask = self.generate_video_replace_mask(
                frames, data_dict["segmentation"], frame_start, frame_end
            )
            if selected_idx is not None:
                # Store selected object phrase
                selected_phrase = data_dict["segmentation"][selected_idx]["phrase"]
                data_dict["mask_phrase"] = selected_phrase
            else:
                data_dict["mask_phrase"] = ""
        # Expand mask to 3 channels
        mask_tensor = mask.unsqueeze(0).expand(3, -1, -1, -1)
        data_dict[self.output_keys[0]] = mask_tensor
        data_dict["mask_type"] = mask_type
        data_dict["mask_sub_type"] = mask_sub_type

        return data_dict


class AddControlInputUpscale(Augmentor):
    """
    Add control input to the data dictionary. control input are expanded to 3-channels
    steps to add new items: modify this file, configs/conditioner.py, conditioner.py
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = ["control_input_upscale"],
        args: Optional[dict] = None,
        use_random: bool = True,
        preset_strength="medium",
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        self.use_random = use_random
        self.preset_strength = preset_strength

    def __call__(self, data_dict: dict, target_size: tuple = None) -> dict:
        if "control_input_upscale" in data_dict:
            # already processed
            return data_dict
        key_img = self.input_keys[1]
        key_out = self.output_keys[0]
        frames = data_dict[key_img]
        frames = np.array(frames)  # CTHW
        is_image = len(frames.shape) < 4
        if is_image:
            frames = frames.transpose((2, 0, 1))[:, None]
        h, w = frames.shape[-2:]
        frames = torch.from_numpy(frames.transpose(1, 0, 2, 3))  # TCHW

        if "__url__" in data_dict and "aspect_ratio" in data_dict["__url__"].meta.opts:
            aspect_ratio = data_dict["__url__"].meta.opts["aspect_ratio"]
        elif "aspect_ratio" in data_dict:  # Non-webdataset format
            aspect_ratio = data_dict["aspect_ratio"]
        else:
            aspect_ratio = "16,9"

        # Define the crop size
        RES_SIZE_INFO = IMAGE_RES_SIZE_INFO if is_image else VIDEO_RES_SIZE_INFO
        crop_width, crop_height = RES_SIZE_INFO["720"][aspect_ratio]

        if self.use_random:  # always on for training, always off for inference
            # During training, randomly crop a patch, then randomly downsize the video and resize it back.
            # Determine a random crop location
            top = torch.randint(0, max(0, h - crop_height) + 1, (1,)).item()
            left = torch.randint(0, max(0, w - crop_width) + 1, (1,)).item()
            cropped_frames = frames[:, :, top : top + crop_height, left : left + crop_width]

            # Randomly downsample the video
            # for 360p, 720p, 1080p -> 4k
            scaler = np.random.choice([1 / 6, 1 / 3, 0.5], p=[0.3, 0.5, 0.2])
            small_crop_width = int(crop_width * scaler)
            small_crop_height = int(crop_height * scaler)
            resized_frames = transforms_F.resize(
                cropped_frames,
                size=(small_crop_height, small_crop_width),
                interpolation=transforms_F.InterpolationMode.BICUBIC,
                antialias=True,
            )
            # Upsample to target size
            resized_frames = transforms_F.resize(
                resized_frames,
                size=(crop_height, crop_width),
                interpolation=transforms_F.InterpolationMode.BILINEAR,
            )
        else:
            if target_size is None:  # for validation
                # During validation, center crop a patch, then resize to target size.
                if self.preset_strength == "low":
                    scaler = 0.5
                elif self.preset_strength == "medium":
                    scaler = 1 / 3
                else:
                    scaler = 1 / 6
                small_crop_width = int(crop_width * scaler)
                small_crop_height = int(crop_height * scaler)

                # Center crop during inference
                top = (h - small_crop_height) // 2
                left = (w - small_crop_width) // 2

                # Perform the crop
                frames = frames[:, :, top : top + small_crop_height, left : left + small_crop_width]
                # Upsample to target size
                resized_frames = transforms_F.resize(
                    frames,
                    size=(crop_height, crop_width),
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                )
            else:  # for inference
                # During inference, directly resize to target size.
                new_h, new_w = target_size
                resized_frames = transforms_F.resize(
                    frames,
                    size=(new_h, new_w),
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                )
            cropped_frames = resized_frames

        resized_frames = resized_frames.permute(1, 0, 2, 3).contiguous()  # CTHW
        cropped_frames = cropped_frames.permute(1, 0, 2, 3).contiguous()  # CTHW

        if is_image:
            resized_frames = resized_frames[:, 0]
            cropped_frames = cropped_frames[:, 0]
        data_dict[key_out] = resized_frames
        data_dict[key_img] = cropped_frames
        return data_dict


class AddControlInputHumanKpts(Augmentor):
    """
    Add control input to the data dictionary. control input are expanded to 3-channels
    steps to add new items: modify this file, configs/conditioner.py, conditioner.py
    """

    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = ["control_input_human_kpts"],
        args: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        self.control_key_names = ["body-keypoints", "hand-keypoints"]

    def denormalize_pose_kpts(self, pose_kps: np.ndarray, h: int, w: int):
        """
        pose_kps has shape = (#keypoints, 2)
            or (#keypoints, 3) where the last dim is the confidence score.
        """
        if pose_kps is not None:
            out = pose_kps * np.array([w, h, 1])
            return out
        else:
            return None

    def draw_skeleton(
        self,
        img: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        kpt_thr: float = 0.5,
        radius: int = 2,
        line_width: int = 2,
    ):
        assert len(keypoints.shape) == 2
        keypoint_info, skeleton_info = (
            coco_wholebody_133_skeleton["keypoint_info"],
            coco_wholebody_133_skeleton["skeleton_info"],
        )
        vis_kpt = [s >= kpt_thr for s in scores]

        link_dict = {}
        for i, kpt_info in keypoint_info.items():
            kpt_color = tuple(kpt_info["color"])
            link_dict[kpt_info["name"]] = kpt_info["id"]

            kpt = keypoints[i]

            if vis_kpt[i]:
                img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius), kpt_color, -1)

        for i, ske_info in skeleton_info.items():
            link = ske_info["link"]
            pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

            if vis_kpt[pt0] and vis_kpt[pt1]:
                link_color = ske_info["color"]
                kpt0 = keypoints[pt0]
                kpt1 = keypoints[pt1]

                img = cv2.line(
                    img, (int(kpt0[0]), int(kpt0[1])), (int(kpt1[0]), int(kpt1[1])), link_color, thickness=line_width
                )
        return img

    def plot_kpt_img(self, kpts_np_dict: dict, h: int, w: int) -> torch.Tensor:
        """
        The raw human keypoint annotation are numpy arrays of pixel coordinates of the joints.
        This function plots the keypoints on a black background to form a 3-channel image compatible with controlnet.

        Args:
            kpts_np_dict (dict): A dict of keypoint annotations. Each value is a frame's annotation (a list of per-person dict).
            H (int): height of the image
            W (int): width of the image
        Returns:
            np.ndarray: keypoints of plotted on black background, shape = (3, T, H, W)
        """

        def plot_person_kpts(person_dict: dict, pose_vis_img: np.ndarray, h: int, w: int) -> np.ndarray:
            """in-place update the pose image"""
            body_keypoints = self.denormalize_pose_kpts(person_dict.get("body-keypoints"), h, w)
            hand_keypoints = self.denormalize_pose_kpts(person_dict.get("hand-keypoints"), h, w)
            assert (
                body_keypoints is not None and hand_keypoints is not None
            ), "Both body and hand keypoints must be present."
            # all_keypoints: shape=(133, 3). following coco-fullbody skeleton config. 3 channels are x, y, confidence
            all_keypoints = np.vstack([body_keypoints, hand_keypoints])
            kpts, scores = all_keypoints[..., :2], all_keypoints[..., -1]
            pose_vis_img = self.draw_skeleton(pose_vis_img, kpts, scores, kpt_thr=0.5)
            return pose_vis_img

        all_pose_img_frames = []
        # TODO: now loops over frames and persons. Check if this is a bottle net and try to optimize.
        for t, kpts_np_frame in kpts_np_dict.items():
            pose_vis_img = np.zeros([h, w, 3])

            # add each person's keypoints to this frame's pose image
            for person_dict in kpts_np_frame:
                plot_person_kpts(person_dict, pose_vis_img, h, w)  # (h, w, 3)
            all_pose_img_frames.append(pose_vis_img)

        pose_vis_vid = np.stack(all_pose_img_frames).transpose((3, 0, 1, 2)).astype(np.uint8)
        return pose_vis_vid

    def extend_annotations_for_vid_chunk(self, annotation_dict: dict, total_frames: int) -> dict:
        """
        For legacy data the annotations are done for chunks of every N frames (N=4).
        This function repeats each chunk's first frame annotation to the rest frames
        so that they become 'per-frame' and are ControlNet compatible.

        If the data is already per-frame annotated, then no need to call this.
        Args:
            annotation_dict (dict): Original annotations annotated every chunk_size frames.
                                    Each value is a list of dicts, where each dict has many
                                    human attributes. Here we only keep keypoint-relevant keys.
            total_frames (int): Total number of frames in the video.

        Returns:
            dict: extended annotations for all frames.
        """
        annotated_frame_idxs = sorted(list(annotation_dict.keys()))
        chunk_size = annotated_frame_idxs[1] - annotated_frame_idxs[0]  # typically 1 or 4
        if chunk_size == 1:
            # each person's dict can contain many irrelevant annotations (e.g. seg masks), here we only keep pose kpts
            annotation_dict_simpler = {
                key: [{k: v for k, v in sub_dict.items() if k in self.control_key_names} for sub_dict in sub_list]
                for key, sub_list in annotation_dict.items()
            }
            return annotation_dict_simpler

        extended_annotations = {}
        annotated_frames = sorted(annotation_dict.keys())

        for start_frame in annotated_frames:
            current_annotation = annotation_dict[start_frame]
            end_frame = min(start_frame + chunk_size, total_frames)
            for frame in range(start_frame, end_frame):
                extended_annotations[frame] = []
                for person in current_annotation:
                    person_annotation = {}
                    for control_key in self.control_key_names:
                        person_annotation[control_key] = np.copy(person[control_key])
                    extended_annotations[frame].append(person_annotation)
        return extended_annotations

    def __call__(self, data_dict: dict) -> dict:
        """
        human_annotations: loaded human annotation data pickle. One annotation per N frames.
        In the past we did N=4; for ControlNet data annotations we will do N=1.
        The pickle is a dict of annotated frames.
        The key is the frame number. For each frame, as there can be multiple people, we maintain a list of per-person
        dicts. Example:
        {
          0: [
                {'body-keypoints': <data_of_person1>, 'hand-keypoints': <data_of_person1>},
                {'body-keypoints': <data_of_person2>, 'hand-keypoints': <data_of_person2>},
           ], # frame 0, 2 people
          4: [
                 {'body-keypoints': <data_of_person1>, 'hand-keypoints': <data_of_person1>},
                 {'body-keypoints': <data_of_person2>, 'hand-keypoints': <data_of_person2>},
                 {'body-keypoints': <data_of_person3>, 'hand-keypoints': <data_of_person3>},
           ], # frame 4, 3 people
        ...
        }
        Note that for the same person, their idx in the per-frame list isn't guaranteed to be consistent.
        """
        human_annotations = data_dict.pop("human_annotation")

        frames = data_dict["video"]
        _, T, H, W = frames.shape

        # same dict format as `human_annotations` but now every frame has an annotation
        kpts_nparray_dict = self.extend_annotations_for_vid_chunk(human_annotations, T)

        # Colored human keypoints figure on black background. All persons in the same frame are plotted together. Shape: [3, T, H, W].
        kpts_cond_img = self.plot_kpt_img(kpts_nparray_dict, H, W)

        key_out = self.output_keys[0]
        data_dict[key_out] = torch.from_numpy(kpts_cond_img)
        return data_dict


if __name__ == "__main__":
    import sys

    from cosmos1.models.diffusion.config.ctrl.augmentors import (
        BilateralOnlyBlurAugmentorConfig,
        GaussianOnlyBlurAugmentorConfig,
    )
    from cosmos1.models.diffusion.inference.demo_video import save_video
    from cosmos1.models.diffusion.utils.inference_long_video import read_video_or_image_into_frames_BCTHW

    path_in = sys.argv[1]

    def main(input_file_path: str) -> None:
        max_length = 10
        video_input = read_video_or_image_into_frames_BCTHW(input_file_path, normalize=False)[0, :, :max_length]
        C, T, H, W = video_input.shape
        blur_processes = {
            "bilateral": BilateralOnlyBlurAugmentorConfig,
            "gaussian": GaussianOnlyBlurAugmentorConfig,
        }
        for blur_name, blur_process in blur_processes.items():
            for preset_strength in ["low", "medium", "high"]:
                process = get_augmentor_for_eval(
                    "video",
                    "control_input_blur",
                    preset_strength=preset_strength,
                    blur_config=blur_process[preset_strength],
                )
                output = process({"video": video_input})
                output = output["control_input_blur"].numpy().transpose((1, 2, 3, 0))

                output_file_path = f"{input_file_path[:-4]}_{blur_name}_{preset_strength}.mp4"
                save_video(
                    grid=output,
                    fps=5,
                    H=H,
                    W=W,
                    video_save_quality=9,
                    video_save_path=output_file_path,
                )

                print(f"Output video saved as {output_file_path}")

    main(path_in)

# flake8: noqa: F821
