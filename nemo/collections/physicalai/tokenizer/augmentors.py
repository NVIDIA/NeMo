# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# pylint: disable=C0115,C0116,C0301


"""Additional augmentors for image and video training loops."""

import logging
import random
from typing import Optional

import torch

from nemo.collections.physicalai.tokenizer.data.augmentors.augmentor import Augmentor
from nemo.collections.physicalai.tokenizer.data.augmentors.image.cropping import RandomCrop
from nemo.collections.physicalai.tokenizer.data.augmentors.image.misc import (
    obtain_augmentation_size,
    obtain_image_size,
)
from nemo.collections.physicalai.tokenizer.data.augmentors.image.resize import ResizeSmallestSideAspectPreserving


class LossMask(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs data normalization.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
        """
        assert self.args is not None, "Please specify args"
        mask_config = self.args["masking"]

        input_key = self.input_keys[0]
        default_mask = torch.ones_like(data_dict[input_key])
        loss_mask = mask_config["nonhuman_mask"] * default_mask
        for curr_key in mask_config:
            if curr_key not in self.input_keys:
                continue
            curr_mask = data_dict[curr_key]
            curr_weight = mask_config[curr_key]
            curr_loss_mask = curr_mask * curr_weight + (1 - curr_mask) * loss_mask
            loss_mask = torch.max(curr_loss_mask, loss_mask)
            _ = data_dict.pop(curr_key)
        data_dict["loss_mask"] = loss_mask
        return data_dict


class UnsqueezeImage(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs horizontal flipping.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
        """
        for key in self.input_keys:
            data_dict[key] = data_dict[key].unsqueeze(1)

        return data_dict


class RandomReverse(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs random temporal reversing of frames.

        Args:
            data_dict (dict): Input data dict, CxTxHxW
        Returns:
            data_dict (dict): Output dict where videos are randomly reversed.
        """
        assert self.args is not None
        p = self.args.get("prob", 0.5)
        coin_flip = torch.rand(1).item() <= p
        for key in self.input_keys:
            if coin_flip:
                data_dict[key] = torch.flip(data_dict[key], dims=[1])

        return data_dict


class RenameInputKeys(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Rename the input keys from the data dict.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with keys renamed.
        """
        assert len(self.input_keys) == len(self.output_keys)
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            if input_key in data_dict:
                data_dict[output_key] = data_dict.pop(input_key)
        return data_dict


class CropResizeAugmentor(Augmentor):
    def __init__(
        self,
        input_keys: list,
        output_keys: Optional[list] = None,
        crop_args: Optional[dict] = None,
        resize_args: Optional[dict] = None,
        args: Optional[dict] = None,
    ) -> None:
        super().__init__(input_keys, output_keys, args)
        self.crop_args = crop_args
        self.resize_args = resize_args
        self.crop_op = RandomCrop(input_keys, output_keys, crop_args)
        self.resize_op = ResizeSmallestSideAspectPreserving(input_keys, output_keys, resize_args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs random temporal reversing of frames.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where videso are randomly reversed.
        """
        assert self.args is not None
        p = self.args.get("prob", 0.5)

        if p > 0.0:
            crop_img_size = obtain_augmentation_size(data_dict, self.crop_args)
            crop_width, crop_height = crop_img_size
            orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
            if orig_w < crop_width or orig_h < crop_height:
                logging.warning(
                    f"Data size ({orig_w}, {orig_h}) is smaller than crop size ({crop_width}, {crop_height}), skip the crop augmentation."
                )
            coin_flip = torch.rand(1).item() <= p
            if coin_flip and crop_width <= orig_w and crop_height <= orig_h:
                data_dict = self.crop_op(data_dict)
                return data_dict

        data_dict = self.resize_op(data_dict)
        data_dict = self.crop_op(data_dict)

        return data_dict


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    Args:
            size (int): Desired length of frames will be seen in the model.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, total_frames):
        rand_end = max(0, total_frames - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, total_frames)
        return begin_index, end_index
