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

from typing import Optional

import torch
import torchvision.transforms.functional as transforms_F
from loguru import logger as logging

from nemo.collections.physicalai.tokenizer.data.augmentors.augmentor import Augmentor
from nemo.collections.physicalai.tokenizer.data.augmentors.image.misc import (
    obtain_augmentation_size,
    obtain_image_size,
)


class CenterCrop(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs center crop.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
            We also save the cropping parameters in the aug_params dict
            so that it will be used by other transforms.
        """
        assert (self.args is not None) and ("size" in self.args), "Please specify size in args"

        img_size = obtain_augmentation_size(data_dict, self.args)
        width, height = img_size

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        for key in self.input_keys:
            data_dict[key] = transforms_F.center_crop(data_dict[key], [height, width])

        # We also add the aug params we use. This will be useful for other transforms
        crop_x0 = (orig_w - width) // 2
        crop_y0 = (orig_h - height) // 2
        cropping_params = {
            "resize_w": orig_w,
            "resize_h": orig_h,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_w": width,
            "crop_h": height,
        }

        if "aug_params" not in data_dict:
            data_dict["aug_params"] = dict()

        data_dict["aug_params"]["cropping"] = cropping_params
        data_dict["padding_mask"] = torch.zeros((1, cropping_params["crop_h"], cropping_params["crop_w"]))
        return data_dict


class RandomCrop(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs random crop.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict where images are center cropped.
            We also save the cropping parameters in the aug_params dict
            so that it will be used by other transforms.
        """

        img_size = obtain_augmentation_size(data_dict, self.args)
        width, height = img_size

        orig_w, orig_h = obtain_image_size(data_dict, self.input_keys)
        # Obtaining random crop coords
        try:
            crop_x0 = int(torch.randint(0, orig_w - width + 1, size=(1,)).item())
            crop_y0 = int(torch.randint(0, orig_h - height + 1, size=(1,)).item())
        except Exception:
            logging.warning(
                f"Random crop failed. Performing center crop, original_size(wxh): {orig_w}x{orig_h}, random_size(wxh): {width}x{height}"
            )
            for key in self.input_keys:
                data_dict[key] = transforms_F.center_crop(data_dict[key], [height, width])
            crop_x0 = (orig_w - width) // 2
            crop_y0 = (orig_h - height) // 2

        # We also add the aug params we use. This will be useful for other transforms
        cropping_params = {
            "resize_w": orig_w,
            "resize_h": orig_h,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_w": width,
            "crop_h": height,
        }

        if "aug_params" not in data_dict:
            data_dict["aug_params"] = dict()

        data_dict["aug_params"]["cropping"] = cropping_params

        # We must perform same random cropping for all input keys
        for key in self.input_keys:
            data_dict[key] = transforms_F.crop(data_dict[key], crop_y0, crop_x0, height, width)
        return data_dict
