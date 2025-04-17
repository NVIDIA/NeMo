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

from nemo.collections.physicalai.tokenizer.data.augmentors.augmentor import Augmentor


class Normalize(Augmentor):
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

        mean = self.args["mean"]
        std = self.args["std"]

        for key in self.input_keys:
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(dtype=torch.get_default_dtype()).div(255)
            else:
                data_dict[key] = transforms_F.to_tensor(data_dict[key])  # division by 255 is applied in to_tensor()

            data_dict[key] = transforms_F.normalize(tensor=data_dict[key], mean=mean, std=std)
        return data_dict
