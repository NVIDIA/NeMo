# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import numpy as np
import torch
from PIL import Image
from nemo.collections.vlm.utils import ImageTransform

class VisualProcessor:
    def __init__(self,
                 crop_height=512,
                 crop_width=512,
                 use_tiling=True,
                 max_num_tiles=12,
                 use_thumbnail=True,
                 augment=False,
                 vision_model_type="radio",
                 target_aspect_ratio=None):
        # Store the default crop size in a dict for compatibility
        self.crop_size = {'height': crop_height, 'width': crop_width}
        self.use_tiling = use_tiling
        self.max_num_tiles = max_num_tiles
        self.use_thumbnail = use_thumbnail
        self.augment = augment
        self.vision_model_type = vision_model_type
        self.target_aspect_ratio = target_aspect_ratio
        self._transform_img = ImageTransform(crop_height, vision_model_type)

    def preprocess(self, image, return_tensors='pt', do_center_crop=True, size=None, do_rescale=False):
        """
        Preprocess the image using get_visual_transform.

        Parameters:
          image (PIL.Image or np.ndarray): The input image.
          return_tensors (str, optional): If 'pt', returns a PyTorch tensor.
          do_center_crop (bool, optional): Whether to use the default center crop.
          size (dict, optional): If provided and do_center_crop is False, should contain a key "shortest_edge"
                                 to resize the image so its shortest side matches the given value.

        Returns:
          dict: A dictionary with key 'pixel_values' containing the processed image(s).
        """
        # If not doing center crop and a resize size is provided,
        # compute new dimensions based on the shortest edge.
        if not do_center_crop and size is not None and "shortest_edge" in size:
            shortest_edge = size["shortest_edge"]
            # Ensure we have a PIL Image (for resizing); if not, convert it.
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            w, h = image.size
            # Calculate new dimensions while preserving aspect ratio
            if w < h:
                new_w = shortest_edge
                new_h = int(h * shortest_edge / w)
            else:
                new_h = shortest_edge
                new_w = int(w * shortest_edge / h)
            image = image.resize((new_w, new_h), resample=Image.BICUBIC)
            # Use the resized dimensions as the target dimensions.
            target_width, target_height = new_w, new_h
        else:
            # Use the default crop size if center cropping.
            target_height = self.crop_size['height']
            target_width = self.crop_size['width']

        # Call the underlying transformation function.
        imgs = self._transform_img(
            image,
            img_h=target_height,
            img_w=target_width,
            use_tiling=self.use_tiling,
            max_num_tiles=self.max_num_tiles,
            use_thumbnail=self.use_thumbnail,
            augment=self.augment,
        )

        # Convert the resulting image(s) to PyTorch tensors if requested.
        if return_tensors == 'pt':
            if isinstance(imgs, list):
                imgs = [self.to_tensor(img) for img in imgs]
                imgs = torch.stack(imgs)
            else:
                imgs = self.to_tensor(imgs)

        return {'pixel_values': imgs}

    def to_tensor(self, img):
        """
        Convert an image (assumed to be a numpy array) to a PyTorch tensor.
        If the image is in HWC format (with 1 or 3 channels), it is permuted to CHW.
        """
        if isinstance(img, np.ndarray):
            tensor = torch.from_numpy(img)
        else:
            tensor = img  # Assume already a tensor
        if tensor.ndim == 3 and tensor.shape[2] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        return tensor


