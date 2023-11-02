# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from torchvision.io import ImageReadMode, read_image


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


def _size_tuple(size):
    if isinstance(size, int):
        return size, size
    else:
        assert len(size) == 2
        return size


class CenterCropResize:
    def __init__(self, target_size: int, interpolation: str = 'bilinear', fill_color: tuple = (0, 0, 0)):
        self.target_size = _size_tuple(target_size)
        self.interpolation = interpolation
        self.fill_color = fill_color

    def __call__(self, img):
        w, h = img.size
        img = np.array(img).astype(np.uint8)
        crop = min(w, h)
        img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
        image = Image.fromarray(img)
        if self.target_size is not None:
            interp_method = _pil_interp(self.interpolation)
            new_img = image.resize(self.target_size, resample=interp_method)
        return new_img


class CustomDataset(data.Dataset):
    def __init__(self, root, target_size=None):
        self.root = root
        self.files = [f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))]
        self.transform = transforms.ToTensor()
        self.target_size = target_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        # image = read_image(os.path.join(self.root, file), mode=ImageReadMode.RGB).type(torch.float32) / 255
        image = Image.open(os.path.join(self.root, file)).convert('RGB')
        if self.target_size is not None:
            image = image.resize((self.target_size, self.target_size), resample=Image.BICUBIC)
        image = self.transform(image)
        image = 2 * image - 1
        return image, file


class CocoDataset(data.Dataset):
    def __init__(self, root, ann_file, captions, transform=None, target_size=None):
        self.root = root
        self.coco = None
        self.captions = captions
        self.img_ids = [x['image_id'] for x in self.captions]
        self.has_annotations = 'image_info' not in ann_file
        self.transforms = [transforms.ToTensor()]
        if transform is not None:
            self.transforms.append(transform)
        self.target_size = target_size
        self.img_ids_invalid = []
        self.img_infos = []
        self._load_annotations(ann_file)

    def _load_annotations(self, ann_file):
        assert self.coco is None
        self.coco = COCO(ann_file)
        img_ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for img_id in self.img_ids:
            info = self.coco.loadImgs([img_id])[0]
            valid_annotation = not self.has_annotations or img_id in img_ids_with_ann
            if valid_annotation and min(info['width'], info['height']) >= 32:
                self.img_infos.append(info)
            else:
                self.img_ids_invalid.append(img_id)

    def __len__(self):
        return len(self.img_infos)

    def _compose(self, image):
        for t in self.transforms[::-1]:
            image = t(image)
        return image

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.img_infos[index]
        cap = self.captions[index]
        path = img_info['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.target_size is not None:
            image = image.resize((512, 512))
        image = self._compose(image)
        return image, cap
