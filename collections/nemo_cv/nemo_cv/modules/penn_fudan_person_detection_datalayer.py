# Copyright (C) NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Tomasz Kornuta"

from torch.utils.data.dataloader import default_collate
import os
import numpy as np
from PIL import Image

from ..utils.utils import pad_tensors_to_max

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from nemo.backends.pytorch.nm import DataLayerNM

from nemo.core import NeuralType, AxisType, DeviceType, \
    BatchTag, ChannelTag, HeightTag, WidthTag, ListTag, BoundingBoxTag


class PennFudanPedestrianDataset(Dataset):
    """
    Dataset containing images of pedestrians for object detection and
    image segmentation (dense prediction).
    """

    def __init__(self, data_folder="~/data/PennFudanPed"):
        """
        Dataset constructor.
        Args:
            data_folder: path to the folder with data, can be relative to user.

        """
        # Get absolute path.
        self.abs_data_folder = os.path.expanduser(data_folder)

        # Load and sort all the image files.
        self.imgs = list(sorted(
            os.listdir(os.path.join(self.abs_data_folder, "PNGImages"))))
        self.masks = list(sorted(
            os.listdir(os.path.join(self.abs_data_folder, "PedMasks"))))

        # Image transforms - to tensor.
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        """
        Returns a single sample, consisting of:
            - index of a sample (Int64Tensor[1]): an image identifier.
            It should be unique between all the images in the dataset,
            can be used during evaluation
            - image: a PIL Image of size (H, W)
            - masks (UInt8Tensor[N, H, W]): The segmentation masks
            for each one of the objects
            - boxes (FloatTensor[N, 4]): the coordinates of the N bounding
            boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
            - labels (Int64Tensor[N]): the label for each bounding box
            - area (Tensor[N]): The area of the bounding box. This is used
            during evaluation with the COCO metric, to separate the metric
            scores between small, medium and large boxes.
            - iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be
            ignored during evaluation.
            - num_objects_per_image (UINT8Tensor[N]): artificial variable 
            storing number of objects (bounding boxes) per image.
        """

        # Load given image and  mask.
        img_path = os.path.join(self.abs_data_folder,
                                "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.abs_data_folder,
                                 "PedMasks", self.masks[idx])

        # Conver image to RGB.
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        # Each color in mask corresponds to a different instance,
        # with 0 being background.
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # Instances are encoded as different colors.
        obj_ids = np.unique(mask)
        # First id is the background, so remove it.
        obj_ids = obj_ids[1:]

        # Split the color-encoded mask into a set of binary masks.
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask.
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        #print("Image {} as {} bounding boxes".format(self.imgs[idx], num_objs))

        # Transform index to tensor.
        image_id = torch.tensor(idx, dtype=torch.int64)

        # Change to tensors.
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        targets = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Calculate the area.
        area = torch.as_tensor(
            boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Suppose all instances are not crowd.
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Change to tensor as well.
        num_objs = torch.tensor(num_objs, dtype=torch.uint8)

        return image_id, img, boxes, targets, masks, area, iscrowd, num_objs

    def __len__(self):
        return len(self.imgs)


class PennFudanDataLayer(DataLayerNM):
    """Wrapper around the PennFudan dataset.
    """

    @staticmethod
    def create_ports():
        """
        Creates definitions of input and output ports.
        """
        input_ports = {}
        output_ports = {
            # Batch of indices.
            "indices": NeuralType({0, AxisType(BatchTag)}),

            # Batch of images.
            "images": NeuralType({0: AxisType(BatchTag),
                                  1: AxisType(ChannelTag, 3),
                                  2: AxisType(HeightTag),
                                  3: AxisType(WidthTag)}),

            # Batch of bounding boxes.
            "bounding_boxes": NeuralType({0: AxisType(BatchTag),
                                          1: AxisType(ListTag),
                                          2: AxisType(BoundingBoxTag)}),
            # Batch of targets.
            "targets": NeuralType({0: AxisType(BatchTag)}),

            # Batch of masks.
            "masks": NeuralType({0: AxisType(BatchTag),
                                 # Each channel = 1 object.
                                 1: AxisType(ChannelTag),
                                 2: AxisType(HeightTag),
                                 3: AxisType(WidthTag)}),
            # Batch of areas.
            "areas": NeuralType({0: AxisType(BatchTag)}),

            # Batch of "is crowd"s.
            "iscrowds": NeuralType({0: AxisType(BatchTag)}),

            # "Artificial" variable - tensor storing numbers of objects.
            "num_objects": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def __init__(
        self,
        batch_size,
        shuffle=True,
        data_folder="~/data/PennFudanPed"
    ):
        """
        Initializes the datalayer.

        Args:
            batch_size: size of batch
            shuffle: shuffle data(True by default)
            data_folder: path to the folder with data, can be relative to user.
        """
        # Passing the default params to base init call.
        DataLayerNM.__init__(self)

        # Do we need to set those??
        self._batch_size = batch_size
        self._shuffle = shuffle

        self._dataset = PennFudanPedestrianDataset()

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None

    def collate_fn(self, batch):
        """
        Overloaded batch collate - zips batch together.

        Args:
            batch: list of samples, each defined as "image_id, img, boxes,
            targets, masks, area, iscrowd, num_objs"
        """
        # Create a batch consisting of a samples zipped "element"-wise.
        # Elements are: image_id, img, boxes, targets, masks, area, iscrowd
        zipped_batch = list(tuple(zip(*batch)))

        # Replace the images with padded_images.
        zipped_batch[1] = pad_tensors_to_max(zipped_batch[1])

        #print(" !!! Bounding boxes per image !!!")
        # for item in zipped_batch[2]:
        #    print(item.size())

        # Pad number of bboxes per image.
        zipped_batch[2] = pad_tensors_to_max(zipped_batch[2])

        #print(" !!! Targets per image !!!")
        # for item in zipped_batch[3]:
        #    print(item.size())

        # Pad targets.
        zipped_batch[3] = pad_tensors_to_max(zipped_batch[3])

        # Pad masks.
        zipped_batch[4] = pad_tensors_to_max(zipped_batch[4])

        # Pad areas.
        zipped_batch[5] = pad_tensors_to_max(zipped_batch[5])
        zipped_batch[6] = pad_tensors_to_max(zipped_batch[6])

        # Finally, collate.
        collated_batch = [torch.stack(zb) for zb in zipped_batch]

        return collated_batch
