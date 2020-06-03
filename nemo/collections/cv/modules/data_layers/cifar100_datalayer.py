# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================
# Copyright (C) IBM Corporation 2019
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
# =============================================================================

__author__ = "Tomasz Kornuta"

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/IBM/pytorchpipe/blob/develop/ptp/components/tasks/image_to_class/cifar_100.py
"""

from os.path import expanduser
from typing import Optional

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, Resize, ToTensor

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import AxisKind, AxisType, ClassificationTarget, ImageValue, Index, NeuralType, StringLabel
from nemo.utils.decorators import add_port_docs

__all__ = ['CIFAR100DataLayer']


class CIFAR100DataLayer(DataLayerNM, Dataset):
    """
    A "thin DataLayer" -  wrapper around the torchvision's CIFAR100 dataset.

    Reference page: http://www.cs.toronto.edu/~kriz/cifar.html
    """

    def __init__(
        self,
        height: int = 32,
        width: int = 32,
        data_folder: str = "~/data/cifar100",
        train: bool = True,
        name: Optional[str] = None,
        batch_size: int = 64,
        shuffle: bool = True,
    ):
        """
        Initializes the CIFAR100 datalayer.

        Args:
            height: image height (DEFAULT: 32)
            width: image width (DEFAULT: 32)
            data_folder: path to the folder with data, can be relative to user (DEFAULT: "~/data/cifar10")
            train: use train or test splits (DEFAULT: True)
            name: Name of the module (DEFAULT: None)
            batch_size: size of batch (DEFAULT: 64) [PARAMETER OF DATALOADER]
            shuffle: shuffle data (DEFAULT: True) [PARAMETER OF DATALOADER]
        """
        # Call the base class constructor of DataLayer.
        DataLayerNM.__init__(self, name=name)

        # Store height and width.
        self._height = height
        self._width = width

        # Create transformations: up-scale and transform to tensors.
        mnist_transforms = Compose([Resize((self._height, self._width)), ToTensor()])

        # Get absolute path.
        abs_data_folder = expanduser(data_folder)

        # Create the CIFAR10 dataset object.
        self._dataset = CIFAR100(root=abs_data_folder, train=train, download=True, transform=mnist_transforms)

        # Remember the params passed to DataLoader. :]
        self._batch_size = batch_size
        self._shuffle = shuffle

        # Process labels.
        all_labels = {
            "aquatic_mammals": "beaver, dolphin, otter, seal, whale".split(", "),
            "fish": "aquarium_fish, flatfish, ray, shark, trout".split(", "),
            "flowers": "orchid, poppy, rose, sunflower, tulip".split(", "),
            "food_containers": "bottle, bowl, can, cup, plate".split(", "),
            "fruit_and_vegetables": "apple, mushroom, orange, pear, sweet_pepper".split(", "),
            "household_electrical_devices": "clock, keyboard, lamp, telephone, television".split(", "),
            "household_furniture": "bed, chair, couch, table, wardrobe".split(", "),
            "insects": "bee, beetle, butterfly, caterpillar, cockroach".split(", "),
            "large_carnivores": "bear, leopard, lion, tiger, wolf".split(", "),
            "large_man-made_outdoor_things": "bridge, castle, house, road, skyscraper".split(", "),
            "large_natural_outdoor_scenes": "cloud, forest, mountain, plain, sea".split(", "),
            "large_omnivores_and_herbivores": "camel, cattle, chimpanzee, elephant, kangaroo".split(", "),
            "medium-sized_mammals": "fox, porcupine, possum, raccoon, skunk".split(", "),
            "non-insect_invertebrates": "crab, lobster, snail, spider, worm".split(", "),
            "people": "baby, boy, girl, man, woman".split(", "),
            "reptiles": "crocodile, dinosaur, lizard, snake, turtle".split(", "),
            "small_mammals": "hamster, mouse, rabbit, shrew, squirrel".split(", "),
            "trees": "maple_tree, oak_tree, palm_tree, pine_tree, willow_tree".split(", "),
            "vehicles_1": "bicycle, bus, motorcycle, pickup_truck, train".split(", "),
            "vehicles_2": "lawn_mower, rocket, streetcar, tank, tractor".split(", "),
        }

        coarse_word_to_ix = {}
        fine_to_coarse_mapping = {}
        fine_labels = []
        for coarse_id, (key, values) in enumerate(all_labels.items()):
            # Add mapping from coarse category name to coarse id.
            coarse_word_to_ix[key] = coarse_id
            # Add mappings from fine category names to coarse id.
            for value in values:
                fine_to_coarse_mapping[value] = coarse_id
            # Add values to list of fine labels.
            fine_labels.extend(values)

        # Sort fine labels.
        fine_labels = sorted(fine_labels)

        # Generate fine word mappings.
        fine_word_to_ix = {fine_labels[i]: i for i in range(len(fine_labels))}

        # Reverse mapping - for labels.
        self._fine_ix_to_word = {value: key for (key, value) in fine_word_to_ix.items()}

        # Reverse mapping - for labels.
        self._coarse_ix_to_word = {value: key for (key, value) in coarse_word_to_ix.items()}

        # Create fine to coarse id mapping.
        self._fine_to_coarse_id_mapping = {}
        for fine_label, fine_id in fine_word_to_ix.items():
            self._fine_to_coarse_id_mapping[fine_id] = fine_to_coarse_mapping[fine_label]
            # print(" {} ({}) : {} ".format(fine_label, fine_id, self.coarse_ix_to_word[fine_to_coarse_mapping[fine_label]]))

    @property
    @add_port_docs()
    def output_ports(self):
        """
        Creates definitions of output ports.
        By default, it sets image width and height to 32.
        """
        return {
            "indices": NeuralType(tuple('B'), elements_type=Index()),
            "images": NeuralType(
                axes=(
                    AxisType(kind=AxisKind.Batch),
                    AxisType(kind=AxisKind.Channel, size=3),
                    AxisType(kind=AxisKind.Height, size=self._height),
                    AxisType(kind=AxisKind.Width, size=self._width),
                ),
                elements_type=ImageValue(),  # uint8, <0-255>
            ),
            "coarse_targets": NeuralType(tuple('B'), elements_type=ClassificationTarget()),
            "coarse_labels": NeuralType(tuple('B'), elements_type=StringLabel()),  # Labels is string!
            "fine_targets": NeuralType(tuple('B'), elements_type=ClassificationTarget()),
            "fine_labels": NeuralType(tuple('B'), elements_type=StringLabel()),  # Labels is string!
        }

    def __len__(self):
        """
        Returns:
            Length of the dataset.
        """
        return len(self._dataset)

    def __getitem__(self, index: int):
        """
        Returns a single sample.

        Args:
            index: index of the sample to return.

        """
        # Get image and target.
        img, fine_target = self._dataset.__getitem__(index)
        # Get coarse target.
        coarse_target = self._fine_to_coarse_id_mapping[fine_target]

        # Labels.
        fine_label = self._fine_ix_to_word[fine_target]
        coarse_label = self._coarse_ix_to_word[self._fine_to_coarse_id_mapping[fine_target]]

        # Return sample.
        return index, img, coarse_target, coarse_label, fine_target, fine_label

    @property
    def dataset(self):
        """
        Returns:
            Self - just to be "compatible" with the current NeMo train action.
        """
        return self  # ! Important - as we want to use this __getitem__ method!
