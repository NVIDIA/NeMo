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

__author__ = "Tomasz Kornuta"

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/IBM/pytorchpipe/blob/develop/ptp/components/tasks/image_text_to_class/clevr.py
"""

from os import makedirs
from os.path import expanduser, join, exists

import json
from PIL import Image

import torch
from torchvision.transforms import transforms
from torchvision.datasets.utils import download_and_extract_archive, check_md5

from typing import Any, Optional
from dataclasses import dataclass

from hydra.types import ObjectConf
from hydra.core.config_store import ConfigStore

from nemo.utils import logging
from nemo.core.classes import Dataset

# from nemo.core.neural_types import CategoricalValuesType, ChannelType, MaskType, NeuralType, RegressionValuesType

from nemo.utils.configuration_parsing import get_value_from_dictionary, get_value_list_from_dictionary
from nemo.utils.configuration_error import ConfigurationError

# Create the config store instance.
cs = ConfigStore.instance()


@dataclass
class CLEVRConfig:
    """
    Structured config for the CLEVR dataset.

    For more details please refer to:
    https://cs.stanford.edu/people/jcjohns/clevr/

    Args:
        _target_: Specification of dataset class
        root: Folder where task will store data (DEFAULT: "~/data/clevr")
        split: Defines the set (split) that will be used (Options: training | validation | test | cogent_a_training | cogent_a_validation | cogent_b_validation) (DEFAULT: training)
        stream_images: Flag indicating whether the task will load and return images (DEFAULT: True)
        transform: TorchVision image preprocessing/augmentations to apply (DEFAULT: None)
        download: downloads the data if not present (DEFAULT: True)
    """

    # Dataset target class name.
    _target_: str = "nemo.collections.vis.datasets.CLEVR"
    root: str = "~/data/clevr"
    split: str = "training"
    stream_images: bool = True
    # transform: Optional[Any] = None # Provided manually?
    download: bool = True


# Register the config.
cs.store(
    group="nemo.collections.vis.datasets",
    name="CLEVR",
    node=ObjectConf(target="nemo.collections.vis.datasets.CLEVR", params=CLEVRConfig()),
)


class CLEVR(Dataset):
    """
    Class fetching data from the CLEVR (Compositional Language andElementary Visual Reasoning) diagnostics dataset.

    The CLEVR dataset consists of three splits:
        - A training set of 70,000 images and 699,989 questions
        - A validation set of 15,000 images and 149,991 questions
        - A test set of 15,000 images and 14,988 questions
        - Answers for all train and val questions
        - Scene graph annotations for train and val images giving ground-truth locations, attributes, and relationships for objects
        - Functional program representations for all training and validation images

    Additionally, class handles the Compositional Generalization Test (CoGenT) "Condition A" and "Condition B" variants.

    CLEVR contains a total of 90 question families, eachwith a single program template and an average of four texttemplates.
    Those are further aggregated into 13 Question Types:
        - Querying attributes (Size, Color, Material, Shape)
        - Comparing attributes (Size, Color, Material, Shape)
        - Existence
        - Counting
        - Integer comparison (Equal, Less, More)

    For more details please refer to the associated _website or _paper.

    .. _website: https://cs.stanford.edu/people/jcjohns/clevr/

    .._paper: https://arxiv.org/pdf/1612.06890

    """

    download_url_prefix = "https://dl.fbaipublicfiles.com/clevr/"
    zip_names = {"clevr": "CLEVR_v1.0.zip", "cogent": "CLEVR_CoGenT_v1.0.zip"}
    zip_md5s = {"clevr": "b11922020e72d0cd9154779b2d3d07d2", "cogent": "9e4a361ab939a4899e6d2ac14a5ee434"}

    def __init__(
        self,
        root: str = "~/data/clevr",
        split: str = "training",
        stream_images: bool = True,
        transform: Optional[Any] = None,
        download: bool = True,
    ):
        """
        Initializes dataset object. Calls base constructor.
        Downloads the dataset if not present and loads the adequate files depending on the mode.

        Args:
        root: Folder where task will store data (DEFAULT: "~/data/clevr")
            split: Defines the set (split) that will be used (Options: training | validation | test | cogent_a_training | cogent_a_validation | cogent_b_validation) (DEFAULT: training)
            stream_images: Flag indicating whether the task will load and return images (DEFAULT: True)
            transform: TorchVision image preprocessing/augmentations to apply (DEFAULT: None)
            download: downloads the data if not present (DEFAULT: True)
        """
        # Call constructors of parent class.
        super().__init__()

        # Get the absolute path.
        self._root = expanduser(root)

        # Process split.
        self._split = get_value_from_dictionary(
            split,
            "training | validation | test | cogent_a_training | cogent_a_validation | cogent_b_validation".split(
                " | "
            ),
        )

        # Download dataset when required.
        if download:
            self.download()

        # Get flag informing whether we want to stream images or not.
        self._stream_images = stream_images

        # Set original image dimensions.
        self._height = 480
        self._width = 320
        self._depth = 3

        # Save image transform(s).
        self._image_transform = transform

        # Check presence of Resize transform.
        if self._image_transform is not None:
            resize = None
            # Check single transform.
            if isinstance(self._image_transform, transforms.Resize):
                resize = self._image_transform
            # Check transform composition.
            elif isinstance(self._image_transform, transforms.Compose):
                # Iterate throught transforms.
                for trans in self._image_transform.transforms:
                    if isinstance(trans, transforms.Resize):
                        resize = trans
            # Update the image dimensions [H,W].
            if resize is not None:
                self._height = resize.size[0]
                self._width = resize.size[1]

        logging.info("Setting image size to [D  x H x W]: {} x {} x {}".format(self._depth, self._height, self._width))

        # Mapping of question subtypes to types (not used, but keeping it just in case).
        # self._question_subtype_to_type_mapping = {
        #    'query_size': 'query_attribute',
        #    'equal_size': 'compare_attribute',
        #    'query_shape': 'query_attribute',
        #    'query_color': 'query_attribute',
        #    'greater_than': 'compare_integer',
        #    'equal_material': 'compare_attribute',
        #    'equal_color': 'compare_attribute',
        #    'equal_shape': 'compare_attribute',
        #    'less_than': 'compare_integer',
        #    'count': 'count',
        #    'exist': 'exist',
        #    'equal_integer': 'compare_integer',
        #    'query_material': 'query_attribute'}

        # Mapping of question subtypes to types.
        self._question_subtype_to_id_mapping = {
            'query_size': 0,
            'equal_size': 1,
            'query_shape': 2,
            'query_color': 3,
            'greater_than': 4,
            'equal_material': 5,
            'equal_color': 6,
            'equal_shape': 7,
            'less_than': 8,
            'count': 9,
            'exist': 10,
            'equal_integer': 11,
            'query_material': 12,
        }

        # Mapping of question families to subtypes.
        self._question_family_id_to_subtype_mapping = {
            0: "equal_integer",
            1: "less_than",
            2: "greater_than",
            3: "equal_integer",
            4: "less_than",
            5: "greater_than",
            6: "equal_integer",
            7: "less_than",
            8: "greater_than",
            9: "equal_size",
            10: "equal_color",
            11: "equal_material",
            12: "equal_shape",
            13: "equal_size",
            14: "equal_size",
            15: "equal_size",
            16: "equal_color",
            17: "equal_color",
            18: "equal_color",
            19: "equal_material",
            20: "equal_material",
            21: "equal_material",
            22: "equal_shape",
            23: "equal_shape",
            24: "equal_shape",
            25: "count",
            26: "exist",
            27: "query_size",
            28: "query_shape",
            29: "query_color",
            30: "query_material",
            31: "count",
            32: "query_size",
            33: "query_color",
            34: "query_material",
            35: "query_shape",
            36: "exist",
            37: "exist",
            38: "exist",
            39: "exist",
            40: "count",
            41: "count",
            42: "count",
            43: "count",
            44: "exist",
            45: "exist",
            46: "exist",
            47: "exist",
            48: "count",
            49: "count",
            50: "count",
            51: "count",
            52: "query_color",
            53: "query_material",
            54: "query_shape",
            55: "query_size",
            56: "query_material",
            57: "query_shape",
            58: "query_size",
            59: "query_color",
            60: "query_shape",
            61: "query_size",
            62: "query_color",
            63: "query_material",
            64: "count",
            65: "count",
            66: "count",
            67: "count",
            68: "count",
            69: "count",
            70: "count",
            71: "count",
            72: "count",
            73: "exist",
            74: "query_size",
            75: "query_color",
            76: "query_material",
            77: "query_shape",
            78: "count",
            79: "exist",
            80: "query_size",
            81: "query_color",
            82: "query_material",
            83: "query_shape",
            84: "count",
            85: "exist",
            86: "query_shape",
            87: "query_material",
            88: "query_color",
            89: "query_size",
        }

        # Finally, "merge" those two.
        self._question_family_id_to_subtype_id_mapping = {
            key: self._question_subtype_to_id_mapping[value]
            for key, value in self._question_family_id_to_subtype_mapping.items()
        }

        # Set split-dependent data.
        if self._split == 'training':
            # Training split folder and file with data question.
            data_file = join(self._root, "CLEVR_v1.0", "questions", 'CLEVR_train_questions.json')
            self.split_image_folder = join(self._root, "images", "train")

        elif self._split == 'validation':
            # Validation split folder and file with data question.
            data_file = join(self._root, "CLEVR_v1.0", "questions", 'CLEVR_val_questions.json')
            self.split_image_folder = join(self._root, "images", "val")

        elif self._split == 'test':
            # Test split folder and file with data question.
            data_file = join(self._root, "CLEVR_v1.0", "questions", 'CLEVR_test_questions.json')
            self.split_image_folder = join(self._root, "images", "test")

        else:  # cogent
            raise ConfigurationError("Split `{}` not supported yet".format(self._split))

        # Load data from file.
        self.data = self.load_data(data_file)

        # Display exemplary sample.
        i = 0
        sample = self.data[i]
        # Check if this is a test set.
        if "answer" not in sample.keys():
            sample["answer"] = "<UNK>"
            sample["question_type_ids"] = -1
            sample["question_type_names"] = "<UNK>"
        else:
            sample["question_type_ids"] = self._question_family_id_to_subtype_id_mapping[
                sample["question_family_index"]
            ]
            sample["question_type_names"] = self._question_family_id_to_subtype_mapping[
                sample["question_family_index"]
            ]

        logging.info(
            "Exemplary sample {} ({}):\n  question_type: {} ({})\n  image_ids: {}\n  question: {}\n  answer: {}".format(
                i,
                sample["question_index"],
                sample["question_type_ids"],
                sample["question_type_names"],
                sample["image_filename"],
                sample["question"],
                sample["answer"],
            )
        )

    def _check_integrity(self) -> bool:
        if "cogent" in self._split:
            # Focus on CoGen-T variant.
            filename = join(self._root, self.zip_names["cogent"])
            md5sum = self.zip_md5s["cogent"]
        else:
            # Focus on CLEVR variant.
            filename = join(self._root, self.zip_names["clevr"])
            md5sum = self.zip_md5s["clevr"]

        if not exists(filename):
            # Make dir - just in case.
            makedirs(self._root, exist_ok=True)
            return False

        logging.info('Files already downloaded, checking integrity...')
        # Check md5 and return the result.
        return check_md5(fpath=filename, md5=md5sum)

    def download(self) -> None:
        if self._check_integrity():
            logging.info('Files verified')
            return
        # Else: download (once again).
        logging.info('Downloading and extracting archive')

        if "cogent" in self._split:
            # Focus on CoGen-T variant.
            filename = join(self._root, self.zip_names["cogent"])
            url = self.download_url_prefix + self.zip_names["cogent"]
            md5sum = self.zip_md5s["cogent"]
        else:
            # Focus on CLEVR variant.
            filename = join(self._root, self.zip_names["clevr"])
            url = self.download_url_prefix + self.zip_names["clevr"]
            md5sum = self.zip_md5s["clevr"]

        # Download and extract the required file.
        download_and_extract_archive(url, self._root, filename=filename, md5=md5sum)

    def load_data(self, source_data_file):
        """
        Loads the dataset from source file.

        Args:
            source_data_file: jSON file with image ids, questions, answers, scene graphs, etc.

        """
        dataset = []

        with open(source_data_file) as f:
            logging.info("Loading samples from '{}'...".format(source_data_file))
            dataset = json.load(f)['questions']

        logging.info("Loaded dataset consisting of {} samples".format(len(dataset)))
        return dataset

    def __len__(self):
        """
        Returns:
            The size of the loaded dataset split.
        """
        return len(self.data)

    def get_image(self, img_id):
        """
        Function loads and returns image along with its size.
        Additionally, it performs all the required transformations.

        Args:
            img_id: Identifier of the images.

        Returns:
            image (PIL Image / Tensor, depending on the applied transforms)
        """

        # Load the image and convert to RGB.
        img = Image.open(join(self.split_image_folder, img_id)).convert('RGB')

        if self._image_transform is not None:
            # Apply transformation(s).
            img = self._image_transform(img)

        # Return image.
        return img

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a single sample.

        Args:
            index: index of the sample to return.

        Returns:
            indices, images, images_ids,questions, answers, question_type_ids, question_type_names
        """
        # Get item.
        item = self.data[index]

        # Load and stream the image ids.
        img_id = item["image_filename"]

        # Load the adequate image - only when required.
        if self._stream_images:
            img = self.get_image(img_id)
        else:
            img = None

        # Return question.
        question = item["question"]

        # Return answer.
        if "answer" in item.keys():
            answer = item["answer"]
        else:
            answer = "<UNK>"

        # Question type related variables.
        if "question_family_index" in item.keys():
            question_type_id = self._question_family_id_to_subtype_id_mapping[item["question_family_index"]]
            question_type_name = self._question_family_id_to_subtype_mapping[item["question_family_index"]]
        else:
            question_type_id = -1
            question_type_name = "<UNK>"

        # Return sample.
        return index, img_id, img, question, answer, question_type_id, question_type_name

    def collate_fn(self, batch):
        """
        Combines a list of samples (retrieved with :py:func:`__getitem__`) into a batch.

        Args:
            batch: list of individual samples to combine

        Returns:
            Batch of: indices, images, images_ids,questions, answers, category_ids, image_sizes

        """
        # Collate indices.
        indices = [sample[0] for sample in batch]

        # Stack images.
        img_ids = [sample[1] for sample in batch]

        if self._stream_images:
            imgs = torch.stack([sample[2] for sample in batch]).type(torch.FloatTensor)
        else:
            imgs = None

        # Collate lists/lists of lists.
        questions = [sample[3] for sample in batch]
        answers = [sample[4] for sample in batch]

        # Stack categories.
        question_type_ids = torch.tensor([sample[5] for sample in batch])
        question_type_names = [sample[6] for sample in batch]

        # Return collated dict.
        return indices, img_ids, imgs, questions, answers, question_type_ids, question_type_names
