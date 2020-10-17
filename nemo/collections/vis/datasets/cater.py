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


__author__ = "Quang Tran"

from os import makedirs
from os.path import expanduser, join, exists

import json
from PIL import Image
import cv2

from itertools import permutations, product

import torch
from torchvision.transforms import transforms
from torchvision.datasets.utils import download_and_extract_archive, check_md5

from typing import Any, Optional
from dataclasses import dataclass

from hydra.types import ObjectConf
from hydra.core.config_store import ConfigStore

from nemo.utils import logging
from nemo.core.classes import Dataset

from nemo.utils.configuration_parsing import get_value_from_dictionary, get_value_list_from_dictionary
from nemo.utils.configuration_error import ConfigurationError

# Create the config store instance.
cs = ConfigStore.instance()

ATOMIC_ACTION = "atomic_action"
COMPOSITIONAL_ACTION = "compositional_action"
SNITCH_LOCALIZATION = "snitch_localization"

CATER_DATA_NAME = "CATER_v1.0"

@dataclass
class CATERConfig:
    """
    Structured config for the CATER dataset.
    For more details please refer to:
    https://rohitgirdhar.github.io/CATER/
    Args:
        _target_: Specification of dataset class
        root: Folder where task will store data (DEFAULT: "~/data/cater")
        task: Defines the task of the data to load (Options: atomic_action | compositional_action | snitch_localization) (DEFAULT: atomic_action)
        camera_motion: If True, load dataset with camera motions, else load dataset with no camera motions (DEFAULT: False)
        split: Defines the set (split) that will be used (Options: training | validation | test) (DEFAULT: training)
        stream_videos: Flag indicating whether the task will load and return images (DEFAULT: True)
        transform: TorchVision image preprocessing/augmentations to apply (DEFAULT: None)
        download: downloads the data if not present (DEFAULT: True)
    """

    # Dataset target class name.
    _target_: str = "nemo.collections.vis.datasets.CATER"
    root: str = "~/data/cater"
    task: str = ATOMIC_ACTION
    camera_motion: bool = False
    split: str = "training"
    stream_videos: bool = True
    # transform: Optional[Any] = None # Provided manually?
    download: bool = True


# Register the config.
cs.store(
    group="nemo.collections.vis.datasets",
    name="CATER",
    node=ObjectConf(target="nemo.collections.vis.datasets.CATER", params=CATERConfig()),
)


class CATER(Dataset):
    """
    Class fetching data from the CATER (Compositional Actions and TEmporal Reasoning) diagnostics dataset.
    The CATER dataset consists of three tasks:
        
        - Atomic action recognition: each video has multiple actions occuring in pairs. The label of a video is 
          all the actions happened in the video. There 14 possible actions such as slide(cone), rotate(cube)
        - Compositional action recognition: each video has multiple actions occuring in pairs. The label of a 
          video is all the pairs of actions happened in the video. There are 301 possible pairs of action 
          such as 'slide(cone) DURING rotate(cube)', 'pick-place(sphere) AFTER rotate(cylinder)
        - Snitch localization: each video has multiple actions. The label of a video is the quantized position
          of the snitch object on a 6x6 grid at the end of the video. Each video only has one label. There are
          36 possible labels.
    For each task, there are 2 variations: with and without camera motion
    For each task and for each camera motion variation, there are 3 splits: train, val and test
    For more details please refer to the associated _website or _paper.
    .. _website: https://rohitgirdhar.github.io/CATER/
    .._paper: https://arxiv.org/pdf/1910.04744.pdf
    """

    download_url_prefix = "https://cmu.box.com/shared/static/"
    zip_url_and_md5s = {
        (ATOMIC_ACTION, False): {
            "lists": ("7svgta3kqat1jhe9kp0zuptt3vrvarzw.zip", None),
            "videos": ("jgbch9enrcfvxtwkrqsdbitwvuwnopl0.zip", None),
        },
        (ATOMIC_ACTION, True): {
            "lists": ("i9kexj33if00t338esnw93uzm5f6sfar.zip", None),
            "videos": ("yvhx9p5haip5abzh9i2fofssjpq34zwz.zip", None),
        },
        (COMPOSITIONAL_ACTION, False): {
            "lists": ("7svgta3kqat1jhe9kp0zuptt3vrvarzw.zip", None),
            "videos": ("jgbch9enrcfvxtwkrqsdbitwvuwnopl0.zip", None),
        },
        (COMPOSITIONAL_ACTION, True): {
            "lists": ("i9kexj33if00t338esnw93uzm5f6sfar.zip", None),
            "videos": ("yvhx9p5haip5abzh9i2fofssjpq34zwz.zip", None),
        },
        (SNITCH_LOCALIZATION, False): {
            "lists": ("jr2cc6zomqb01hmzvjztpuip3h0sz8vz.zip", None),
            "videos": ("97kztikj3zwcgv5zu2by6c1jhlvh1d7x.zip", None),
        },
        (SNITCH_LOCALIZATION, True): {
            "lists": ("zua3a0afxtuxrkh3jnnkduwkodv816k1.zip", None),
            "videos": ("fevyo9fyzeb6vm418yaqjp3gi7tl9azb.zip", None),
        },
    }

    def __init__(
        self,
        root: str = "~/data/cater",
        task: str = ATOMIC_ACTION,
        camera_motion: bool = False,
        split: str = "training",
        stream_videos: bool = True,
        transform: Optional[Any] = None,
        download: bool = True,
    ):
        """
        Initializes dataset object. Calls base constructor.
        Downloads the dataset if not present and loads the adequate files depending on the mode.
        Args:
            root: Folder where task will store data (DEFAULT: "~/data/cater")
            task: Defines the task of the data to load (Options: atomic_action | compositional_action | snitch_localization) (DEFAULT: atomic_action)
            camera_motion: If True, load dataset with camera motions, else load dataset with no camera motions (DEFAULT: False)
            split: Defines the set (split) that will be used (Options: training | validation | test | cogent_a_training | cogent_a_validation | cogent_b_validation) (DEFAULT: training)
            stream_videos: Flag indicating whether the task will load and return videos (DEFAULT: True)
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
            "training | validation | test".split(" | "),
        )

        self._task = get_value_from_dictionary(
            task,
            [ATOMIC_ACTION, COMPOSITIONAL_ACTION, SNITCH_LOCALIZATION],
        )

        self._camera_motion = camera_motion

        # Get flag informing whether we want to stream images or not.
        self._stream_videos = stream_videos

        # Set original image dimensions.
        self._height = 240
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

        logging.info("Setting image size to [D x H x W]: {} x {} x {}".format(self._depth, self._height, self._width))
        ACTION_CLASSES = [
            # object, movement
            ('sphere', '_slide'),
            ('sphere', '_pick_place'),
            ('spl', '_slide'),
            ('spl', '_pick_place'),
            ('spl', '_rotate'),
            ('cylinder', '_pick_place'),
            ('cylinder', '_slide'),
            ('cylinder', '_rotate'),
            ('cube', '_slide'),
            ('cube', '_pick_place'),
            ('cube', '_rotate'),
            ('cone', '_contain'),
            ('cone', '_pick_place'),
            ('cone', '_slide'),
        ]
        _BEFORE = 'before'
        _AFTER = 'after'
        _DURING = 'during'
        
        ORDERING = [
            _BEFORE,
            _DURING,
            _AFTER,
        ]
        
        if task == ATOMIC_ACTION:
            classes = ACTION_CLASSES

        elif task == COMPOSITIONAL_ACTION:
            num_actions = 2
            action_sets = list(product(ACTION_CLASSES, repeat=num_actions))
            # all orderings
            orderings = list(product(ORDERING, repeat=(num_actions-1)))
            # all actions and orderings
            classes = list(product(action_sets, orderings))

            # Remove classes such as "X before Y" when "Y after X" already exists in the data
            classes = self._action_order_unique(classes)
            print('Action orders classes {}'.format(len(classes)))
        
        elif task == SNITCH_LOCALIZATION:
            classes = []
            for y in range(6):
                for x in range(6):
                    classes.append((x,y))
        else:
            raise ConfigurationError("Task `{}` not supported yet".format(task))

        self.class_id_to_class_mapping = classes

        self._task_to_data_subdir_mapping = {
            ATOMIC_ACTION: join("lists", "actions_present"),
            COMPOSITIONAL_ACTION: join("lists", "actions_order_uniq"),
            SNITCH_LOCALIZATION: join("lists", "localize"),
        }
        self._task_and_camera_motion_to_data_dir_mapping = {
            (ATOMIC_ACTION, False): "max2action",
            (ATOMIC_ACTION, True):  "max2action_cameramotion",
            (COMPOSITIONAL_ACTION, False): "max2action",
            (COMPOSITIONAL_ACTION, True): "max2action_cameramotion",
            (SNITCH_LOCALIZATION, False): "all_actions",
            (SNITCH_LOCALIZATION, True): "all_actions_cameramotion",
        } 

        self._split_to_data_file_mapping = {
            "training": "train_subsetT.txt",
            "validation": "train_subsetV.txt",
            "test": "val.txt",
        }

        self.split_folder = join(self._root, CATER_DATA_NAME, 
                                 self._task_and_camera_motion_to_data_dir_mapping[(task, camera_motion)])
        self.split_video_folder = join(self.split_folder, "videos")
        self.split_annotation_folder = join(self.split_folder, "lists")
        data_file = join(self.split_folder,
                         self._task_to_data_subdir_mapping[task],
                         self._split_to_data_file_mapping[split])

        # Download dataset when required.
        if download:
            self.download()

        # Load data from file.
        self.data = self.load_data(data_file)

        # Display exemplary sample.
        i = 0
        sample = self.data[i]

        logging.info(
            "Exemplary sample {}:\n  class_id: {} ({})\n  video_id: {}\n".format(
                i,
                sample["class_id"],
                sample["class"],
                sample["video_filename"],
            )
        )
    
    def _action_order_unique(self, classes):
        def reverse(el):
            if el == ('during',):
                return el
            elif el == ('before',):
                return ('after',)
            elif el == ('after',):
                return ('before',)
            else:
                raise ValueError('This should not happen')
        classes_uniq = []
        for el in classes:
            if el not in classes_uniq and ((el[0][1], el[0][0]), reverse(el[1])) not in classes_uniq:
                classes_uniq.append(el)
        return classes_uniq

    def _check_exist(self) -> bool:
        if not (exists(self.split_video_folder) and exists(self.split_annotation_folder)):
            # Make dir - just in case.
            makedirs(self.split_folder, exist_ok=True)
            return False

        logging.info('Files already downloaded')
        return True

    def download(self) -> None:
        if self._check_exist():
            return
        # Else: download (once again).
        logging.info('Downloading and extracting archive')

        video_url, video_md5 = self.zip_url_and_md5s[(self._task, self._camera_motion)]["videos"]
        annotation_url, annotation_md5 = self.zip_url_and_md5s[(self._task, self._camera_motion)]["lists"]
        video_url = self.download_url_prefix + video_url
        annotation_url = self.download_url_prefix + annotation_url

        # Download and extract the required file.
        download_and_extract_archive(video_url, self.split_folder, filename="videos.zip", md5=video_md5)
        download_and_extract_archive(annotation_url, self.split_folder, filename="lists.zip", md5=video_md5)


    def load_data(self, source_data_file):
        """
        Loads the dataset from source file.
        Args:
            source_data_file: txt file with lines of the format <video_id> <labels>. For example, CATER_new_009.avi 2,8,9,10,11,12,13
        """
        dataset = []

        with open(source_data_file) as f:
            logging.info("Loading samples from '{}'...".format(source_data_file))
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0: continue
                video_filename, labels = line.split()
                labels = [int(x) for x in labels.split(",")]
                classes = [self.class_id_to_class_mapping[x] for x in labels]
                elem = {"class": classes, "class_id": labels, "video_filename": video_filename}
                dataset.append(elem)

        logging.info("Loaded dataset consisting of {} samples".format(len(dataset)))
        return dataset

    def __len__(self):
        """
        Returns:
            The size of the loaded dataset split.
        """
        return len(self.data)

    # finished
    def get_frames(self, vid_id):
        """
        Function loads and returns video
        Additionally, it performs all the required transformations.
        Args:
            vid_id: Identifier of the video.
        Returns:
            list of images (PIL Image / Tensor, depending on the applied transforms)
        """
        # Read the video from specified path
        vid_path = join(self.split_video_folder, vid_id)
        cam = cv2.VideoCapture(vid_path)

        # frame
        frames = []  
        while(True):       
            # reading from frame 
            ret,frame = cam.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                if self._image_transform is not None:
                    # Apply transformation(s).
                    img = self._image_transform(img)

                frames.append(img)
            else: 
                break
        # Release all space and windows once done 
        cam.release()
        cv2.destroyAllWindows()
        # Return image.
        return frames

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a single sample.
        Args:
            index: index of the sample to return.
        Returns:
            index, video_id, video, class_ids, classes
        """
        # Get item.
        item = self.data[index]

        # Load and stream the image ids.
        video_id = item["video_filename"]

        # Load the adequate image - only when required.
        if self._stream_videos:
            video = self.get_frames(video_id)
        else:
            video = None

        classes = item["class"]
        class_ids = item["class_id"]

        # Return sample.
        return index, video_id, video, class_ids, classes

    def collate_fn(self, batch):
        """
        Combines a list of samples (retrieved with :py:func:`__getitem__`) into a batch.
        Args:
            batch: list of individual samples to combine
        Returns:
            Batch of: indices, video_ids, videos, class_ids_batch, classes_batch
        """
        # Collate indices.
        indices = [sample[0] for sample in batch]

        # Stack videos.
        video_ids = [sample[1] for sample in batch]

        if self._stream_videos:
            videos = [sample[2] for sample in batch]
        else:
            videos = None

        # Collate lists/lists of lists.
        classes_batch = [sample[4] for sample in batch]
        class_ids_batch = [sample[3] for sample in batch]

        # Return collated dict.
        return indices, video_ids, videos, class_ids_batch, classes_batch