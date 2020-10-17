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

__author__ = "Anh Tuan Nguyen"


import json
from dataclasses import dataclass
from glob import glob
from os import makedirs
from os.path import exists, expanduser, join
from typing import Any, Optional

import cv2
import numpy as np
from hydra.core.config_store import ConfigStore
from hydra.types import ObjectConf
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive, download_url
from torchvision.transforms import transforms

from nemo.core.classes import Dataset
from nemo.utils import logging

# from nemo.core.neural_types import CategoricalValuesType, ChannelType, MaskType, NeuralType, RegressionValuesType

# Create the config store instance.
cs = ConfigStore.instance()


@dataclass
class CLEVRERConfig:
    """
    Structured config for the CLEVRER dataset.

    For more details please refer to:
    http://clevrer.csail.mit.edu/

    Args:
        _target_: Specification of dataset class
        root: Folder where task will store data (DEFAULT: "~/data/clevrer")
        split: Defines the set (split) that will be used (Options: train | val | test ) (DEFAULT: train)
        stream_frames: Flag indicating whether the task will load and return frames in the video (DEFAULT: True)
        transform: TorchVision image preprocessing/augmentations to apply (DEFAULT: None)
        download: downloads the data if not present (DEFAULT: True)
    """

    # Dataset target class name.
    _target_: str = "nemo.collections.vis.datasets.CLEVRER"
    root: str = "~/data/clevrer"
    split: str = "train"
    stream_frames: bool = True
    # transform: Optional[Any] = None # Provided manually?
    download: bool = True


# Register the config.
cs.store(
    group="nemo.collections.vis.datasets",
    name="CLEVRER",
    node=ObjectConf(target="nemo.collections.vis.datasets.CLEVRER", params=CLEVRERConfig()),
)


class CLEVRER(Dataset):
    """
    Class fetching data from the CLEVRER (Video Question Answering for Temporal and Causal Reasoning) dataset.

    The CLEVRER dataset consists of the followings:

        - 20,000 videos, separated into train (index 0 - 9999), validation (index 10000 - 14999), and test (index 15000 - 19999) splits.
        - Questions which are categorized into descriptives, explanatory, predictive and counterfactual
        - Annotation files which contain object properties, motion trajectories and collision events

    For more details please refer to the associated _website or _paper.

    After downloading and extracting, we will have the following directory

    data/clevrer
    videos/
    # Train
        video_00000-01000
        ...
        video_09000-10000
    # Validation
        video_10000-11000
        ...
        video_14000-15000
    # Test
        video_15000-16000
        ...
        video_19000-20000
    
    annotations/
    # Train 
            annotation_00000-01000
            ...
            annotation_09000-10000
    # Validation
            annotation_11000-12000
            ...
            annotation_14000-15000
    question/
        train.json
        validation.json
        test.json

    video_frames/
        sim_00000/ frame_00000.jpg, frame_00001.jpg, ...
        sim_00001/ frame_00000.jpg, frame_00001.jpg, ...
        ...
        sim_19999/ frame_00000.jpg, frame_00001.jpg, ...

    .. _website: http://clevrer.csail.mit.edu/

    .._paper: https://arxiv.org/pdf/1910.01442

    """

    download_url_prefix_videos = "http://data.csail.mit.edu/clevrer/videos/"
    download_url_prefix_annotations = "http://data.csail.mit.edu/clevrer/annotations/"
    download_url_prefix_questions = "http://data.csail.mit.edu/clevrer/questions/"

    videos_names = {"train": "video_train.zip", "validation": "video_validation.zip", "test": "video_test.zip"}
    annotations_names = {"train": "annotation_train.zip", "validation": "annotation_validation.zip"}
    question_names = {"train": "train.json", "validation": "validation.json", "test": "test.json"}

    def __init__(
        self,
        root: str = "~/data/clevrer",
        split: str = "train",
        stream_frames: bool = True,
        save_frames: bool = False,
        transform: Optional[Any] = None,
        download: bool = True,
    ):
        """
        Initializes dataset object. Calls base constructor.
        Downloads the dataset if not present and loads the adequate files depending on the mode.

        Args:
        root: Folder where task will store data (DEFAULT: "~/data/clevrer")
            split: Defines the set (split) that will be used (Options: train | val | test) (DEFAULT: train)
            stream_frames: Flag indicating whether the task will return frames from the video (DEFAULT: True)
            save_frames: Flag indicating whether user want to save frames into disk or not (DEFAULT: False)
            transform: TorchVision image preprocessing/augmentations to apply (DEFAULT: None)
            download: downloads the data if not present (DEFAULT: True)
        """

        # Call constructors of parent class.
        super().__init__()

        # Get the absolute path.
        self._root = expanduser(root)

        # if don't have the root, create it
        if not exists(self._root):
            makedirs(self._root)

        # Process split.
        self._split = split
        self._save_frames = save_frames

        # Download dataset when required.
        if download:
            self.download()

        # Get flag informing whether we want to stream frames back to user or not.
        self._stream_frames = stream_frames

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

        if self._split == 'train':
            data_file = join(self._root, "question", 'train.json')
        elif self._split == 'validation':
            data_file = join(self._root, "question", 'validation.json')
        elif self._split == 'test':
            data_file = join(self._root, "question", 'test.json')
        else:
            raise ValueError("Split `{}` not supported yet".format(self._split))

        # Video frames folder
        self._video_frames = join(self._root, "video_frames")

        if not exists(self._video_frames):
            makedirs(self._video_frames)

        # Load data from file.
        self.data = self.load_data(data_file)

        # Display exemplary sample.
        i = 0
        sample = self.data[i]
        # Check if this is a test set.
        if "answer" not in sample.keys():
            sample["answer"] = "<UNK>"
        logging.info(
            "Exemplary sample number: {}\n  question_type: {}\n  question_subtype: {}\n  question_id: {}\n question: {}\n  answer: {}".format(
                i,
                sample["question_type"],
                sample["question_subtype"],
                sample["question_id"],
                sample["question"],
                sample["answer"],
            )
        )

    def _check_exist(self) -> bool:
        # Check md5 and return the result.
        split = self._split
        # Check video files
        videofile = join(self._root, "videos", self.videos_names[split])
        if not exists(videofile):
            logging.info("Cannot find video files")
            return False

        if split == "train" or split == "validation":
            # Check annotations files
            annotationfile = join(self._root, "annotations", self.annotations_names[split])
            if not exists(annotationfile):
                logging.info("Cannot find annotation files")
                return False
        logging.info('Files already exists, do not need to re-download...')
        return True

    def download(self) -> None:
        if self._check_exist():
            return
        # Else: download (once again).
        logging.info('Downloading and extracting archive')

        split = self._split
        # Download video files
        videofile = self.videos_names[split]
        videourl = self.download_url_prefix_videos + split + '/' + self.videos_names[split]
        videodir = join(self._root, "videos")
        if not exists(videodir):
            makedirs(videodir)
        download_and_extract_archive(videourl, download_root=videodir, filename=videofile)

        # Download questions files
        questionfile = self.question_names[split]
        questionurl = self.download_url_prefix_questions + self.question_names[split]
        questiondir = join(self._root, "question")
        if not exists(questiondir):
            makedirs(questiondir)
        download_url(questionurl, root=questiondir, filename=questionfile)

        if split == "train" or split == "validation":
            # Download annotation files
            annotationfile = self.annotations_names[split]
            annotationurl = self.download_url_prefix_annotations + split + '/' + self.annotations_names[split]
            annotationdir = join(self._root, "annotations")
            if not exists(annotationdir):
                makedirs(annotationdir)
            download_and_extract_archive(annotationurl, download_root=annotationdir, filename=annotationfile)

    def load_data(self, source_data_file):
        """
        Loads the dataset from source file.

        """
        dataset = []

        with open(source_data_file) as f:
            logging.info("Loading samples from '{}'...".format(source_data_file))
            data = json.load(f)
            for questions in data:
                for question in questions['questions']:
                    question_data = question
                    question_data['scene_index'] = questions['scene_index']
                    question_data['video_filename'] = questions['video_filename']
                    dataset.append(question_data)
        logging.info("Loaded dataset consisting of {} samples".format(len(dataset)))
        return dataset

    def __len__(self):
        """
        Returns:
            The size of the loaded dataset split.
        """
        return len(self.data)

    def get_frames(self, video_index):
        """
        Function loads and returns video frames along with its size.
        Additionally, it performs all the required transformations.

        Args:
            video_index: Identifier of the video.

        Returns:
            List of video frames (PIL Image / Tensor, depending on the applied transforms)
        """

        # Directory to save the frames
        frame_dir = join(self._video_frames, "sim_%05d" % video_index)

        # Check which directory our video belong to
        prev_endpoint = 0
        for idx, end_point in enumerate(np.arange(0, 21000, 1000)):
            if idx > 0:
                if video_index in range(prev_endpoint, end_point):
                    video_dir = join(self._root, "videos", "video_%05d" % prev_endpoint + "-" + "%05d" % end_point)
                    video_file = join(video_dir, "video_%05d.mp4" % video_index)
                    break
                prev_endpoint = end_point

        frames = []

        # Extract frame once, reused later
        if not exists(frame_dir):
            makedirs(frame_dir)
        
        frame_extractor = cv2.VideoCapture(video_file)
        currentframe = 0
        while True:
            # extract frames from videos
            ret, frame = frame_extractor.read()
            if ret:
                # if we want to save frames
                if self._save_frames:
                    filename = 'frame_%05d.jpg' % currentframe
                    frame_file = join(frame_dir, filename)
                    if not exists(frame_file):
                        logging.info('Creating frame file: ' + filename)
                        # store frame into file
                        cv2.imwrite(frame_file, frame)
                    
                # return frames
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                if self._image_transform is not None:
                    # Apply transformation(s).
                    img = self._image_transform(img)

                frames.append(img)
                currentframe += 1
            else:
                logging.info('Finish extracting frame from the video')
                break
        # Release all space and windows once done 
        frame_extractor.release()
        cv2.destroyAllWindows()
            
        # Return frame list.
        return frames

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a single sample.

        Args:
            index: index of the sample to return.

        Returns:
            indices, video_id, frames, question_id, questions, answers, question_types, question_subtypes
        """
        # Get item.
        item = self.data[index]

        # Load and stream the image ids.
        video_index = item["scene_index"]

        # Load the adequate image - only when required.
        if self._stream_frames:
            frames = self.get_frames(video_index)
        else:
            frames = None

        # Return question.
        question_id = item["question_id"]
        question = item["question"]

        # Return answer.
        if "answer" in item.keys():
            answer = item["answer"]
        else:
            answer = "<UNK>"

        # Question type and subtype
        question_type = item["question_type"]

        if "question_subtype" in item.keys():
            question_subtype = item["question_subtype"]
        else:
            question_subtype = "<UNK>"


        # Return sample.
        return index, video_index, frames, question_id, question, answer, question_type, question_subtype

    def collate_fn(self, batch):
        """
        Combines a list of samples (retrieved with :py:func:`__getitem__`) into a batch.

        Args:
            batch: list of individual samples to combine

        Returns:
            Batch of: indices, video_id, frames, question_id, questions, answers, question_types, question_subtypes

        """
        # Collate indices.
        indices_batch = [sample[0] for sample in batch]

        # Stack video_ids and list of frames.
        video_ids_batch = [sample[1] for sample in batch]

        # stack list of frames
        if self._stream_frames:
            frames_batch = [sample[2] for sample in batch]
        else:
            frames_batch = None

        # Collate questions and answers
        question_id_batch = [sample[3] for sample in batch]
        questions_batch = [sample[4] for sample in batch]
        answers_batch = [sample[5] for sample in batch]

        # Collate question_types
        question_type_batch = [sample[6] for sample in batch]
        question_subtype_batch = [sample[7] for sample in batch]

        # Return collated dict.
        return (
            indices_batch,
            video_ids_batch,
            frames_batch,
            question_id_batch,
            questions_batch,
            answers_batch,
            question_type_batch,
            question_subtype_batch,
        )