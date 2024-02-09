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
import copy
import json
import logging
import os
import re
import tarfile
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from einops import rearrange
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset, default_collate
from transformers import CLIPImageProcessor

import nemo.collections.multimodal.data.neva.conversation as conversation_lib
from nemo.collections.multimodal.data.clip.augmentations.augmentations import image_transform
from nemo.collections.multimodal.data.neva.conversation import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_LABELS_TOKEN,
    DEFAULT_PAD_TOKEN,
    DEFAULT_SEPARATOR_TOKEN,
    DEFAULT_SYSTEM_TOKEN,
    DEFAULT_UNK_TOKEN,
)
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids

DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"

MAX_NUM_IMAGES = 1
MAX_NUM_VIDEOS = 1
IGNORE_INDEX = -1

class TarOrFolderVideoLoader:
    """
    A class for loading images from a tar archive or a regular folder.

    This class provides functionality to open and read images from either a tar archive
    (.tar file) or a standard directory with image files. It builds an index of images
     if the source is a tar archive for efficient access.

    Attributes:
        image_folder (str): The path to the tar archive or video/image folder.
        tar_index (dict): A dictionary that maps file names to their tarfile member
                          objects if the video/image source is a tar archive.

    Methods:
        __init__(self, image_folder): Initializes the loader with the specified video/image folder.
        build_index(self): Builds an index of video/image file names and their corresponding
                           tarfile member objects for a tar archive.
        open_image(self, file_name): Opens and returns an image by its file name. The image
                                     is returned as an RGB PIL Image object.
        open_video(self, file_name): Opens and returns a video by its file name. The video
                                     is returned as an OpenCV VideoCapture object.
    """

    def __init__(self, video_folder):
        self.video_folder = video_folder
        self.tar_index = {}
        if self.video_folder.endswith('.tar'):
            self.build_index()

    def build_index(self):
        with tarfile.open(self.video_folder, 'r') as tar:
            for member in tar.getmembers():
                self.tar_index[member.name] = member

    def open_video(self, file_name):
        if self.video_folder.endswith('.tar'):
            with tarfile.open(self.video_folder, 'r') as tar:
                member = self.tar_index.get(file_name)
                if member:
                    f = tar.extractfile(member)
                    video_data = np.frombuffer(f.read(), dtype=np.uint8)
                    return cv2.imdecode(video_data, cv2.IMREAD_UNCHANGED)
        else:
            return cv2.VideoCapture(os.path.join(self.video_folder, file_name))
        return None

    def flatten_frames(self, cap, num_frames):

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []

        for index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()

            if not ret:
                print(f"Error: Could not read frame at index {index}.")
                break

            frames.append(frame)

        cap.release()
        frames_array = np.array(frames)

        return frames_array

class TarOrFolderImageLoader:
    """
    A class for loading images and videos from a tar archive or a regular folder.

    This class provides functionality to open and read images and videos from either a tar archive
    (.tar file) or a standard directory with image or video files. It builds an index of images and
    videos if the source is a tar archive for efficient access.

    Attributes:
        visual_folder (str): The path to the tar archive or video/image folder.
        tar_index (dict): A dictionary that maps file names to their tarfile member
                          objects if the video/image source is a tar archive.

    Methods:
        __init__(self, image_folder): Initializes the loader with the specified video/image folder.
        build_index(self): Builds an index of video/image file names and their corresponding
                           tarfile member objects for a tar archive.
        open_image(self, file_name): Opens and returns an image by its file name. The image
                                     is returned as an RGB PIL Image object.
        open_video(self, file_name): Opens and returns a video by its file name. The video
                                     is returned as an OpenCV VideoCapture object.
    """

    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.tar_index = {}
        if self.image_folder.endswith('.tar'):
            self.build_index()

    def build_index(self):
        with tarfile.open(self.image_folder, 'r') as tar:
            for member in tar.getmembers():
                self.tar_index[member.name] = member

    def open_image(self, file_name):
        if self.image_folder.endswith('.tar'):
            with tarfile.open(self.image_folder, 'r') as tar:
                member = self.tar_index.get(file_name)
                if member:
                    f = tar.extractfile(member)
                    return Image.open(f).convert('RGB')
        else:
            return Image.open(os.path.join(self.image_folder, file_name)).convert('RGB')
        return None