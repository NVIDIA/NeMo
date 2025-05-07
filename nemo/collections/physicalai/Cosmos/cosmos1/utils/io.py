# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from io import BytesIO
from typing import Dict, List

import imageio
import numpy as np


def read_prompts_from_file(prompt_file: str) -> List[Dict[str, str]]:
    """Read prompts from a JSONL file where each line is a dict with 'prompt' key and optionally 'visual_input' key.

    Args:
        prompt_file (str): Path to JSONL file containing prompts

    Returns:
        List[Dict[str, str]]: List of prompt dictionaries
    """
    prompts = []
    with open(prompt_file, "r") as f:
        for line in f:
            prompt_dict = json.loads(line.strip())
            prompts.append(prompt_dict)
    return prompts


def save_video(video, fps, H, W, video_save_quality, video_save_path):
    """Save video frames to file.

    Args:
        grid (np.ndarray): Video frames array [T,H,W,C]
        fps (int): Frames per second
        H (int): Frame height
        W (int): Frame width
        video_save_quality (int): Video encoding quality (0-10)
        video_save_path (str): Output video file path
    """
    kwargs = {
        "fps": fps,
        "quality": video_save_quality,
        "macro_block_size": 1,
        "ffmpeg_params": ["-s", f"{W}x{H}"],
        "output_params": ["-f", "mp4"],
    }
    imageio.mimsave(video_save_path, video, "mp4", **kwargs)


def load_from_fileobj(filepath: str, format: str = "mp4", mode: str = "rgb", **kwargs):
    """
    Load video from a file-like object using imageio with specified format and color mode.

    Parameters:
        file (IO[bytes]): A file-like object containing video data.
        format (str): Format of the video file (default 'mp4').
        mode (str): Color mode of the video, 'rgb' or 'gray' (default 'rgb').

    Returns:
        tuple: A tuple containing an array of video frames and metadata about the video.
    """
    with open(filepath, "rb") as f:
        value = f.read()
    with BytesIO(value) as f:
        f.seek(0)
        video_reader = imageio.get_reader(f, format, **kwargs)

        video_frames = []
        for frame in video_reader:
            if mode == "gray":
                import cv2  # Convert frame to grayscale if mode is gray

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = np.expand_dims(frame, axis=2)  # Keep frame dimensions consistent
            video_frames.append(frame)

    return np.array(video_frames), video_reader.get_meta_data()
