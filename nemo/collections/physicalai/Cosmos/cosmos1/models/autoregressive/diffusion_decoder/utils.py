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

import torch
import torch.nn.functional as F


def split_with_overlap(video_BCTHW, num_video_frames, overlap=2, tobf16=True):
    """
    Splits the video tensor into chunks of num_video_frames with a specified overlap.

    Args:
    - video_BCTHW (torch.Tensor): Input tensor with shape [Batch, Channels, Time, Height, Width].
    - num_video_frames (int): Number of frames per chunk.
    - overlap (int): Number of overlapping frames between chunks.

    Returns:
    - List of torch.Tensors: List of video chunks with overlap.
    """
    # Get the dimensions of the input tensor
    B, C, T, H, W = video_BCTHW.shape

    # Ensure overlap is less than num_video_frames
    assert overlap < num_video_frames, "Overlap should be less than num_video_frames."

    # List to store the chunks
    chunks = []

    # Step size for the sliding window
    step = num_video_frames - overlap

    # Loop through the time dimension (T) with the sliding window
    for start in range(0, T - overlap, step):
        end = start + num_video_frames
        # Handle the case when the last chunk might go out of bounds
        if end > T:
            # Get the last available frame
            num_padding_frames = end - T
            chunk = F.pad(video_BCTHW[:, :, start:T, :, :], (0, 0, 0, 0, 0, num_padding_frames), mode="reflect")
        else:
            # Regular case: no padding needed
            chunk = video_BCTHW[:, :, start:end, :, :]
        if tobf16:
            chunks.append(chunk.to(torch.bfloat16))
        else:
            chunks.append(chunk)
    return chunks


def linear_blend_video_list(videos, D):
    """
    Linearly blends a list of videos along the time dimension with overlap length D.

    Parameters:
    - videos: list of video tensors, each of shape [b, c, t, h, w]
    - D: int, overlap length

    Returns:
    - output_video: blended video tensor of shape [b, c, L, h, w]
    """
    assert len(videos) >= 2, "At least two videos are required."
    b, c, t, h, w = videos[0].shape
    N = len(videos)

    # Ensure all videos have the same shape
    for video in videos:
        assert video.shape == (b, c, t, h, w), "All videos must have the same shape."

    # Calculate total output length
    L = N * t - D * (N - 1)
    output_video = torch.zeros((b, c, L, h, w), device=videos[0].device)

    output_index = 0  # Current index in the output video

    for i in range(N):
        if i == 0:
            # Copy frames from the first video up to t - D
            output_video[:, :, output_index : output_index + t - D, :, :] = videos[i][:, :, : t - D, :, :]
            output_index += t - D
        else:
            # Blend overlapping frames between videos[i-1] and videos[i]
            blend_weights = torch.linspace(0, 1, steps=D, device=videos[0].device)

            for j in range(D):
                w1 = 1 - blend_weights[j]
                w2 = blend_weights[j]
                frame_from_prev = videos[i - 1][:, :, t - D + j, :, :]
                frame_from_curr = videos[i][:, :, j, :, :]
                output_frame = w1 * frame_from_prev + w2 * frame_from_curr
                output_video[:, :, output_index, :, :] = output_frame
                output_index += 1

            if i < N - 1:
                # Copy non-overlapping frames from current video up to t - D
                frames_to_copy = t - 2 * D
                if frames_to_copy > 0:
                    output_video[:, :, output_index : output_index + frames_to_copy, :, :] = videos[i][
                        :, :, D : t - D, :, :
                    ]
                    output_index += frames_to_copy
            else:
                # For the last video, copy frames from D to t
                frames_to_copy = t - D
                output_video[:, :, output_index : output_index + frames_to_copy, :, :] = videos[i][:, :, D:, :, :]
                output_index += frames_to_copy

    return output_video
