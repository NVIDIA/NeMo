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

# pylint: disable=C0115,C0116,C0301

"""
Combined script to process videos in two directories (reference and generated)
and then compute video quality metrics: PSNR and SSIM.

Each video is clipped to 10 seconds (or at least 300 frames) and resized using the specified
number of frames (--num_frames) and width (--width), with the processed output written to a
"processed" subfolder in the original directory.

Example usage:
    pip install scikit-image imageio mediapy
    python cosmos1/models/autoregressive/nemo/post_training/tokenizer/token_bench.py --gtpath /path/to/ref_videos --targetpath /path/to/gen_videos --width 320 --num_frames 17 --recursive
"""

import argparse
import json
import os
from glob import glob

import imageio

# Keep mediapy for functions like resize_video and write_video
import mediapy as media
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


# --------------------------------------------------
# New Video Reader Using imageio (Replaces media.read_video)
# --------------------------------------------------
def read_video(filepath: str) -> np.ndarray:
    """Reads a video file using imageio and returns a NumPy array of frames."""
    try:
        reader = imageio.get_reader(filepath, "ffmpeg")
    except Exception as e:
        raise RuntimeError(f"Error reading video file {filepath}: {e}")
    frames = []
    for frame in reader:
        frames.append(frame)
    return np.array(frames)


# ----------------------------
# Video Processing Functions
# ----------------------------
def resize_video(video: np.ndarray, num_frames: int = 17, short_size: int = None) -> np.ndarray:
    """Resizes a video so that its shorter side is equal to `short_size` and clips to `num_frames` frames."""
    if short_size is None:
        return video[:num_frames]
    height, width = video.shape[-3:-1]
    if height <= width:
        height_new, width_new = short_size, int(width * short_size / height + 0.5)
    else:
        height_new, width_new = int(height * short_size / width + 0.5), short_size
    return media.resize_video(video[:num_frames], shape=(height_new, width_new))


def process_video_file(video_file: str, output_dir: str, num_frames: int, width: int) -> None:
    """Process a single video file: clip to 10 seconds (or at least 300 frames) and resize using provided parameters."""
    try:
        video_reader = imageio.get_reader(video_file)
    except Exception as e:
        print(f"Error reading {video_file}: {e}")
        return

    video_frames = []
    for frame in video_reader:
        video_frames.append(frame)
    input_video = np.array(video_frames)
    meta_data = video_reader.get_meta_data()
    video_fps = meta_data["fps"]
    T, H, W, C = input_video.shape

    # Clip the video to 10 seconds (or at least 300 frames)
    num_frame_thres = max(int(np.ceil(video_fps * 10)), 300)
    output_video = input_video[:num_frame_thres] if T > num_frame_thres else input_video

    output_video = resize_video(output_video, num_frames, width)

    # Write output as .mp4 regardless of input extension
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    output_file = os.path.join(output_dir, base_name + ".mp4")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("Writing processed video to", output_file)
    media.write_video(output_file, output_video, fps=video_fps)


def process_directory(
    input_dir: str, output_dir: str, recursive: bool = False, num_frames: int = 17, width: int = 320
) -> None:
    """Process all video files in input_dir (excluding those in a 'processed' folder)
    and write the processed videos to output_dir."""
    video_extensions = ["mp4", "mov", "avi", "mkv"]
    input_files = []
    if recursive:
        for ext in video_extensions:
            pattern = os.path.join(input_dir, "**", f"*.{ext}")
            input_files.extend(glob(pattern, recursive=True))
    else:
        for ext in video_extensions:
            pattern = os.path.join(input_dir, f"*.{ext}")
            input_files.extend(glob(pattern))

    # Exclude files that are already in a 'processed' subdirectory.
    input_files = [f for f in input_files if "processed" not in os.path.normpath(f).split(os.sep)]

    input_files = sorted(input_files)
    print(f"Found {len(input_files)} videos in {input_dir} to process.")
    for video_file in input_files:
        process_video_file(video_file, output_dir, num_frames, width)


# ----------------------------
# Metric Computation Functions
# ----------------------------
_FLOAT32_EPS = np.finfo(np.float32).eps
_UINT8_MAX_F = float(np.iinfo(np.uint8).max)


def PSNR(input0: np.ndarray, input1: np.ndarray) -> float:
    """Compute PSNR between two videos or images."""
    assert input0.shape == input1.shape, "inputs should have the same shape"
    mse = ((input0 - input1) ** 2).mean()
    psnr = 20 * np.log10(_UINT8_MAX_F / (np.sqrt(mse) + _FLOAT32_EPS))
    return psnr.item()


def SSIM(input0: np.ndarray, input1: np.ndarray) -> float:
    """Compute SSIM between two videos or images."""
    assert input0.shape == input1.shape, "inputs should have the same shape"
    # If a single video/image, wrap it in an array for uniformity.
    if input0.ndim == 3:
        input0, input1 = np.array([input0]), np.array([input1])
    from concurrent.futures import ThreadPoolExecutor

    def compute_ssim(pair):
        one_image0, one_image1 = pair
        return ssim(one_image0, one_image1, data_range=_UINT8_MAX_F, multichannel=True, channel_axis=-1)

    with ThreadPoolExecutor() as executor:
        ssim_values = list(executor.map(compute_ssim, zip(input0, input1)))
    return np.mean(ssim_values)


def main_psnr_ssim() -> None:
    # Build dictionaries mapping file basename to full path.
    gt_files = {os.path.basename(f): f for f in glob(os.path.join(args.gtpath, f"*.{args.ext}"))}
    target_files = {os.path.basename(f): f for f in glob(os.path.join(args.targetpath, f"*.{args.ext}"))}

    # Files that are common in both directories.
    included_keys = sorted(set(gt_files.keys()) & set(target_files.keys()))
    # Files that exist only in one directory.
    skipped_keys = sorted(set(gt_files.keys()).symmetric_difference(set(target_files.keys())))

    print(f"Included: {', '.join(included_keys)} | Skipped: {', '.join(skipped_keys)}")

    if not included_keys:
        print("No matching files found.")
        return

    psnr_values, ssim_values = [], []
    print(f"Calculating PSNR and SSIM on {len(included_keys)} pairs ...")
    for key in tqdm(included_keys, total=len(included_keys)):
        file0 = gt_files[key]
        file1 = target_files[key]
        vid0 = read_video(file0).astype(np.float32)
        vid1 = read_video(file1).astype(np.float32)
        psnr_val = PSNR(vid0, vid1)
        ssim_val = SSIM(vid0, vid1)
        psnr_values.append([key, float(psnr_val)])
        ssim_values.append([key, float(ssim_val)])
        print(f"{key} PSNR: {psnr_val:.3f}, SSIM: {ssim_val:.3f}")

    mean_psnr = np.mean([val for _, val in psnr_values])
    mean_ssim = np.mean([val for _, val in ssim_values])
    print(f"Mean PSNR: {mean_psnr:.3f}")
    print(f"Mean SSIM: {mean_ssim:.3f}")

    with open(os.path.join(args.targetpath, "psnr.json"), "w") as fw:
        json.dump(psnr_values, fw)
    with open(os.path.join(args.targetpath, "ssim.json"), "w") as fw:
        json.dump(ssim_values, fw)


# ----------------------------
# Main Function and Argument Parsing
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process two directories of videos (reference and generated) and compute PSNR and SSIM metrics."
    )
    parser.add_argument(
        "--gtpath", type=str, required=True, help="Path to the directory of reference (ground-truth) videos"
    )
    parser.add_argument("--targetpath", type=str, required=True, help="Path to the directory of generated videos")
    parser.add_argument("--ext", type=str, default="mp4", help="Video file extension (e.g., mp4)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for metrics computation")
    parser.add_argument(
        "--recursive", action="store_true", help="If set, process videos in subdirectories recursively"
    )
    parser.add_argument("--width", type=int, default=320, help="Target width (shorter side) for video resizing")
    parser.add_argument("--num_frames", type=int, default=17, help="Number of frames to keep for each processed video")
    args = parser.parse_args()

    # Always process videos.
    gt_processed = os.path.join(args.gtpath, "processed")
    os.makedirs(gt_processed, exist_ok=True)
    gen_processed = os.path.join(args.targetpath, "processed")
    os.makedirs(gen_processed, exist_ok=True)
    print("Processing reference videos ...")
    process_directory(
        args.gtpath, gt_processed, recursive=args.recursive, num_frames=args.num_frames, width=args.width
    )
    print("Processing generated videos ...")
    process_directory(
        args.targetpath, gen_processed, recursive=args.recursive, num_frames=args.num_frames, width=args.width
    )
    # Use the processed directories for metric computation.
    args.gtpath = gt_processed
    args.targetpath = gen_processed

    # Always compute PSNR and SSIM.
    main_psnr_ssim()
