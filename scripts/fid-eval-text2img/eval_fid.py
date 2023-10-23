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
"""
Example usage:
   python eval_fid.py \
     --coco_images_path /path/to/coco2014_val \
     --fid_images_path /path/to/synthetic_images \
     --output_path /path/to/output/fid_scores.csv

1. `--coco_images_path`: The path to the real images directory. In this example,
    it is set to `/path/to/coco2014_val`. This path should point to the
    directory containing the COCO 2014 validation dataset images, resized
    to 256x256 pixels.

2. `--fid_images_path`: The path to the directory containing subfolders
    with synthetic images. In this example, it is set to
    `/path/to/synthetic_images`. Each subfolder should contain a
    set of synthetic images for which you want to compute FID scores
    against the real images from `--coco_images_path`.

3. `--output_path`: The path to the output CSV file where the FID scores
    will be saved. In this example, it is set to
    `/path/to/output/fid_scores.csv`. This file will contain a table with
    two columns: `cfg` and `fid`. The `cfg` column lists the
    names of the subfolders in `--fid_images_path`, and the `fid` column
    lists the corresponding FID scores between the synthetic images in
    each subfolder and the real images from `--coco_images_path`.
"""

import argparse
import csv
import os
import torch

from compute_fid import compute_fid_data
from fid_dataset import CustomDataset

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_images_path', default='/coco2014/coco2014_val/images_256', type=str)
    parser.add_argument('--fid_images_path', default=None, type=str)
    parser.add_argument('--output_path', default='./fid_scores.csv', type=str)
    args = parser.parse_args()

    # Set paths for synthetic images and real images
    fid_images_path = args.fid_images_path
    real_path = args.coco_images_path

    # Create dataset and data loader for real images
    real_dataset = CustomDataset(real_path)
    loader_real = torch.utils.data.DataLoader(
        real_dataset, batch_size=32, num_workers=0, pin_memory=True, drop_last=False
    )

    # Create output CSV file
    with open(args.output_path, 'w', newline='') as csvfile:
        fieldnames = ['cfg', 'fid']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through subfolders in fid_images_path
        for subfolder in os.listdir(fid_images_path):
            subfolder_path = os.path.join(fid_images_path, subfolder)
            if os.path.isdir(subfolder_path):
                # Create dataset and data loader for synthetic images in subfolder
                synthetic_dataset = CustomDataset(subfolder_path, target_size=256)
                loader_synthetic = torch.utils.data.DataLoader(
                    synthetic_dataset, batch_size=32, num_workers=0, pin_memory=True, drop_last=False
                )

                # Compute FID score between synthetic images in subfolder and real images
                fid = compute_fid_data(
                    './',
                    loader_real,
                    loader_synthetic,
                    key_a=0,
                    key_b=0,
                    sample_size=None,
                    is_video=False,
                    few_shot_video=False,
                    network='tf_inception',
                    interpolation_mode='bilinear',
                )

                print(f"The FID score between {subfolder_path} and {real_path} is {fid}")

                # Write FID score to output CSV file
                writer.writerow({'cfg': subfolder, 'fid': fid})
