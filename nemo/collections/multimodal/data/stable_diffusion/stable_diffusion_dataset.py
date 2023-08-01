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
import torch

from nemo.collections.multimodal.data.common.webdataset import WebDatasetCommon
from nemo.collections.multimodal.data.stable_diffusion.augmentation.augmentations import (
    construct_image_augmentations,
    identical_transform,
)


def build_train_valid_datasets(
    model_cfg, consumed_samples,
):
    data_cfg = model_cfg.data

    def build_resolution_filter(value=None, method='larger'):
        assert method == 'larger' or method == 'smaller'
        if method == 'larger':
            print(f'Only Selecting images with resolution >= {value}')
            return lambda x: x['jpg'].size[0] >= value and x['jpg'].size[1] >= value
        print(f'Only Selecting images with resolution <= {value}')
        return lambda x: x['jpg'].size[0] <= value and x['jpg'].size[1] <= value

    # This function maps data that are tuples to dictionary.
    def tuple_to_dict(inp):
        for input in inp:
            out_dict = dict()
            out_dict['images'] = input[0].permute(1, 2, 0)
            out_dict['captions'] = input[1]
            yield out_dict

    def transform_fn(sample):
        image, text = sample["jpg"], sample["txt"]
        # TODO : If no agumentations just return the image ?
        img_transform = construct_image_augmentations(data_cfg.train.get("augmentations", None))
        text_transform = identical_transform
        return img_transform(image), text_transform(text)

    filter_cfg = data_cfg.train.get('filterings', None)
    filter_fn = build_resolution_filter(**filter_cfg.resolution) if filter_cfg else None
    train_data = WebDatasetCommon(
        dataset_cfg=data_cfg,
        consumed_samples=consumed_samples,
        map_fn=transform_fn,
        compose_fn=tuple_to_dict,
        filter_fn=filter_fn,
        is_train=True,
    )

    val_data = None
    if data_cfg.get("validation") is not None and data_cfg.validation.get("data_path"):
        val_data = WebDatasetCommon(
            dataset_cfg=data_cfg,
            consumed_samples=consumed_samples,
            map_fn=transform_fn,
            compose_fn=tuple_to_dict,
            filter_fn=filter_fn,
            is_train=False,
        )

    return train_data, val_data


def build_train_valid_precached_datasets(
    model_cfg, consumed_samples,
):
    data_cfg = model_cfg.data

    # This function maps data that are tuples to dictionary.
    def tuple_to_dict(inp):
        for input in inp:
            out_dict = dict()
            out_dict[model_cfg.first_stage_key] = input[0]
            out_dict[model_cfg.cond_stage_key] = input[1]
            yield out_dict

    def transform_fn(sample):
        latents, text = sample["npy"], sample["txt"]
        latents = torch.from_numpy(latents)

        # latents are of shape ([4, 64, 64])
        text_transform = identical_transform
        return latents, text_transform(text)

    train_data = WebDatasetCommon(
        dataset_cfg=data_cfg,
        consumed_samples=consumed_samples,
        map_fn=transform_fn,
        compose_fn=tuple_to_dict,
        is_train=True,
    )

    val_data = None
    if data_cfg.get("validation") is not None and data_cfg.validation.get("data_path"):
        val_data = WebDatasetCommon(
            dataset_cfg=data_cfg,
            consumed_samples=consumed_samples,
            map_fn=transform_fn,
            compose_fn=tuple_to_dict,
            is_train=False,
        )

    return train_data, val_data
