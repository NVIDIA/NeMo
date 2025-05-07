# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# pylint: disable=C0115,C0116,C0301

import gc
import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from einops import rearrange
from torch.utils.data import DataLoader

import nemo.collections.physicalai.datasets.dataverse.dataverse.utils.alpamayo.rig_decoder as rig_decoder
import nemo.collections.physicalai.datasets.dataverse.dataverse.utils.alpamayo.transformation as transformation
from nemo.collections.physicalai.datasets.dataverse.dataverse.datasets.base import DataField
from nemo.collections.physicalai.datasets.dataverse_dataset.driving_dataloader.config_dataverse import DATAVERSE_CONFIG
from nemo.collections.physicalai.datasets.dataverse_dataset.driving_dataloader.dataloader_utils import (
    dict_collation_fn,
)
from nemo.collections.physicalai.datasets.dataverse_dataset.instantiate_utils import instantiate_from_config

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False

import torch.profiler


class DrivingVideoDataLoader(DataLoader):
    def __init__(self, dataset, batch_size: int = 1, *args, **kw):
        dataset_obj = dataset.build_dataset()
        if "dataloaders" in kw:
            kw.pop("dataloaders")
        super().__init__(dataset_obj, batch_size, collate_fn=dict_collation_fn, *args, **kw)


def get_driving_dataset(
    dataset_name="alpamayo_v2",
):
    return DrivingDataset(dataset_name)


class DrivingDataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset_config = DATAVERSE_CONFIG[dataset_name]

    def build_dataset(self):
        # we only create the dataset when we call this function
        return InfiniteDataVerse(**self.dataset_config)


class InfiniteDataVerse:
    def __init__(
        self,
        dataset_cfg,
        batch_size=1,
        sample_n_frames=8,
        sample_size=[320, 512],
        crop_size=None,
        load_trajectory=False,
        load_frame_repeat=False,
        fps=14,
        load_video=False,
    ):
        self.dataset = instantiate_from_config(dataset_cfg)
        self.n_data = self.dataset.num_videos()

        self.load_trajectory = load_trajectory
        self.load_frame_repeat = load_frame_repeat
        self.fps = fps
        self.load_video = load_video
        # Split the data by node, make sure each node has different data sample
        # Ranks of the same pp/tp/cp group will have the same dp rank and thus share the same group id.
        if parallel_state.is_initialized():
            dp_group_id = parallel_state.get_data_parallel_rank()
            dp_world_size = parallel_state.get_data_parallel_world_size()
            logging.critical(
                f"Using parallelism size CP :{parallel_state.get_context_parallel_world_size()}, "
                + f"TP :{parallel_state.get_tensor_model_parallel_world_size()} for video dataset, "
                + f"DP: {dp_group_id}, DP World size: {dp_world_size}"
            )
        else:
            dp_world_size = 1
            dp_group_id = 0
        self.n_data_per_node = self.n_data // dp_world_size
        self.data_start_idx = dp_group_id * self.n_data_per_node
        self.dp_group_id = dp_group_id

        # Make an infinite loop
        maximum_iter = 1e8 * batch_size  # a hack to create infinite loop
        self.multiplier = int(maximum_iter // self.n_data_per_node)

        self.sample_n_frames = sample_n_frames
        self.sample_size = sample_size

        if crop_size is None:
            if self.sample_size != []:
                self.crop_size = self.sample_size
            else:
                self.crop_size = [512, 1024]
        else:
            self.crop_size = crop_size

        if self.sample_size != []:
            self.img_transform = transforms.Compose(
                [
                    transforms.Resize(
                        sample_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                    ),
                    transforms.CenterCrop(self.crop_size),
                ]
            )
            self.norm_image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

        cache_dir = os.environ.get("XDG_CACHE_HOME")
        self.camera_t5_embedding = {}
        for camera in self.dataset.camkeys:
            self.camera_t5_embedding[camera] = torch.load(
                os.path.join(cache_dir, "multicamera", f"video_camera_embeddings_v0_{camera}.pt")
            )

        self.camera_text_caption = {
            "camera_front_wide_120fov": "The video is captured from a camera mounted on a car. The camera is facing forward.",
            "camera_rear_tele_30fov": "The video is captured from a camera mounted on a car. The camera is facing backwards.",
            "camera_cross_left_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
            "camera_cross_right_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
            "camera_rear_right_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear right side.",
            "camera_rear_left_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
        }

    def __len__(self):
        return self.multiplier * self.n_data_per_node

    def transform_data(self, sampled_images):
        n_frames, _, H, W = sampled_images.shape  # (N, C, H, W)
        if self.sample_size != []:
            sampled_images = self.img_transform(sampled_images)
            sampled_images = self.norm_image(sampled_images)  # (N, C, H, W)
        sample = {
            "video": sampled_images.permute(1, 0, 2, 3).contiguous(),  # (C, N, H, W) format for cosmos
            "is_preprocessed": True,
        }
        return sample

    def convert_coordinate(self, xyz, rig_info):
        cam_name = 'camera_front_wide_120fov'
        xyz = rearrange(xyz, '(t c) -> t c', c=3)
        rig_info = rig_decoder.decode_rig_info(rig_info)
        camera_extrinsics = torch.from_numpy(transformation.sensor_to_rig(rig_info[cam_name]))

        rig_to_camera = np.zeros_like(camera_extrinsics)
        rig_to_camera[:3, :3] = camera_extrinsics[:3, :3].T
        rig_to_camera[:3, 3] = -camera_extrinsics[:3, :3].T @ camera_extrinsics[:3, 3]
        camera_xyz = (xyz @ rig_to_camera[:3, :3].T) + rig_to_camera[:3, 3]
        return camera_xyz

    def sample_frame_indices(self, chunk_frame_start, num_samples_in_chunk):
        video_fps = 30  # video fps is fixed for alpamayo data
        # stride used for subsampling video
        stride = int(video_fps / self.fps)
        # This is the actual target fps we obtain after subsampling
        # Start index is randomly selected in the chunk
        if num_samples_in_chunk != int(self.sample_n_frames * stride):
            frame_start = chunk_frame_start + int(
                np.random.choice(num_samples_in_chunk - int(self.sample_n_frames * stride), 1)
            )
        else:
            frame_start = chunk_frame_start

        frame_end = frame_start + self.sample_n_frames * stride

        return np.arange(frame_start, frame_end, stride).tolist(), frame_start

    def __getitem__(self, idx):
        rank = dist.get_rank() if dist.is_initialized() else 0
        data_idx = (idx % self.n_data_per_node) + self.data_start_idx
        assert data_idx < self.n_data
        data_fields = [DataField.IMAGE_RGB]
        if self.load_trajectory:
            data_fields.append(DataField.TRAJECTORY)

        view_indices = [i for i in range(len(self.dataset.camkeys))]
        try:
            t5_and_meta_data = self.dataset._read_t5_and_meta(video_idx=data_idx, view_idxs=view_indices)
        except Exception:
            print(
                f"RANK {rank}: T5 Meta Data Loading ERROR for video_idx {data_idx}. Skip and continue load the next Video."
            )
            return self.__getitem__((idx + 1) % len(self))
        clip_id = self.dataset.clip_names[data_idx]

        available_views = t5_and_meta_data.keys()
        if len(available_views) == 0:
            print(
                f"RANK {rank}: T5 Meta Data Loading ERROR for video_idx {data_idx}. The available_views is empty. Skip and continue load the next Video."
            )
            return self.__getitem__((idx + 1) % len(self))
        determin_view_id = min(list(available_views))
        random_selection_chunk_id = 0

        start_ids = int(t5_and_meta_data[determin_view_id]["meta_data"][2][random_selection_chunk_id])
        end_ids = int(t5_and_meta_data[determin_view_id]["meta_data"][3][random_selection_chunk_id])

        sample_frame_indices, start_index = self.sample_frame_indices(start_ids, end_ids - start_ids)

        read_data_view_indices = []
        read_data_frame_indices = []
        for view in view_indices:
            read_data_frame_indices.extend(sample_frame_indices)
            read_data_view_indices.extend([view] * len(sample_frame_indices))

        if self.load_video:
            try:
                data = self.dataset._read_raw_video(
                    video_idx=data_idx,
                    data_fields=data_fields,
                    frame_idxs=read_data_frame_indices,
                    view_idxs=read_data_view_indices,
                )
            except Exception:
                print(
                    f"RANK {rank}: Data reading ERROR for video_idx {data_idx}. Skip and continue load the next Video."
                )
                return self.__getitem__((idx + 1) % len(self))
            sample = self.transform_data(data[DataField.IMAGE_RGB].clone())
        else:
            sample = {}
        clip_name = "%d-%03d" % (data_idx, start_index)
        caption = ""
        try:
            dummy_text_embedding = torch.zeros(512 * len(view_indices), 1024)
            dummy_text_mask = torch.zeros(512 * len(view_indices))
            raw_embeddings = []
            raw_caption = []
            for view_id in view_indices:
                if view_id in t5_and_meta_data:
                    curr_camera_text = self.camera_text_caption[self.dataset.camkeys[view_id]]

                    curr_caption = t5_and_meta_data[view_id]["meta_data"][1][random_selection_chunk_id]
                    curr_caption = curr_camera_text + curr_caption
                    text_embedding_np = t5_and_meta_data[view_id]["T5"][random_selection_chunk_id]
                    raw_embeddings.append(text_embedding_np)
                    raw_caption.append(t5_and_meta_data[view_id]["meta_data"][1][random_selection_chunk_id])
                    text_embedding = torch.from_numpy(text_embedding_np)
                    curr_camera_t5_embedding = self.camera_t5_embedding[self.dataset.camkeys[view_id]]

                    # Concatenate the embeddings
                    combined_text_embedding = torch.cat([curr_camera_t5_embedding, text_embedding], dim=0)
                    n_text = combined_text_embedding.shape[0]

                    # Ensure the combined embedding does not exceed the maximum size
                    if n_text > 512:
                        n_text = 512
                        combined_text_embedding = combined_text_embedding[:n_text]

                    start_idx = view_id * 512
                    end_idx = start_idx + n_text

                    # Check for dimension overflow
                    if end_idx > dummy_text_embedding.shape[0]:
                        n_text = dummy_text_embedding.shape[0] - start_idx
                        combined_text_embedding = combined_text_embedding[:n_text]
                        end_idx = start_idx + n_text

                    dummy_text_embedding[start_idx:end_idx] = combined_text_embedding
                    dummy_text_mask[start_idx:end_idx] = 1
                    del text_embedding_np

                else:
                    curr_camera_text = self.camera_text_caption[self.dataset.camkeys[view_id]]
                    curr_camera_t5_embedding = self.camera_t5_embedding[self.dataset.camkeys[view_id]]
                    camera_n_text = curr_camera_t5_embedding.shape[0]

                    curr_caption = curr_camera_text
                    dummy_text_embedding[view_id * 512 : view_id * 512 + camera_n_text] = curr_camera_t5_embedding
                    dummy_text_mask[view_id * 512 : view_id * 512 + camera_n_text] = 1

                caption += curr_caption
                caption += " ;"
            sample["t5_text_embeddings"] = dummy_text_embedding
            sample["t5_text_mask"] = dummy_text_mask
            sample["t5_raw_text_embeddings"] = raw_embeddings
            sample["raw_caption"] = raw_caption

        except Exception:
            print(
                f"RANK {rank}: Dataloading ERROR for video_idx {data_idx} on T5 loading. Skip and continue load the next Video."
            )
            return self.__getitem__((idx + 1) % len(self))
        sample["num_frames"] = self.sample_n_frames
        sample["image_size"] = torch.from_numpy(np.asarray(self.crop_size))
        sample["fps"] = self.fps
        sample["__key__"] = clip_name
        sample["clip_name"] = clip_name
        sample["padding_mask"] = torch.zeros(1, self.crop_size[0], self.crop_size[1])
        sample["caption"] = caption
        sample["clip_id"] = clip_id
        if self.load_frame_repeat:
            sample['frame_repeat'] = data['num_repeated_frames']

        if self.load_trajectory:
            # with lvg, use rig coordinate
            trajectory = rearrange(data[DataField.TRAJECTORY], '(t c) -> t c', c=3)
            trajectory = trajectory / torch.FloatTensor([[10.0, 4.0, 1.0]])
            sample["trajectory"] = rearrange(trajectory, 't c -> (t c)')

        gc.collect()
        return sample
