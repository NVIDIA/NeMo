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

import json
import math
import os
import tarfile
import zlib
from pathlib import Path
from typing import Any, List

import decord
import numpy as np
import torch
from dataverse.datasets import DataField
from dataverse.utils.ndas.av_metadata import (
    extract_calibration_from_tar,
    extract_obstacle_and_epomotion_from_tar,
    get_clip_to_tar,
    get_egopose_interp,
    get_obstacle_interp,
    get_rig_transform,
    key2id,
    pose_to_corn,
)
from dataverse.utils.ndas.camera_model import FThetaCamera, IdealPinholeCamera
from lru import LRU
from platformdirs import user_cache_path
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from torch.nn.functional import grid_sample

from .base import BaseDataset


class MetadataLoaderV1(object):
    def __init__(
        self,
        tar_dir=None,
        obs_pose_dir=None,
        calib_dir=None,
        load_obstacle=False,
    ):
        self.tar_dir = tar_dir
        self.obstaclepose_clip2tar = get_clip_to_tar(obs_pose_dir)
        self.calibration_clip2tar = get_clip_to_tar(calib_dir)
        self.load_obstacle = load_obstacle

        # Find common keys of the metadata
        self.tar_index = self._build_tar_index()
        annotated_clip_keys = set(self.obstaclepose_clip2tar.keys()).intersection(
            set(self.calibration_clip2tar.keys())
        )

        tar_clip_names = set(self.tar_index.keys())
        for clip_name in tar_clip_names:
            if key2id(clip_name) not in annotated_clip_keys:
                del self.tar_index[clip_name]
        self.clip_names = list(self.tar_index.keys())

    def _build_tar_index(self):
        assert self.tar_dir is not None
        tar_files = list(Path(self.tar_dir).glob("*.tar"))
        tar_index_path = user_cache_path("dataverse") / "av_tar_index.json"
        tar_index_path.parent.mkdir(parents=True, exist_ok=True)

        if tar_index_path.exists():
            with tar_index_path.open("r") as f:
                tar_index = json.load(f)

        else:
            tar_index = {}
            for tar_file in tar_files:
                with tarfile.open(tar_file, "r") as tar:
                    tar_members = tar.getmembers()
                tar_members = [member.name.split(".")[0] for member in tar_members]
                tar_members = list(set(tar_members))
                for clip_name in tar_members:
                    tar_index[clip_name] = str(tar_file)
            with tar_index_path.open("w") as f:
                json.dump(tar_index, f)

        if len(set(tar_index.values())) != len(tar_files):
            print("Cache sanity check failed. Rebuilding")
            os.unlink(tar_index_path)
            return self._build_tar_index()

        return tar_index

    def load_metadata(self, sample_key, camkeys):
        """Returns dictionary for the clip with the following contents:
        {
            camera_key_i:
                {
                    sensor_name: str, name of the lidar sensor
                    egopose: list, egoposes sorted in time
                    lidar2rig: array, lidar to rig transformation
                    rig2cam: array, rig to camera transformation
                    ftheta: Ftheta camera object
                    obstacles: dict, contains trackid->[detection1,detection2,...] sorted in time
                }
        }
        """

        # Load obstacles, egoposes, sensor_name
        metadata = extract_obstacle_and_epomotion_from_tar(
            sample_key,
            camkeys,
            self.obstaclepose_clip2tar,
            load_obstacle=self.load_obstacle,
        )
        if not metadata:
            return None

        # Load rig_info egoparams
        calibration = extract_calibration_from_tar(sample_key, self.calibration_clip2tar)
        if not calibration:
            return None

        # Merge calibration to metadata
        rig_info = calibration["rig_info"]
        metadata["egoparams"] = calibration["egoparams"]
        for camera_name in camkeys:
            camera_data = metadata[camera_name]
            lidarkey = camera_data["sensor_name"]
            camera_data["lidar2rig"] = get_rig_transform(rig_info[lidarkey], rig2sensor=False)
            camera_data["rig2cam"] = get_rig_transform(rig_info[camera_name + ".mp4"], rig2sensor=True)
            camera_data["ftheta"] = FThetaCamera.from_dict(rig_info[camera_name + ".mp4"])
        return metadata


class AlpamayoV1(BaseDataset):
    MAX_CACHED_VIDEOS = 2
    ROT_FIXER = torch.from_numpy(Rotation.from_euler("xzy", [0.0, -np.pi / 2, np.pi / 2]).as_matrix())[None].float()

    def __init__(
        self,
        tar_dir: str,
        calib_dir: str,
        obs_pose_dir: str,
        frame_height: int = 720,
        frame_width: int = 1280,
        fov_range: list = [30, 70],
        fov_rng_seed: int = 0,
        multi_cam: bool = False,
    ):
        super().__init__()

        if multi_cam:
            self.camkeys = [
                "camera_front_wide_120fov",
                "camera_cross_left_120fov",
                "camera_cross_right_120fov",
            ]
        else:
            self.camkeys = ["camera_front_wide_120fov"]
        self.metadata_loader = MetadataLoaderV1(
            tar_dir=tar_dir,
            obs_pose_dir=obs_pose_dir,
            calib_dir=calib_dir,
        )

        self.tar_dir = tar_dir
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.fov_range = fov_range
        self.fov_rng_seed = fov_rng_seed
        self.video_loader_cache = LRU(self.MAX_CACHED_VIDEOS)
        self.meta_data_cache = LRU(self.MAX_CACHED_VIDEOS)

    def num_videos(self) -> int:
        return len(self.metadata_loader.clip_names)

    def num_views(self, video_idx: int) -> int:
        return len(self.camkeys)

    def num_frames(self, video_idx: int, view_idx: int = 0) -> int:
        loaders, _ = self._get_video_loader(video_idx)
        return len(loaders[self.camkeys[view_idx]])

    def available_data_fields(self) -> List[DataField]:
        return [
            DataField.IMAGE_RGB,
            DataField.CAMERA_C2W_TRANSFORM,
            DataField.CAMERA_INTRINSICS,
        ]

    def _get_video_loader(self, video_idx: int):
        if video_idx in self.video_loader_cache:
            return self.video_loader_cache[video_idx]

        clip_name = self.metadata_loader.clip_names[video_idx]
        video_loaders, video_info = {}, {}
        tar_name = self.metadata_loader.tar_index[clip_name]

        with tarfile.open(os.path.join(self.tar_dir, tar_name), "r") as f:
            for key in self.camkeys:
                mp4_name = f"{clip_name}.{key}.mp4"

                vr = decord.VideoReader(f.extractfile(mp4_name))
                video_loaders[key] = vr

                infokey = mp4_name.replace(".mp4", ".json")
                vidinfo = json.load(f.extractfile(infokey))
                video_info[key] = vidinfo

        return video_loaders, video_info

    def _get_meta_data(self, video_idx: int):
        if video_idx in self.meta_data_cache:
            return self.meta_data_cache[video_idx]

        clip_name = self.metadata_loader.clip_names[video_idx]
        mdata = self.metadata_loader.load_metadata(clip_name, self.camkeys)

        ego_data = {}
        for camera_name in self.camkeys:
            camera_metadata = mdata[camera_name]
            egopose_lerp, cliptbase = get_egopose_interp(camera_metadata["egopose"])
            if "obstacles" in camera_metadata:
                obstacle_lerp = get_obstacle_interp(["obstacles"], egopose_lerp, cliptbase)
            else:
                obstacle_lerp = None
            ego_data[camera_name] = {
                "egopose_lerp": egopose_lerp,
                "cliptbase": cliptbase,
                "obstacle_lerp": obstacle_lerp,
            }

        return mdata, ego_data

    @staticmethod
    def _get_obstacles_and_camera(
        egopose_lerp,
        lidar2rig,
        rig2cam,
        t0,
        obstacle_lerp=None,
    ):
        # Process camera poses
        # convert global lidar -> local lidar
        ego_qw, ego_qx, ego_qy, ego_qz, ego_x, ego_y, ego_z = egopose_lerp["interp"](t0)
        ego_quat = Quaternion(w=ego_qw, x=ego_qx, y=ego_qy, z=ego_qz)

        # convert global lidar -> local lidar
        global2lidar_trans = np.eye(4)
        global2lidar_trans[:3, 3] = -np.array([ego_x, ego_y, ego_z])
        global2lidar_rot = np.eye(4)
        global2lidar_rot[:3, :3] = ego_quat.inverse.rotation_matrix

        # calculate world to camera transform
        # Lidar Global -> Lidar Local -> Rig -> Camera
        world2cam = rig2cam @ lidar2rig @ global2lidar_rot @ global2lidar_trans

        # Process obstacles
        pts = np.empty((0, 3))
        if obstacle_lerp:
            # obstacle points (8 corners)
            N = 0
            for id in obstacle_lerp:
                if obstacle_lerp[id]["tmin"] <= t0 <= obstacle_lerp[id]["tmax"]:
                    qw, qx, qy, qz, x, y, z, le, wi, he = obstacle_lerp[id]["interp"](t0)
                    corn = pose_to_corn(qw, qx, qy, qz, x, y, z, le, wi, he)
                    pts = np.concatenate((pts, corn), 0)
                    N += 1

            pts = pts.astype(np.float32)

            # append 1 to pts
            pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)

            pts = pts @ world2cam.T
            pts = pts[:, :3]

            # pts = pts[pts[..., 0] > 0]

        return pts, world2cam

    def _read_data(
        self,
        video_idx: int,
        frame_idxs: List[int],
        view_idxs: List[int],
        data_fields: List[DataField],
    ) -> dict[DataField, Any]:
        # Load videos
        clip_name = self.metadata_loader.clip_names[video_idx]
        video_loaders, video_info = self._get_video_loader(video_idx)
        meta_data, ego_data = self._get_meta_data(video_idx)

        # Set fov if randomized
        seed = zlib.adler32(f"{clip_name}-{self.fov_rng_seed}".encode("utf-8"))
        pinhole_fovd = np.random.RandomState(seed).randint(self.fov_range[0], self.fov_range[1] + 1)
        # target pinhole image/camera in the original resolution
        tgt_focal = self.frame_width / (2.0 * math.tan(np.deg2rad(pinhole_fovd) / 2.0))
        tgt_focal = np.array([tgt_focal, tgt_focal])
        ys, xs = np.mgrid[0 : self.frame_height, 0 : self.frame_width]
        pixels = np.stack((xs, ys), axis=2)
        pinhole_cam = IdealPinholeCamera(
            f_x=tgt_focal[0],
            f_y=tgt_focal[1],
            width=self.frame_width,
            height=self.frame_height,
        )
        pinhole_rays = pinhole_cam.pixel2ray(pixels.reshape(-1, 2))  # hw x 3

        # Pre-compute projection for all possible cameras.
        pos_norms = {}
        for view_idx in set(view_idxs):
            cam_key = self.camkeys[view_idx]
            fisheye_cam = meta_data[cam_key]["ftheta"]
            pos = fisheye_cam.ray2pixel(pinhole_rays)
            # fish eye to pinhole
            pos_norm = (
                2.0
                * pos
                / np.array(
                    [fisheye_cam.width - 1.0, fisheye_cam.height - 1.0],
                    dtype=np.float32,
                )
                - 1.0
            )
            pos_norm = torch.from_numpy(pos_norm).reshape(1, self.frame_height, self.frame_width, 2)
            pos_norms[view_idx] = pos_norm

        output_dict = {}
        for data_field in data_fields:
            if data_field == DataField.IMAGE_RGB:
                rgb_list = []
                for frame_idx, view_idx in zip(frame_idxs, view_idxs):
                    cam_key = self.camkeys[view_idx]
                    pos_norm = pos_norms[view_idx]
                    img_fisheye = video_loaders[cam_key][frame_idx].asnumpy()
                    img = torch.from_numpy(img_fisheye).permute(2, 0, 1).float().unsqueeze(0)
                    img = grid_sample(img, pos_norm, mode="bilinear", align_corners=False)[0]
                    img = img.float() / 255.0
                    rgb_list.append(img)

                output_dict[data_field] = torch.stack(rgb_list)

            elif data_field == DataField.CAMERA_C2W_TRANSFORM:
                c2w_list = []
                for frame_idx, view_idx in zip(frame_idxs, view_idxs):
                    cam_key = self.camkeys[view_idx]
                    t0_stamp = video_info[cam_key][frame_idx]["timestamp"]
                    t0_val = 1e-6 * (t0_stamp - ego_data[cam_key]["cliptbase"])

                    _, world2cam = self._get_obstacles_and_camera(
                        ego_data[cam_key]["egopose_lerp"],
                        meta_data[cam_key]["lidar2rig"],
                        meta_data[cam_key]["rig2cam"],
                        t0_val,
                        obstacle_lerp=None,
                    )

                    world2cam = torch.from_numpy(world2cam).float()
                    cam2world = torch.inverse(world2cam)
                    cam2world[:3, :3] = cam2world[:3, :3] @ self.ROT_FIXER
                    c2w_list.append(cam2world)

                output_dict[data_field] = torch.stack(c2w_list)

            elif data_field == DataField.CAMERA_INTRINSICS:
                single_intr = torch.tensor(
                    [
                        tgt_focal[0],
                        tgt_focal[1],
                        self.frame_width / 2,
                        self.frame_height / 2,
                    ]
                ).float()
                output_dict[data_field] = single_intr.repeat(len(frame_idxs), 1)

            else:
                raise NotImplementedError(f"Can't handle data field {data_field}")

        return output_dict
