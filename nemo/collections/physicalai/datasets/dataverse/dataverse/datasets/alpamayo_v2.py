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

import io
import json
import math
import os
import random
import tarfile
import zlib
from pathlib import Path
from typing import Any, List

import av
import dataverse.utils.alpamayo.egomotion_decoder as egomotion_decoder
import numpy as np
import torch
from dataverse.utils.ndas.av_metadata import get_egopose_interp, get_rig_transform, parse_calibration_data
from dataverse.utils.ndas.camera_model import FThetaCamera, IdealPinholeCamera
from lru import LRU
from omegaconf import DictConfig
from platformdirs import user_cache_path
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from torch.nn.functional import grid_sample

from .base import BaseDataset, DataField


class AlpamayoV2(BaseDataset):
    MAX_CACHED_VIDEOS = 2
    ROT_FIXER = torch.from_numpy(Rotation.from_euler("xzy", [0.0, -np.pi / 2, np.pi / 2]).as_matrix())[None].float()

    def __init__(
        self,
        tar_dirs: list[str],
        probe_tar: bool,
        rectify: DictConfig,
        camkeys: list[str] = ["camera_front_wide_120fov"],
        tar_cache_path: str = "av22_tar_index_full.json",
    ):
        super().__init__()

        self.camkeys = camkeys
        self.trajectory_base_camera_index = self.camkeys.index("camera_front_wide_120fov")
        self.tar_cache_path = tar_cache_path
        self.tar_dirs = tar_dirs
        self.rectify_cfg = rectify
        self.probe_tar = probe_tar

        # Mapping from clip_name to tar file
        self.tar_index = self._build_tar_index()
        self.clip_names = sorted(list(self.tar_index.keys()))

        self.video_loader_cache = LRU(self.MAX_CACHED_VIDEOS)
        self.meta_data_cache = LRU(self.MAX_CACHED_VIDEOS)

    def _build_tar_index(self):
        """
        Builds the tar index from tar files, using a cached version if available.
        Ensures consistency in cache and rebuilds if the cache sanity check fails.

        Returns:
            dict: A mapping of clip names to their respective tar file paths.
        """
        tar_files = []
        for tar_dir in self.tar_dirs:
            tar_files.extend([str(p) for p in Path(tar_dir).glob("*.tar")])
        tar_index_path = user_cache_path("dataverse") / self.tar_cache_path
        tar_index_path.parent.mkdir(parents=True, exist_ok=True)

        if tar_index_path.exists():
            with tar_index_path.open("r") as f:
                tar_index = json.load(f)
            if len(set(tar_index.values())) != len(tar_files):
                print("Cache sanity check failed. Rebuilding")
                os.unlink(tar_index_path)
                return self._build_tar_index()
        else:
            tar_index = {}
            for tar_file in tar_files:
                if self.probe_tar:
                    clip_name = Path(tar_file).stem
                    tar_index[clip_name] = str(tar_file)
                    continue
                with tarfile.open(tar_file, "r") as tar:
                    tar_members = tar.getmembers()
                tar_members = [member.name.split(".")[0] for member in tar_members]
                tar_members = list(set(tar_members))
                for clip_name in tar_members:
                    tar_index[clip_name] = str(tar_file)
            with tar_index_path.open("w") as f:
                json.dump(tar_index, f)

        return tar_index

    def num_videos(self) -> int:
        return len(self.clip_names)

    def num_views(self, video_idx: int) -> int:
        return len(self.camkeys)

    def num_frames(self, video_idx: int, view_idx: int = 0) -> int:
        _, _, _, sync_to_original, _ = self._get_meta_data(video_idx)
        return len(sync_to_original[self.camkeys[view_idx]])

    def available_data_fields(self) -> List[DataField]:
        fields = [
            DataField.IMAGE_RGB,
            DataField.CAMERA_C2W_TRANSFORM,
            DataField.RAY_DIRECTION,
            DataField.TRAJECTORY,
        ]
        if self.rectify_cfg.enabled:
            fields.append(DataField.CAMERA_INTRINSICS)
        return fields

    def _get_video_frames(
        self,
        video_idx: int,
        view_idxs: list[int],
        frame_idxs: list[int],
        sync_to_original: dict,
    ):
        clip_name = self.clip_names[video_idx]
        tar_path = self.tar_index[clip_name]

        video_frames = {}
        frame_idx_to_batch_idx = {}
        with tarfile.open(tar_path, "r") as f:
            for key in self.camkeys:
                # get frame_idxs of the current view
                target_frame_idxs = []
                for view_idx, frame_idx in zip(view_idxs, frame_idxs):
                    if self.camkeys[view_idx] == key:
                        target_frame_idxs.append(frame_idx)

                if len(target_frame_idxs) == 0:
                    # this camera was not requested in view_idxs
                    break

                # check if frame_idx are in an increasing order
                diff = np.diff(np.array(target_frame_idxs))
                assert np.all(diff > 0), "frame_idxs are given as non-increasing list"

                # read video of the current view
                fp = f.extractfile(f"{clip_name}.{key}.mp4")
                assert fp is not None, f"Video {clip_name}.{key}.mp4 not found"
                vid = io.BytesIO(fp.read())
                vid.seek(0)

                input_container = av.open(vid)
                input_container.streams.video[0].thread_type = 3
                average_fps = input_container.streams.video[0].average_rate
                time_base = input_container.streams.video[0].time_base
                average_frame_duration = int(1 / average_fps / time_base)

                # collect required frames of the current view
                frames = []
                frame_iterator = input_container.decode(video=0)
                cur_frame_idx_to_batch_idx = {}
                for batch_idx, target_frame_number in enumerate(target_frame_idxs):
                    adjusted_target_frame_number = sync_to_original[key][target_frame_number]
                    target_pts = adjusted_target_frame_number * average_frame_duration
                    for frame in frame_iterator:
                        # find the frame
                        if frame.pts == target_pts:
                            break

                    cur_frame_idx_to_batch_idx[target_frame_number] = batch_idx
                    frames.append(torch.as_tensor(frame.to_rgb().to_ndarray()))
                video_frames[key] = torch.stack(frames)
                frame_idx_to_batch_idx[key] = cur_frame_idx_to_batch_idx

        return video_frames, frame_idx_to_batch_idx

    @staticmethod
    def _extract_egomotion_from_sample(fp):
        """Extract egomotions from sample, used by AV2.1"""
        egopose_info = np.load(fp, allow_pickle=True)
        N, C = egopose_info["labels_data"].shape
        (K,) = egopose_info["labels_keys"].shape
        assert K == C, f"{K} {C}"

        egopose_parsed = [{} for i in range(N)]
        sensor_names = []
        target_keys = ["x", "y", "z", "qx", "qy", "qz", "qw", "timestamp"]
        for key_id, key in enumerate(egopose_info["labels_keys"]):
            values = egopose_info["labels_data"][:, key_id]
            if key == "coordinate_frame":
                sensor_names = values
                continue
            elif key not in target_keys:
                continue
            if key == "timestamp":
                key = "t"
            for i in range(N):
                egopose_parsed[i][key] = values[i]
        return {
            "egopose": egopose_parsed,
            "sensor_name": sensor_names[0],
        }

    @staticmethod
    def _extract_calibration_from_sample(fp):
        """Extract calibrations from sample, used by AV2.1"""
        calibration = json.load(fp)
        rig_info, egoparams = parse_calibration_data(calibration)
        return {"rig_info": rig_info, "egoparams": egoparams, "rig_raw": calibration}

    def _get_meta_data(self, video_idx: int):
        """Returns dictionary for the clip with the following contents:
        {
            camera_key_i:
                {
                    sensor_name: str, name of the lidar sensor
                    egopose: list, egoposes sorted in time
                    lidar2rig: array, lidar to rig transformation
                    rig2cam: array, rig to camera transformation
                    ftheta: Ftheta camera object
                }
        }
        """

        if video_idx in self.meta_data_cache:
            return self.meta_data_cache[video_idx]

        clip_name = self.clip_names[video_idx]
        tar_path = self.tar_index[clip_name]
        video_info = {}

        with tarfile.open(tar_path, "r") as f:
            camera_metadata = self._extract_egomotion_from_sample(f.extractfile(f"{clip_name}.egomotion.npz"))
            calibration = self._extract_calibration_from_sample(f.extractfile(f"{clip_name}.rig.json"))

            for key in self.camkeys:
                vidinfo = json.load(f.extractfile(f"{clip_name}.{key}.json"))
                video_info[key] = vidinfo

        # Synchronize timestamps
        min_t, max_t = np.inf, -np.inf
        original_timestamps = {}
        for key in self.camkeys:
            timestamps = np.array([info["timestamp"] for info in video_info[key]])
            min_t = min(min_t, timestamps.min())
            max_t = max(max_t, timestamps.max())
            original_timestamps[key] = timestamps
        ref_timestamps = original_timestamps[self.camkeys[0]]
        ref_timestamps = ref_timestamps[ref_timestamps >= min_t]
        ref_timestamps = ref_timestamps[ref_timestamps <= max_t]

        sync_to_original = {}
        for key in self.camkeys:
            sync_to_original[key] = np.searchsorted(original_timestamps[key], ref_timestamps)

        metadata, ego_data = {}, {}
        lidarkey = camera_metadata["sensor_name"]
        rig_info = calibration["rig_info"]
        metadata["egoparams"] = calibration["egoparams"]
        for camera_name in self.camkeys:
            camera_data = {}
            camera_data["egopose"] = camera_metadata["egopose"]
            camera_data["sensor_name"] = lidarkey
            camera_data["lidar2rig"] = get_rig_transform(rig_info[lidarkey], rig2sensor=False)
            camera_data["rig2cam"] = get_rig_transform(rig_info[camera_name + ".mp4"], rig2sensor=True)
            camera_data["ftheta"] = FThetaCamera.from_dict(rig_info[camera_name + ".mp4"])
            egopose_lerp, cliptbase = get_egopose_interp(camera_data["egopose"])

            metadata[camera_name] = camera_data
            ego_data[camera_name] = {
                "egopose_lerp": egopose_lerp,
                "cliptbase": cliptbase,
            }

        return metadata, ego_data, video_info, sync_to_original, calibration["rig_raw"]

    @staticmethod
    def _get_obstacles_and_camera(egopose_lerp, lidar2rig, rig2cam, t0):
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

        return world2cam

    def _get_rectify_pinhole_info(self, video_idx: int, view_idxs: List[int]):
        cfg = self.rectify_cfg

        # Set fov if randomized
        seed = zlib.adler32(f"{self.clip_names[video_idx]}-{cfg.fov_rng_seed}".encode("utf-8"))
        pinhole_fovd = np.random.RandomState(seed).randint(cfg.fov_range[0], cfg.fov_range[1] + 1)
        # target pinhole image/camera in the original resolution
        tgt_focal = cfg.width / (2.0 * math.tan(np.deg2rad(pinhole_fovd) / 2.0))
        tgt_focal = np.array([tgt_focal, tgt_focal])
        ys, xs = np.mgrid[0 : cfg.height, 0 : cfg.width]
        pixels = np.stack((xs, ys), axis=2)
        pinhole_cam = IdealPinholeCamera(
            f_x=tgt_focal[0],
            f_y=tgt_focal[1],
            width=cfg.width,
            height=cfg.height,
        )
        pinhole_intr = torch.tensor(
            [
                tgt_focal[0],
                tgt_focal[1],
                cfg.width / 2,
                cfg.height / 2,
            ]
        ).float()
        pinhole_rays = pinhole_cam.pixel2ray(pixels.reshape(-1, 2))  # hw x 3

        # Pre-compute projection for all possible cameras.
        meta_data, _, _, _, _ = self._get_meta_data(video_idx)

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
            pos_norm = torch.from_numpy(pos_norm).reshape(1, cfg.height, cfg.width, 2)
            pos_norms[view_idx] = pos_norm

        return pinhole_rays, pinhole_intr, pos_norms

    def egomotion_alpamayo_parser(self, video_idx, timestamps):
        clip_name = self.clip_names[video_idx]
        tar_path = self.tar_index[clip_name]

        ed_config = egomotion_decoder.EgoMotionDecoderConfig(decode_strategy=f"uniform_{len(timestamps)}_frame")
        with tarfile.open(tar_path, "r") as f:
            data = {}
            data["egomotion.npz"] = f.extractfile(f"{clip_name}.egomotion.npz").read()
            data["live_egomotion.npz"] = f.extractfile(f"{clip_name}.live_egomotion.npz").read()
            data["rig.json"] = f.extractfile(f"{clip_name}.rig.json").read()

        ego_data = egomotion_decoder.decode_egomotion(data, timestamps, ed_config)

        # flatten xyz and rotation into a 12-dim vector
        xyz = ego_data["ego_future_xyz"]
        rot = ego_data["ego_future_rot"]
        # traj is of shape (B, N, 12) where N is the number of futuer ego poses
        # default N = 64, 6.4 seconds, each one sampled at 0.1 sec interval.
        traj = torch.cat([xyz, rot.flatten(2, 3)], dim=-1)
        return traj

    def _read_data(
        self,
        video_idx: int,
        frame_idxs: List[int],
        view_idxs: List[int],
        data_fields: List[DataField],
    ) -> dict[DataField, Any]:
        meta_data, ego_data, video_info, sync_to_original, rig_info = self._get_meta_data(video_idx)
        video_frames, frame_idx_to_batch_idx = self._get_video_frames(
            video_idx, view_idxs, frame_idxs, sync_to_original
        )

        if self.rectify_cfg.enabled:
            cam_rays, tgt_intr, pos_norms = self._get_rectify_pinhole_info(video_idx, view_idxs)
            rays = {view_idx: cam_rays for view_idx in set(view_idxs)}
        else:
            tgt_intr, pos_norms = None, None
            rays = {}
            for view_idx in set(view_idxs):
                fisheye_cam: FThetaCamera = meta_data[self.camkeys[view_idx]]["ftheta"]
                height, width = video_frames[self.camkeys[view_idx]][0].shape[:2]
                ys, xs = np.meshgrid(
                    np.linspace(0, fisheye_cam.height - 1, height),
                    np.linspace(0, fisheye_cam.width - 1, width),
                    indexing="ij",
                )
                pixels = np.stack((xs, ys), axis=2).reshape(-1, 2)
                cam_rays = fisheye_cam.pixel2ray(pixels)[0].reshape(height, width, 3)
                rays[view_idx] = cam_rays

        output_dict = {}
        for data_field in data_fields:
            if data_field == DataField.IMAGE_RGB:
                rgb_list = []
                for frame_idx, view_idx in zip(frame_idxs, view_idxs):
                    cam_key = self.camkeys[view_idx]
                    batch_idx = frame_idx_to_batch_idx[cam_key][frame_idx]
                    img_fisheye = video_frames[cam_key][batch_idx]

                    img = img_fisheye.permute(2, 0, 1).float().unsqueeze(0)
                    if pos_norms is not None:
                        img = grid_sample(
                            img,
                            pos_norms[view_idx],
                            mode="bilinear",
                            align_corners=False,
                        )
                    img = img[0].float() / 255.0
                    rgb_list.append(img)
                output_dict[data_field] = torch.stack(rgb_list)

            elif data_field == DataField.CAMERA_C2W_TRANSFORM:
                c2w_list = []
                for frame_idx, view_idx in zip(frame_idxs, view_idxs):
                    cam_key = self.camkeys[view_idx]
                    t0_stamp = video_info[cam_key][sync_to_original[cam_key][frame_idx]]["timestamp"]
                    t0_val = 1e-6 * (t0_stamp - ego_data[cam_key]["cliptbase"])

                    world2cam = self._get_obstacles_and_camera(
                        ego_data[cam_key]["egopose_lerp"],
                        meta_data[cam_key]["lidar2rig"],
                        meta_data[cam_key]["rig2cam"],
                        t0_val,
                    )

                    world2cam = torch.from_numpy(world2cam).float()
                    cam2world = torch.inverse(world2cam)
                    cam2world[:3, :3] = cam2world[:3, :3] @ self.ROT_FIXER
                    c2w_list.append(cam2world)

                output_dict[data_field] = torch.stack(c2w_list)

            elif data_field == DataField.CAMERA_INTRINSICS:
                assert tgt_intr is not None
                output_dict[data_field] = tgt_intr.repeat(len(frame_idxs), 1)

            elif data_field == DataField.RAY_DIRECTION:
                ray_list = []
                for view_idx in view_idxs:
                    ray_list.append(torch.from_numpy(rays[view_idx]).float())
                output_dict[data_field] = torch.stack(ray_list)
            elif data_field == DataField.TRAJECTORY:
                unique_view_idxs = list(set(view_idxs))
                view_count = view_idxs.count(unique_view_idxs[0])
                for view_idx in unique_view_idxs[1:]:
                    assert view_count == view_idxs.count(
                        view_idx
                    ), "trajectory datafield expects equal number of frames per view"

                # choose one camera timestamps to compute trajectory
                if self.trajectory_base_camera_index in view_idxs:
                    target_traj_cam_index = self.trajectory_base_camera_index
                else:
                    target_traj_cam_index = random.sample(view_idxs, 1)[0]
                timestamps = []
                for frame_idx, view_idx in zip(frame_idxs, view_idxs):
                    if view_idx == target_traj_cam_index:
                        # compute timestamps of the target base camera
                        cam_key = self.camkeys[view_idx]
                        t0_stamp = video_info[cam_key][sync_to_original[cam_key][frame_idx]]["timestamp"]
                        timestamps.append(t0_stamp)
                if len(timestamps) == 0:
                    raise ValueError("view_idxs do not contain the base camera for trajectory computation.")

                # compute trajectory for the target base camera
                output_dict[data_field] = self.egomotion_alpamayo_parser(video_idx, timestamps)
                output_dict["rig_info"] = rig_info
            else:
                raise NotImplementedError(f"Can't handle data field {data_field}")

        return output_dict
