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

import glob
import io
import json
import os
import pickle
import random
import shutil
import tarfile
import tempfile
import time
import uuid
from typing import Any, List

import av
import dataverse.utils.alpamayo.egomotion_decoder as egomotion_decoder
import numpy as np
import pandas as pd
import torch
import tqdm
from einops import rearrange
from lru import LRU
from omegaconf import DictConfig
from platformdirs import user_cache_path

from .alpamayo_v2 import AlpamayoV2
from .base import DataField


class CosmosAV(AlpamayoV2):
    def __init__(
        self,
        tar_dirs: list[str],
        uuid_dirs: str,
        t5_dirs: str,
        probe_tar: bool,
        rectify: DictConfig,
        camkeys: list[str] = ["camera_front_wide_120fov"],
        decode_traj: bool = False,
        use_hq_data: bool = False,
        use_non_mb_data: bool = False,
        tar_cache_path: str = "av22_tar_index_full.json",
        t5_cache_path: str = "av22_qwen_t5_index_full.json",
    ):
        super().__init__(tar_dirs, probe_tar, rectify, camkeys, tar_cache_path)
        self.camkeys = camkeys
        self.trajectory_base_camera_index = self.camkeys.index("camera_front_wide_120fov")
        self.tar_dirs = tar_dirs
        self.uuid_dirs = uuid_dirs
        self.t5_dirs = t5_dirs
        self.rectify_cfg = rectify
        self.probe_tar = probe_tar
        self.decode_traj = decode_traj
        self.tar_cache_path = tar_cache_path
        self.t5_cache_path = t5_cache_path
        if use_hq_data:
            self.t5_index = self._buiild_qwen_t5_dirs()
        else:
            self.t5_index = self._buiild_t5_dirs()  # build clip name based on t5 index
        clip_names = sorted(list(self.t5_index.keys()))
        self.video_loader_cache = LRU(self.MAX_CACHED_VIDEOS)
        self.meta_data_cache = LRU(self.MAX_CACHED_VIDEOS)
        # Mapping from clip_name to tar file, remove the missing index
        self.tar_index = self._build_tar_index()
        missing_keys = [key for key in clip_names if key not in self.tar_index]
        self.clip_names = [key for key in clip_names if key in self.tar_index]
        print("Missing Clip number: {}".format(len(missing_keys)))
        print(f"Total number of clips: {len(self.clip_names)}")
        # Filter out unwanted data
        if use_non_mb_data:
            non_mb_clip = self._build_non_mb_data_idx()
        if use_non_mb_data:
            self.clip_names = [key for key in self.clip_names if key in non_mb_clip]
            print(f"Totoal number of clips after filtering: {len(self.clip_names)}")

    def _build_non_mb_data_idx(self):
        """
        Builds an index of non-MB clip IDs using a cached version if available.

        Returns:
            set: A set of clip IDs that are not associated with the "mb" partner.
        """
        clip_id_filtering_path = user_cache_path("dataverse") / self.tar_cache_path.replace(".json", "_non_mb.json")
        clip_id_filtering_path.parent.mkdir(parents=True, exist_ok=True)

        if clip_id_filtering_path.exists():
            with clip_id_filtering_path.open("r") as f:
                keep = json.load(f)
        else:
            clip_id_filtering_parquet_path = user_cache_path("dataverse") / "query_clip_id_filtering.parquet"
            df = pd.read_parquet(clip_id_filtering_parquet_path)
            keep = set()
            for i, row in df.iterrows():
                if row.partner == "mb":
                    continue
                keep.add(row['id'])

            with clip_id_filtering_path.open("w") as f:
                json.dump(list(keep), f)
        return set(keep)

    def _buiild_t5_dirs(self):
        """
        Builds the t5 index from cached mapping files.
        Returns:
            dict: The built t5 index dictionary.
        """
        t5_index_path = user_cache_path("dataverse") / self.t5_cache_path
        t5_index_path.parent.mkdir(parents=True, exist_ok=True)
        if t5_index_path.exists():
            with t5_index_path.open("r") as f:
                t5_index = json.load(f)
        else:
            t5_index = dict()

            for file in tqdm.tqdm(glob.glob(self.uuid_dirs + "*.parquet", recursive=True)):
                cache_file_path = os.path.join(
                    file.replace(".parquet", "").replace(self.uuid_dirs, self.t5_dirs),
                    f"{file.replace('.parquet', '').split('/')[-1]}_mapping.json",
                )

                if os.path.exists(cache_file_path):
                    with open(cache_file_path, "r") as f:
                        curr_t5_index = json.load(f)
                        for key, value in curr_t5_index.items():
                            clip_id, camera_id = key.split(".")

                            if os.path.exists(value.replace(".tar", ".json")):
                                if clip_id not in t5_index:
                                    t5_index[clip_id] = []
                                t5_index[clip_id].append(value.split(self.t5_dirs)[-1])
                else:
                    print(f"{cache_file_path} missing")

            for key, value in t5_index.items():
                t5_index[key] = list(set(value))
            with t5_index_path.open("w") as f:
                json.dump(t5_index, f)
                print(f"**** Cache t5_index to {t5_index_path}")
        return t5_index

    def _buiild_qwen_t5_dirs(self):
        """
        Builds the t5 index using qwen from cached mapping files.
        Returns:
            dict: The built t5 index dictionary.
        """
        t5_index_path = user_cache_path("dataverse") / self.t5_cache_path
        t5_index_path.parent.mkdir(parents=True, exist_ok=True)
        if t5_index_path.exists():
            with t5_index_path.open("r") as f:
                t5_index = json.load(f)
        else:
            t5_index = dict()
            for chunk_id in tqdm.tqdm(os.listdir(self.t5_dirs)):
                cache_file_path = os.path.join(self.t5_dirs, chunk_id, "mapping.json")
                if os.path.exists(cache_file_path):
                    with open(cache_file_path, "r") as f:
                        curr_t5_index = json.load(f)
                        for key, value in curr_t5_index.items():
                            clip_id, camera_id = key.split(".")

                            if os.path.exists(value.replace(".tar", ".json")):
                                if clip_id not in t5_index:
                                    t5_index[clip_id] = []
                                t5_index[clip_id].append(value.split(self.t5_dirs)[-1])
                else:
                    print(f"{cache_file_path} missing")

            for key, value in t5_index.items():
                t5_index[key] = list(set(value))
            with t5_index_path.open("w") as f:
                json.dump(t5_index, f)
                print(f"**** Cache t5_index to {t5_index_path}")

        return t5_index

    def _get_video_frames(
        self,
        video_idx: int,
        view_idxs: list[int],
        frame_idxs: list[int],
        sync_to_original: dict,
    ):

        # rank = distributed.get_rank()

        clip_name = self.clip_names[video_idx]
        tar_path = self.tar_index[clip_name]

        # Copy tar file to a temporary folder with retry logic
        retries = 3
        temp_dir = tempfile.mkdtemp()
        random_base_name = f"{uuid.uuid4().hex}.tar"
        temp_tar_path = os.path.join(temp_dir, random_base_name)
        for attempt in range(retries):
            try:
                shutil.copy(tar_path, temp_tar_path)
                break
            except Exception as e:
                print(f"Failed to copy tar file {tar_path} after {retries} attempts: {e}")
                if attempt == retries - 1:
                    shutil.rmtree(temp_dir)
                    raise RuntimeError(f"Failed to copy tar file {tar_path} after {retries} attempts: {e}. Abort")
                time.sleep(1)
        video_frames = {}
        frame_idx_to_batch_idx = {}
        repeat_happend = {}
        try:
            with tarfile.open(temp_tar_path, "r") as f:
                for key in self.camkeys:
                    target_frame_idxs = []
                    for view_idx, frame_idx in zip(view_idxs, frame_idxs):
                        if self.camkeys[view_idx] == key:
                            target_frame_idxs.append(frame_idx)

                    if len(target_frame_idxs) == 0:
                        break

                    diff = np.diff(np.array(target_frame_idxs))
                    assert np.all(diff > 0), "frame_idxs are given as non-increasing list"
                    fp = f.extractfile(f"{clip_name}.{key}.mp4")
                    assert fp is not None, f"Video {clip_name}.{key}.mp4 not found"
                    vid = io.BytesIO(fp.read())
                    vid.seek(0)
                    with av.open(vid) as input_container:
                        average_fps = input_container.streams.video[0].average_rate
                        time_base = input_container.streams.video[0].time_base
                        average_frame_duration = int(1 / average_fps / time_base)
                        frames = []
                        frame_iterator = input_container.decode(video=0)
                        cur_frame_idx_to_batch_idx = {}
                        prev_target_pts = -1
                        repeat_happend_for_key = 0
                        for batch_idx, target_frame_number in enumerate(target_frame_idxs):
                            adjusted_target_frame_number = sync_to_original[key][target_frame_number]
                            target_pts = adjusted_target_frame_number * average_frame_duration
                            if prev_target_pts == target_pts:
                                frames.append(frames[-1])
                                repeat_happend_for_key += 1
                                cur_frame_idx_to_batch_idx[target_frame_number] = batch_idx
                                continue
                            for frame in frame_iterator:
                                # find the frame
                                if frame.pts == target_pts:
                                    prev_target_pts = target_pts
                                    break
                            cur_frame_idx_to_batch_idx[target_frame_number] = batch_idx
                            frames.append(torch.as_tensor(frame.to_rgb().to_ndarray()))
                        video_frames[key] = torch.stack(frames)
                        frame_idx_to_batch_idx[key] = cur_frame_idx_to_batch_idx
                        repeat_happend[key] = repeat_happend_for_key
                        del frame_iterator
                    vid.close()
                    fp.close()
                    del vid

        finally:
            if os.path.exists(temp_tar_path):
                os.remove(temp_tar_path)
            shutil.rmtree(temp_dir)

        return video_frames, frame_idx_to_batch_idx, repeat_happend

    def _read_t5_and_meta(self, video_idx: int, view_idxs: List[int]) -> List[dict[str, Any]]:
        clip_name = self.clip_names[video_idx]
        data = {}

        for item in self.t5_index[clip_name]:
            meta_data_file = os.path.join(self.t5_dirs, item.replace(".tar", ".json"))
            with open(meta_data_file, "r") as f:
                meta_data = json.load(f)
            if clip_name in meta_data:
                for view_idx in view_idxs:
                    if str(view_idx) in meta_data[clip_name]:
                        data[int(view_idx)] = {}
                        data[int(view_idx)]['meta_data'] = meta_data[clip_name][str(view_idx)]

                        tar_file_path = os.path.join(self.t5_dirs, item)
                        retries = 3
                        temp_dir = tempfile.mkdtemp()

                        random_base_name = f"{uuid.uuid4().hex}.tar"
                        temp_tar_file_path = os.path.join(temp_dir, random_base_name)

                        # Copy the tar file to a temporary location with retries
                        for attempt in range(retries):
                            try:
                                shutil.copy(tar_file_path, temp_tar_file_path)
                                break
                            except Exception as e:
                                if attempt == retries - 1:
                                    shutil.rmtree(temp_dir)
                                    raise RuntimeError(f"Failed to copy tar file after {retries} attempts: {e}")
                                time.sleep(1)

                        try:
                            # Open the tar file from the temporary location
                            with tarfile.open(temp_tar_file_path, "r") as f:
                                file_buf = f.extractfile(f"{clip_name}.{self.camkeys[view_idx]}.bin")
                                if file_buf is None:
                                    raise FileNotFoundError(
                                        f"File {clip_name}.{self.camkeys[view_idx]}.bin not found in {item}"
                                    )

                                with io.BytesIO(file_buf.read()) as fp:
                                    data[int(view_idx)]['T5'] = pickle.load(fp)

                                file_buf.close()

                        finally:
                            # Ensure temporary tar file and directory are cleaned up
                            if os.path.exists(temp_tar_file_path):
                                os.remove(temp_tar_file_path)
                            shutil.rmtree(temp_dir)

        assert len(data) != 0, "No data was loaded from the T5 index."
        return data

    def _get_basic_meta_data(self, video_idx: int):
        if video_idx in self.meta_data_cache:
            return self.meta_data_cache[video_idx]

        clip_name = self.clip_names[video_idx]
        tar_path = self.tar_index[clip_name]
        video_info = {}

        retries = 3
        temp_dir = tempfile.mkdtemp()

        random_base_name = f"{uuid.uuid4().hex}.tar"
        temp_tar_path = os.path.join(temp_dir, random_base_name)

        for attempt in range(retries):
            try:
                shutil.copy(tar_path, temp_tar_path)
                break
            except Exception as e:
                if attempt == retries - 1:
                    shutil.rmtree(temp_dir)
                    raise RuntimeError(f"Failed to copy tar file after {retries} attempts: {e}")
                time.sleep(1)

        try:
            # Open the tar file from the temporary location
            with tarfile.open(temp_tar_path, "r") as f:
                for key in self.camkeys:
                    fp = f.extractfile(f"{clip_name}.{key}.json")
                    if fp is None:
                        raise FileNotFoundError(f"Metadata file {clip_name}.{key}.json not found in {tar_path}")

                    vidinfo = json.load(fp)
                    video_info[key] = vidinfo
                    fp.close()

            # Synchronize timestamps
            min_t, max_t = -np.inf, np.inf
            original_timestamps = {}
            for key in self.camkeys:
                timestamps = np.array([info["timestamp"] for info in video_info[key]])
                min_t = max(min_t, timestamps.min())
                max_t = min(max_t, timestamps.max())
                original_timestamps[key] = timestamps
            ref_timestamps = original_timestamps[self.camkeys[0]]
            ref_timestamps = ref_timestamps[ref_timestamps >= min_t]
            ref_timestamps = ref_timestamps[ref_timestamps <= max_t]
            sync_to_original = {}
            for key in self.camkeys:
                sync_to_original[key] = np.searchsorted(original_timestamps[key], ref_timestamps)

        finally:
            # Ensure temporary tar file and directory are deleted
            if os.path.exists(temp_tar_path):
                os.remove(temp_tar_path)
            shutil.rmtree(temp_dir)

        return video_info, sync_to_original

    def egomotion_alpamayo_parser_initial_frame(self, video_idx, timestamps):
        clip_name = self.clip_names[video_idx]
        tar_path = self.tar_index[clip_name]

        ed_config = egomotion_decoder.EgoMotionDecoderConfig(
            decode_strategy="at_0_frame", prediction_start_offset_range=[0.0, 0.0]
        )
        with tarfile.open(tar_path, "r") as f:
            data = {}
            data["egomotion.npz"] = f.extractfile(f"{clip_name}.egomotion.npz").read()
            data["live_egomotion.npz"] = f.extractfile(f"{clip_name}.live_egomotion.npz").read()
            calibration = self._extract_calibration_from_sample(f.extractfile(f"{clip_name}.rig.json"))
            data["rig.json"] = f.extractfile(f"{clip_name}.rig.json").read()

        ego_data = egomotion_decoder.decode_egomotion(data, timestamps, ed_config)

        # flatten xyz and rotation into a 12-dim vector
        xyz = ego_data["ego_future_xyz"]
        rot = ego_data["ego_future_rot"]
        # traj is of shape (B, N, 12) where N is the number of futuer ego poses
        # default N = 64, 6.4 seconds, each one sampled at 0.1 sec interval.
        traj = torch.cat([xyz, rot.flatten(2, 3)], dim=-1)
        return traj, calibration["rig_raw"]

    def _read_raw_video(
        self,
        video_idx: int,
        frame_idxs: List[int],
        view_idxs: List[int],
        data_fields: List[DataField],
    ) -> dict[DataField, Any]:

        video_info, sync_to_original = self._get_basic_meta_data(video_idx)
        video_frames, frame_idx_to_batch_idx, repeat_happend = self._get_video_frames(
            video_idx, view_idxs, frame_idxs, sync_to_original
        )
        output_dict = {}
        for data_field in data_fields:
            if data_field == DataField.IMAGE_RGB:
                rgb_list = []
                num_repeated_frames = []
                cur_view_idx = -1
                for frame_idx, view_idx in zip(frame_idxs, view_idxs):
                    cam_key = self.camkeys[view_idx]
                    batch_idx = frame_idx_to_batch_idx[cam_key][frame_idx]
                    rgb_list.append(video_frames[cam_key][batch_idx].permute(2, 0, 1).float() / 255.0)
                    if cur_view_idx != view_idx:
                        num_repeated_frames.append(repeat_happend[cam_key])
                        cur_view_idx = view_idx
                output_dict[data_field] = torch.stack(rgb_list)
                output_dict['num_repeated_frames'] = torch.FloatTensor(num_repeated_frames)
            elif self.decode_traj and data_field == DataField.TRAJECTORY:
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
                traj, rig_info = self.egomotion_alpamayo_parser_initial_frame(video_idx, timestamps)

                traj = traj[0][..., :3]  # use xyz
                output_dict[data_field] = rearrange(traj, 't c -> (t c)')
                output_dict['rig_info'] = rig_info
            else:
                raise NotImplementedError(f"Can't handle data field {data_field}")
        return output_dict
