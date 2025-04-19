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
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.spatial.transform as spt
import torch
from dataverse.utils import logger
from dataverse.utils.alpamayo import transformation
from dataverse.utils.alpamayo.camera import Quaternion
from scipy.interpolate import interp1d

_VALID_DECODE_STRATEGY_PATTERN = r"^(random|uniform|at)_(-?\d+)_frame$"


class EgoMotionData(TypedDict):
    """Encompasses all required information to interpolate ego motion data."""

    tmin: int
    tmax: int
    tparam: npt.NDArray[np.float32]
    xyzs: npt.NDArray[np.float32]
    quats: npt.NDArray[np.float32]


def parse_egopose_data(egopose_info: dict):
    """Minimal parsing here.

    Args:
        egopose_info: a dict containing the raw egopose data.

    Returns:
        info (dict): a dict of numpy arrays with dtype object
    """
    N, C = egopose_info["labels_data"].shape
    (K,) = egopose_info["labels_keys"].shape
    assert K == C, f"{K} {C}"

    info = {egopose_info["labels_keys"][ki]: egopose_info["labels_data"][:, ki] for ki in range(K)}

    # make sure sorted by time
    assert np.all(0 < info["timestamp"][1:] - info["timestamp"][:-1]), info["timestamp"]

    return info


def adjust_orientation(
    vals: npt.NDArray[np.float32] | torch.Tensor,
) -> npt.NDArray[np.float32] | torch.Tensor:
    """Adjusts the orientation of the quaternions.

    Adjusts the orientation of the quaternions so that the dot product
    between vals[i] and vals[i+1] is non-negative.

    Args:
        vals (np.array or torch.tensor): (N, C)

    Returns:
        vals (np.array or torch.tensor): (N, C) adjusted quaternions
    """
    N, C = vals.shape
    if isinstance(vals, torch.Tensor):
        signs = torch.ones(N, dtype=vals.dtype, device=vals.device)
        signs[1:] = torch.where(0 <= (vals[:-1] * vals[1:]).sum(dim=1), 1.0, -1.0)
        signs = torch.cumprod(signs, dim=0)

        return vals * signs.reshape((N, 1))

    else:
        signs = np.ones(N, dtype=vals.dtype)
        signs[1:] = np.where(0 <= (vals[:-1] * vals[1:]).sum(axis=1), 1.0, -1.0)
        signs = np.cumprod(signs)

        return vals * signs.reshape((N, 1))


def preprocess_egopose(poses: dict) -> EgoMotionData:
    """Converts the poses to for interpolation.

    The dtype of all the inputs to the interpolator is float32.
    TODO: instead of a linear interpolator for quaternions,
    it'd be better to do slerp.

    Args:
        poses (dict): a dict containing the raw egopose data.

    Returns:
        A dictionary containing the following
            tmin: int, the start time of the egopose data in microseconds
            tmax: int, the end time of the egopose data in microseconds
            tparam: list of floats, the relative (starting from 0)
                timestamps of the egopose data in seconds
            xyzs: list of lists of floats, the x,y,z position of the egopose
            quats: list of lists of floats, the quaternion orientation of
                the egopose
    """
    # bounds of the interpolator as timestamps (ints)
    tmin = poses["timestamp"][0]
    tmax = poses["timestamp"][-1]

    # convert timestamps to float32 only after subtracting off tmin and
    # converting from microseconds to seconds
    tparam = (1e-6 * (poses["timestamp"] - tmin)).astype(np.float32)

    # prep x,y,z
    # convert to float64, subtract off mean, convert to float32
    xyzs = np.stack(
        (
            poses["x"].astype(np.float64),
            poses["y"].astype(np.float64),
            poses["z"].astype(np.float64),
        ),
        1,
    )
    xyzs = xyzs - xyzs.mean(axis=0, keepdims=True)
    xyzs = xyzs.astype(np.float32)

    # prep quaternions
    # parse directly as float32
    quats = np.stack(
        (
            poses["qw"].astype(np.float32),
            poses["qx"].astype(np.float32),
            poses["qy"].astype(np.float32),
            poses["qz"].astype(np.float32),
        ),
        1,
    )

    # prep quaternions for interpolation https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L847
    # make sure normalized
    quat_norm = np.linalg.norm(quats, axis=1)
    EPS = 1e-3
    if not np.all(np.abs(quat_norm - 1.0) < EPS):
        raise ValueError(f"Raw pose quaternions are too far from normalized; {quat_norm=}")
    # adjust signs so that sequential dot product is always positive
    quats = adjust_orientation(quats / quat_norm[:, None])

    return EgoMotionData(
        tmin=tmin,
        tmax=tmax,
        tparam=tparam,
        xyzs=xyzs,
        quats=quats,
    )


def load_dm_egopose(sample: dict) -> dict | None:
    """Load egopose from raw tar files.

    Args:
        sample: A dictionary containing the raw data to be decoded.

    Returns:
        A dictionary containing the following keys:
            tmin: int, the start time of the egopose data in microseconds
            tmax: int, the end time of the egopose data in microseconds
            tparam: list of floats, the relative (starting from 0)
                timestamps of the egopose data in seconds
            xyzs: list of lists of floats, the x,y,z position of the egopose
            quats: list of lists of floats, the quaternion orientation of
                the egopose
    """
    ego_info = pd.read_parquet(io.BytesIO(sample["egomotion_estimate.parquet"]))["egomotion_estimate"]
    ego_ts = pd.read_parquet(io.BytesIO(sample["clip.parquet"]))["key"][0]["time_range"]
    ego_start_ts = ego_ts["start_micros"]
    ego_end_ts = ego_ts["end_micros"]

    x = np.array([item["location"]["x"] for item in ego_info])
    y = np.array([item["location"]["y"] for item in ego_info])
    z = np.array([item["location"]["z"] for item in ego_info])

    qx = np.array([item["orientation"]["x"] for item in ego_info])
    qy = np.array([item["orientation"]["y"] for item in ego_info])
    qz = np.array([item["orientation"]["z"] for item in ego_info])
    qw = np.array([item["orientation"]["w"] for item in ego_info])

    timestamps = np.linspace(ego_start_ts, ego_end_ts, x.shape[0], endpoint=True)
    timestamps = timestamps.astype(int)

    egopose_parsed = {}
    egopose_parsed["coordinate_frame"] = "rig"
    egopose_parsed["timestamp"] = timestamps
    egopose_parsed["x"] = x
    egopose_parsed["y"] = y
    egopose_parsed["z"] = z
    egopose_parsed["qw"] = qw
    egopose_parsed["qx"] = qx
    egopose_parsed["qy"] = qy
    egopose_parsed["qz"] = qz

    ego_lerp_inp = preprocess_egopose(egopose_parsed)
    return {
        "tmin": ego_lerp_inp["tmin"],
        "tmax": ego_lerp_inp["tmax"],
        "tparam": ego_lerp_inp["tparam"].tolist(),
        "xyzs": ego_lerp_inp["xyzs"].tolist(),
        "quats": ego_lerp_inp["quats"].tolist(),
    }


def load_egopose(sample: dict, live: bool = False, min_fps: int = 5) -> EgoMotionData | None:
    """Load egopose from raw tar files.

    If we have fewer than 20 seconds, return None

    Args:
        sample: A dictionary containing the raw data to be decoded.
        live: if `True`, the "live" (estimated online) egomotion will be loaded
            from the sample, otherwise the ground truth (optimized offline)
            egomotion will be loaded.
        min_fps: The minimum FPS of the egomotion data.

    Returns:
        A dictionary containing the following keys:
            tmin: int, the start time of the egopose data in microseconds
            tmax: int, the end time of the egopose data in microseconds
            tparam: list of floats, the relative (starting from 0)
                timestamps of the egopose data in seconds
            xyzs: list of lists of floats, the x,y,z position of the egopose
            quats: list of lists of floats, the quaternion orientation of
                the egopose
    """
    pose_info = np.load(
        io.BytesIO(sample["live_egomotion.npz" if live else "egomotion.npz"]),
        allow_pickle=True,
    )
    egopose_parsed = parse_egopose_data(pose_info)

    TMIN = 20.0  # seconds
    egopose_span = 1e-6 * (egopose_parsed["timestamp"][-1] - egopose_parsed["timestamp"][0])
    if egopose_span < TMIN:
        logger.warning(f"Insufficient egomotion data for this clip: {egopose_span=}")
        return None

    # Check the FPS of egomotion data
    delta = 1e-6 * (egopose_parsed["timestamp"][1:] - egopose_parsed["timestamp"][:-1])
    max_delta = 1.0 / min_fps
    if not np.all(delta < max_delta):
        logger.warning(f"Egomotion data does not meet frequency requirement: {max(delta)=}")
        return None

    ego_lerp_inp = preprocess_egopose(egopose_parsed)

    coordinate_frame = egopose_parsed["coordinate_frame"][0].replace("_", ":")
    if coordinate_frame != "rig":
        # The logged egomotion is tracking a sensor's coordinate frame (e.g., the pose
        # of the lidar) that is not the rig frame (origin at the rear axle center projected
        # to ground, oriented front-left-up with respect to the vehicle's body).
        # Here we use the "rig.json" to convert the sensor's pose to the rig's pose.
        sensors = transformation.parse_rig_sensors_from_dict(json.loads(sample["rig.json"]))
        if coordinate_frame not in sensors:
            raise ValueError(f"Egomotion {coordinate_frame=} not found in rig.json.")
        sensor_to_rig = transformation.sensor_to_rig(sensors[coordinate_frame])
        sensor_to_world_rots = spt.Rotation.from_quat(ego_lerp_inp["quats"], scalar_first=True)
        # We derive rig_to_world by composing sensor_to_world and rig_to_sensor;
        # with 4x4 rigid transformation matrices this would be:
        # rig_to_world = sensor_to_world @ np.linalg.inv(sensor_to_rig)
        rig_to_world_rots = sensor_to_world_rots * spt.Rotation.from_matrix(sensor_to_rig[:3, :3].T)
        ego_lerp_inp["xyzs"] = ego_lerp_inp["xyzs"] - rig_to_world_rots.apply(sensor_to_rig[:3, 3])
        ego_lerp_inp["quats"] = adjust_orientation(rig_to_world_rots.as_quat(scalar_first=True))

    return EgoMotionData(
        tmin=ego_lerp_inp["tmin"],
        tmax=ego_lerp_inp["tmax"],
        tparam=ego_lerp_inp["tparam"].tolist(),
        xyzs=ego_lerp_inp["xyzs"].tolist(),
        quats=ego_lerp_inp["quats"].tolist(),
    )


class EgoPoseInterp:
    """Interpolates egopose data."""

    @classmethod
    def load_from_raw_dict(
        cls,
        sample: Mapping[str, Any],
        source: str = "online",
    ):
        """Init from a decoded data sample.

        Args:
            sample: a decoded data sample.
            source: the source of the egopose data, either "online" or "offline" or "deepmap".
        """
        if source == "online":
            ego_pose = load_egopose(sample, True)
        elif source == "offline":
            ego_pose = load_egopose(sample, False)
        elif source == "deepmap":
            ego_pose = load_dm_egopose(sample)
        else:
            raise ValueError(f"Invalid source: {source}")
        if ego_pose is None:
            raise ValueError("No ego pose available.")
        return cls(
            tmin=ego_pose["tmin"],
            tmax=ego_pose["tmax"],
            tparam=ego_pose["tparam"],
            xyzs=ego_pose["xyzs"],
            quats=ego_pose["quats"],
        )

    def __init__(
        self,
        tmin: int,
        tmax: int,
        tparam: npt.NDArray[np.float32],
        xyzs: npt.NDArray[np.float32],
        quats: npt.NDArray[np.float32],
    ):
        """Initialize the interpolator.

        Args:
            tmin: int, the start time of the egopose data in microseconds
            tmax: int, the end time of the egopose data in microseconds
            tparam: list of floats, the relative (starting from 0)
                timestamps of the egopose data in seconds
            xyzs: list of lists of floats, the x,y,z position of the egopose
            quats: list of lists of floats, the quaternion orientation of
                the egopose
        """
        self.tmin = tmin
        self.tmax = tmax

        self.interp = interp1d(
            tparam,
            np.concatenate((xyzs, quats), 1),
            kind="linear",
            axis=0,
            copy=False,
            bounds_error=True,
            assume_sorted=True,
        )

    def convert_tstamp(self, tstamp: int | npt.NDArray[np.int64]):
        """Converts the absolute timestamp (microsecond) to relative (s)."""
        return 1e-6 * (tstamp - self.tmin)

    def __call__(
        self, t: npt.NDArray[np.float32], is_microsecond: bool = False
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Interpolate pose for t in seconds or microsecond."""
        EPS = 1e-5
        if is_microsecond:
            t = self.convert_tstamp(t)

        out = self.interp(t)
        xyzs = out[..., :3]
        quats = out[..., 3:]

        # normalize quats
        norm = np.linalg.norm(quats, axis=-1, keepdims=True)
        assert np.all(EPS < norm), norm
        quats = quats / norm

        return xyzs, quats


def _check_valid_tstamp_0(
    tstamp_0: int,
    prediction_start_offset_range: list[float],
    ego_history_tvals: np.ndarray,
    ego_future_tvals: np.ndarray,
    history_ego_lerp_inp: dict,
    future_ego_lerp_inp: dict,
) -> bool:
    """Check if the tstamp_0 is valid for the given prediction_start_offset_range."""
    min_random_offset = int(prediction_start_offset_range[0] * 1e6)
    max_random_offset = int(prediction_start_offset_range[1] * 1e6)
    return (
        tstamp_0 + min_random_offset + int(ego_history_tvals[0] * 1e6) >= history_ego_lerp_inp["tmin"]
        and tstamp_0 + max_random_offset + int(ego_history_tvals[-1] * 1e6) <= history_ego_lerp_inp["tmax"]
        and tstamp_0 + min_random_offset + int(ego_future_tvals[0] * 1e6) >= future_ego_lerp_inp["tmin"]
        and tstamp_0 + max_random_offset + int(ego_future_tvals[-1] * 1e6) <= future_ego_lerp_inp["tmax"]
    )


def interpolate_egopose(
    ego_lerp_inp: dict | None,
    live_ego_lerp_inp: dict | None,
    prediction_start_offset_range: tuple[float, float],
    ego_history_tvals: list[float],
    ego_future_tvals: list[float],
    base_timestamps: list[int],
    decode_strategy: str,
    num_route_points: int = 32,
) -> dict:
    """Interpolates the egopose data starting at the certain timestamps.

    Taking the raw egopose data (ego_lerp_inp), we first decide starting time
    indices (`ego_t0_frame_idx`) and then interpolate the egopose at the timestamps
    base_timestamps[ego_t0_frame_idx] + trajectory_tvals.

    Args:
        ego_lerp_inp (dict): A dictionary containing the complete egopose data (gt).
        live_ego_lerp_inp (dict): A dictionary containing the complete live egopose data (live).
        prediction_start_offset_range (tuple): range of possible relative
            time offsets from last input image frame to prediction start time.
        ego_history_tvals (list): the notional relative timestamps (i.e. t0 = 0.0s) of the
            ego history data in seconds.
        ego_future_tvals (list): the notional relative timestamps (i.e. t0 = 0.0s) of the
            ego future data (i.e., ground truth for prediction) in seconds.
        base_timestamps (list): the timestamps of the base frame
            in microseconds (assume sorted).
        decode_strategy: the strategy defines at which frames to decode the egopose.
            valid strategies are:
                - `random_N_frame`: randomly sample N frames from the available frames.
                - `uniform_N_frame`: sample N uniformly spaced frames from the available frames.
                - `at_N_frame`: sample the N-th frame from the available frames.
        num_route_points: The number of points from egopose data to select as a temporary (TODO)
            stand-in for route information.

    Returns:
        A dictionary containing the following
            ego_available (bool): whether the egopose data is available
            ego_t0 (torch.tensor): (num_sample,)
                the absolute start time of the egopose data in microseconds
            ego_t0_relative (torch.tensor): (num_sample,) the relative start time of the egopose
                data in seconds it is normalized to the first timestamp of the base_timestamps
            ego_t0_frame_idx (torch.tensor): (num_sample,) the frame index of the base frame
            prediction_start_offset (torch.tensor): (num_sample,)
                the prediction start time offset
            ego_history_tvals (torch.tensor): (Th,)
                time in seconds corresponding to each position
            ego_history_xyz (torch.tensor): (num_sample,Th,3)
                the ego history (live) x,y,z positions
            ego_history_rot (torch.tensor): (num_sample,Th,3,3)
                the ego history (live) orientations as 3x3 matrices
            ego_future_tvals (torch.tensor): (Tf,)
                time in seconds corresponding to each position
            ego_future_xyz (torch.tensor): (num_sample,Tf,3)
                the ego future (gt) x,y,z positions
            ego_future_rot (torch.tensor): (num_sample,Tf,3,3)
                the ego future (gt) orientations as 3x3 matrices
            route_xy (torch.tensor): (num_sample, num_route_points, 3)
                the route x,y positions (from ego gt)
    """
    # raise an error when we don't have ego pose data for the clip
    if ego_lerp_inp is None:
        raise ValueError("Invalid ego pose data for this clip.")

    if not all(x <= y for x, y in zip(base_timestamps[:-1], base_timestamps[1:])):
        raise ValueError("base_timestamps is not sorted.")

    ego_lerp = EgoPoseInterp(
        tmin=ego_lerp_inp["tmin"],
        tmax=ego_lerp_inp["tmax"],
        tparam=ego_lerp_inp["tparam"],
        xyzs=ego_lerp_inp["xyzs"],
        quats=ego_lerp_inp["quats"],
    )

    if live_ego_lerp_inp is None:
        logger.warning("Using ego_lerp_inp in place of live_ego_lerp_inp (= None).")
        live_ego_lerp_inp = ego_lerp_inp
        live_ego_lerp = ego_lerp
    else:
        live_ego_lerp = EgoPoseInterp(
            tmin=live_ego_lerp_inp["tmin"],
            tmax=live_ego_lerp_inp["tmax"],
            tparam=live_ego_lerp_inp["tparam"],
            xyzs=live_ego_lerp_inp["xyzs"],
            quats=live_ego_lerp_inp["quats"],
        )

    ego_history_tvals = np.array(ego_history_tvals, dtype=np.float32)
    ego_future_tvals = np.array(ego_future_tvals, dtype=np.float32)

    match = re.match(_VALID_DECODE_STRATEGY_PATTERN, decode_strategy)
    if match is None:
        raise NotImplementedError(f"Decode strategy {decode_strategy} not implemented.")
    strategy_type = match.group(1)
    if strategy_type in {"random", "uniform"}:
        # We work in timestamps (microseconds) to ensure temporal consistency between ego_lerp
        # and live_ego_lerp (which have different notions of time t relative to their respective
        # first timestamps).
        valid_frame_indices = [
            ori
            for ori, _tstamp_0 in enumerate(base_timestamps)
            if _check_valid_tstamp_0(
                tstamp_0=_tstamp_0,
                prediction_start_offset_range=prediction_start_offset_range,
                ego_history_tvals=ego_history_tvals,
                ego_future_tvals=ego_future_tvals,
                history_ego_lerp_inp=live_ego_lerp_inp,
                future_ego_lerp_inp=ego_lerp_inp,
            )
        ]
        if len(valid_frame_indices) == 0:
            raise ValueError(
                "Insufficient ego pose data to fit history + future "
                f"with maximum start offset {prediction_start_offset_range[1]}."
            )
        num_frames = int(match.group(2))
        if num_frames > len(valid_frame_indices):
            raise ValueError(
                f"Requested {num_frames} frames, but only {len(valid_frame_indices)} " "frames are available."
            )
        prediction_start_offset = np.random.uniform(*prediction_start_offset_range, size=num_frames)
        if strategy_type == "random":
            # sample randomly from timestamps for which history + future steps (shifted by
            # prediction_start_offset) fit within the available ego data.
            ego_t0_frame_idx = sorted(random.sample(valid_frame_indices, k=num_frames))
        else:
            # sample uniformly from timestamps for which history + future steps (shifted by
            # prediction_start_offset) fit within the available ego data.
            # NOTE: we sample the last frame first to ensure that the last frame is included.
            _step = len(valid_frame_indices) / num_frames
            ego_t0_frame_idx = [valid_frame_indices[-(int(_step * i) + 1)] for i in range(num_frames)][::-1]
    elif strategy_type == "at":
        frame_idx = int(match.group(2))
        if frame_idx < 0:
            ego_t0_frame_idx = [len(base_timestamps) + frame_idx]
        else:
            ego_t0_frame_idx = [frame_idx]
        prediction_start_offset = np.random.uniform(*prediction_start_offset_range, size=1)

    tstamp_0 = [
        base_timestamps[_idx] + int(_start_offset * 1e6)
        for _idx, _start_offset in zip(ego_t0_frame_idx, prediction_start_offset)
    ]
    # convert prediction-relative tvals to data sample tvals
    # shape: (num_frames, num_history_steps)
    history_live_ego_lerp_tvals = (
        np.array(
            [live_ego_lerp.convert_tstamp(_tstamp_0) for _tstamp_0 in tstamp_0],
            dtype=np.float32,
        )[:, None]
        + ego_history_tvals[None, :]
    )
    # This check should only fail at "at" strategy
    if not (
        live_ego_lerp_inp["tparam"][0] <= history_live_ego_lerp_tvals[:, 0].min()
        and history_live_ego_lerp_tvals[:, -1].max() <= live_ego_lerp_inp["tparam"][-1]
    ):
        raise ValueError(
            f"data: {live_ego_lerp_inp['tparam'][0]=} to {live_ego_lerp_inp['tparam'][-1]=}, "
            f"while asking {history_live_ego_lerp_tvals[:, 0].min()=} to "
            f"{history_live_ego_lerp_tvals[:, -1].max()=}"
        )

    # shape: (num_frames, num_future_steps)
    future_future_ego_lerp_tvals = (
        np.array(
            [ego_lerp.convert_tstamp(_tstamp_0) for _tstamp_0 in tstamp_0],
            dtype=np.float32,
        )[:, None]
        + ego_future_tvals[None, :]
    )
    # This check should only fail at "at" strategy
    if not (
        ego_lerp_inp["tparam"][0] <= future_future_ego_lerp_tvals[:, 0].min()
        and future_future_ego_lerp_tvals[:, -1].max() <= ego_lerp_inp["tparam"][-1]
    ):
        raise ValueError(
            f"data: {ego_lerp_inp['tparam'][0]=} to {ego_lerp_inp['tparam'][-1]=}, "
            f"while asking {future_future_ego_lerp_tvals[:, 0].min()=} to "
            f"{future_future_ego_lerp_tvals[:, -1].max()=}"
        )

    # evaluate live pose at the history timesteps
    ego_history_xyz, ego_history_quat = live_ego_lerp(history_live_ego_lerp_tvals)
    # (num_frames, num_history_steps, 3)
    ego_history_xyz = torch.tensor(ego_history_xyz, dtype=torch.float32)
    # (num_frames, num_history_steps, 4)
    ego_history_quat = torch.tensor(ego_history_quat, dtype=torch.float32)

    # transform coordinates to the ego's body frame (according to live) at the start of prediction
    # (num_frames, 3), (num_frames, 3)
    quaternion = Quaternion()

    live_xyz0, live_quat0 = live_ego_lerp(
        np.array(
            [live_ego_lerp.convert_tstamp(_tstamp_0) for _tstamp_0 in tstamp_0],
            dtype=np.float32,
        )
    )
    live_xyz0 = torch.tensor(live_xyz0, dtype=torch.float32)
    live_inv_quat0 = quaternion.invert(torch.tensor(live_quat0, dtype=torch.float32))

    ego_history_xyz = quaternion.apply(live_inv_quat0.unsqueeze(1), ego_history_xyz - live_xyz0.unsqueeze(1))
    ego_history_quat = quaternion.product(live_inv_quat0.unsqueeze(1), ego_history_quat)

    # TODO: remove duplicated code
    # evaluate gt pose at the future timesteps
    ego_future_xyz, ego_future_quat = ego_lerp(future_future_ego_lerp_tvals)
    # (num_frames, num_future_steps, 3)
    ego_future_xyz = torch.tensor(ego_future_xyz, dtype=torch.float32)
    # (num_frames, num_future_steps, 3)
    ego_future_quat = torch.tensor(ego_future_quat, dtype=torch.float32)

    # transform coordinates to the ego's body frame (according to gt) at the start of prediction
    # (num_frames, 3), (num_frames, 3)
    xyz0, quat0 = ego_lerp(
        np.array(
            [ego_lerp.convert_tstamp(_tstamp_0) for _tstamp_0 in tstamp_0],
            dtype=np.float32,
        )
    )
    xyz0 = torch.tensor(xyz0, dtype=torch.float32)
    inv_quat0 = quaternion.invert(torch.tensor(quat0, dtype=torch.float32))
    ego_future_xyz = quaternion.apply(inv_quat0.unsqueeze(1), ego_future_xyz - xyz0.unsqueeze(1))
    ego_future_quat = quaternion.product(inv_quat0.unsqueeze(1), ego_future_quat)

    # infer ego route from gt pose and transform to ego's body frame at the start of prediction
    # TODO: update route decoding when we have access to an alternative source
    # (num_route_points, 3)
    route_points = torch.tensor(ego_lerp_inp["xyzs"])[
        np.linspace(0, len(ego_lerp_inp["xyzs"]) - 1, num_route_points, dtype=int)
    ]
    route_xy = quaternion.apply(inv_quat0.unsqueeze(1), route_points.unsqueeze(0) - xyz0.unsqueeze(1))[..., :2]

    time_base = base_timestamps[0]
    # make sure the relative_timestamp is slightly larger than original timestamp
    # because later we will sort the timestamps of ego and images
    # and we want to make sure the ego appears after the image
    ego_t0_relative = [(t0 - time_base) * 1e-6 + 1e-5 for t0 in tstamp_0]

    return {
        "ego_available": torch.tensor(True),
        "ego_t0": torch.tensor(tstamp_0),
        "ego_t0_relative": torch.tensor(ego_t0_relative),
        "ego_t0_frame_idx": torch.tensor(ego_t0_frame_idx),
        "prediction_start_offset": torch.from_numpy(prediction_start_offset).float(),
        "ego_history_tvals": torch.from_numpy(ego_history_tvals).float(),
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": quaternion.q_to_R(ego_history_quat),
        "ego_future_tvals": torch.from_numpy(ego_future_tvals).float(),
        "ego_future_xyz": ego_future_xyz,
        "ego_future_rot": quaternion.q_to_R(ego_future_quat),
        "route_xy": route_xy,
    }


@dataclass(frozen=True)
class EgoMotionDecoderConfig:
    """Configuration for the egomotion decoder.

    Attributes:
        num_history_steps: number of history steps to load, i.e., with relative timestamps
            `(-(num_history_steps - 1) * time_step, ..., -2 * time_step, -time_step, 0)`
            to the prediction start time.
        num_future_step: number of future steps to load, i.e., with relative timestamps
            `(time_step, 2 * time_step, ..., num_future_steps * time_step)`
            to the prediction start time.
        time_step: time step (in seconds) between successive egomotion poses.
        prediction_start_offset_range: min and max of possible relative time offsets from base image
            frame to prediction start time.
        force_base_frame_index: if not `None`, loaded trajectory is based from the specified
            frame index in image_frames. (TODO: Deprecate this in favor of `decode_strategy`)
        num_route_points: The number of points from egopose data to select as a temporary (TODO)
            stand-in for route information.
        decode_strategy: the strategy defines at which frames to decode the egopose.
            valid strategies are:
                - `random_N_frame`: randomly sample N frames from the available frames.
                - `uniform_N_frame`: sample N uniformly spaced frames from the available frames.
                - `at_N_frame`: sample the N-th frame from the available frames.
    """

    num_history_steps: int = 15
    num_future_steps: int = 64
    time_step: float = 0.1
    prediction_start_offset_range: tuple[float, float] = field(default_factory=lambda: (0.0, 1.5))
    force_base_frame_index: int | None = None
    num_route_points: int = 32
    decode_strategy: str = "random_1_frame"

    def __post_init__(self):
        """Make sure the config is valid."""
        if self.force_base_frame_index is not None:
            logger.warning(
                "force_base_frame_index is deprecating, using "
                "`decode_strategy=at_%d_frame` instead." % self.force_base_frame_index
            )
            self.decode_strategy = f"at_{self.force_base_frame_index}_frame"
        if not re.match(_VALID_DECODE_STRATEGY_PATTERN, self.decode_strategy):
            raise ValueError(f"Invalid decode strategy: {self.decode_strategy}")


def decode_egomotion(
    data: dict,
    base_timestamps: list[int],
    config: EgoMotionDecoderConfig,
) -> dict:
    """Decode egomotion from the data.

    Args:
        data (dict): The raw data to be decoded.
            it is assumed to contain the "egomotion.npz" (gt) and
            "live_egomotion.npz" (live) fields
        base_timestamps (list[int]): time stamps for each image frame
            in microseconds. This is used to decide the 0-th timestamp of
            the trajectory to load.
        config: EgoMotionDecoderConfig

    Returns:
        decoded_data (dict): containing the following fields:
            ego_available (bool): whether the egopose data is available
            ego_t0 (torch.tensor): (num_sample,)
                the absolute start time of the egopose data in microseconds
            ego_t0_relative (torch.tensor): (num_sample,) the relative start time of the egopose
                data in seconds it is normalized to the first timestamp of the base_timestamps
            ego_t0_frame_idx (torch.tensor): (num_sample,) the frame index of the base frame
            prediction_start_offset (torch.tensor): (num_sample,)
                the prediction start time offset
            ego_history_tvals (torch.tensor): (Th,)
                time in seconds corresponding to each position
            ego_history_xyz (torch.tensor): (num_sample,Th,3)
                the ego history (live) x,y,z positions
            ego_history_rot (torch.tensor): (num_sample,Th,3,3)
                the ego history (live) orientations as 3x3 matrices
            ego_future_tvals (torch.tensor): (Tf,)
                time in seconds corresponding to each position
            ego_future_xyz (torch.tensor): (num_sample,Tf,3)
                the ego future (gt) x,y,z positions
            ego_future_rot (torch.tensor): (num_sample,Tf,3,3)
                the ego future (gt) orientations as 3x3 matrices
            route_xy (torch.tensor): (num_sample, config.num_route_points, 3)
                the route x,y positions (from ego gt)
    """
    ego_pose = load_egopose(data, live=False)
    live_ego_pose = load_egopose(data, live=True)
    ego_history_tvals = [config.time_step * t for t in range(-config.num_history_steps + 1, 1)]
    ego_future_tvals = [config.time_step * t for t in range(1, config.num_future_steps + 1)]
    return interpolate_egopose(
        ego_lerp_inp=ego_pose,
        live_ego_lerp_inp=live_ego_pose,
        prediction_start_offset_range=config.prediction_start_offset_range,
        ego_history_tvals=ego_history_tvals,
        ego_future_tvals=ego_future_tvals,
        base_timestamps=base_timestamps,
        decode_strategy=config.decode_strategy,
        num_route_points=config.num_route_points,
    )
