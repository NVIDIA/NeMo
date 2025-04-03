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

"""This file contains sensor transformation utilities, e.g., coordinate transforms from camera or
lidar frames to the ego-vehicle rig frame.

"""

import sys
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np
import yaml
from pyquaternion import Quaternion

if TYPE_CHECKING:
    from alpamayo.data import ndas_camera_model

import dataverse.utils.alpamayo.rotation as rotation_utils


def get_sensor_to_sensor_flu(sensor: str) -> np.ndarray:
    """Compute a rotation matrix that rotates sensor to Front-Left-Up format.

    Args:
        sensor (str): sensor name.

    Returns:
        np.ndarray: the resulting rotation matrix.
    """
    if "cam" in sensor:
        rot = [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    else:
        rot = np.eye(4, dtype=np.float32)

    return np.asarray(rot, dtype=np.float32)


def parse_rig_sensors_from_dict(rig: dict) -> dict[str, dict]:
    """Parses the provided rig dict into a dictionary indexed by sensor name.

    Args:
        rig (Dict): Complete rig file as a dictionary.

    Returns:
        (Dict): Dictionary of sensor rigs indexed by sensor name.
    """
    # Parse the properties from the rig file
    sensors = rig["rig"]["sensors"]

    sensors_dict = {sensor["name"]: sensor for sensor in sensors}
    return sensors_dict


def sensor_to_rig(sensor: dict) -> np.ndarray | None:
    """Obtain sensor to rig transformation matrix."""
    sensor_name = sensor["name"]
    sensor_to_FLU = get_sensor_to_sensor_flu(sensor_name)

    if "nominalSensor2Rig_FLU" not in sensor:
        # Some sensors (like CAN sensors) don't have an associated sensorToRig
        return None

    nominal_T = sensor["nominalSensor2Rig_FLU"]["t"]
    nominal_R = sensor["nominalSensor2Rig_FLU"]["roll-pitch-yaw"]

    correction_T = np.zeros(3, dtype=np.float32)
    correction_R = np.zeros(3, dtype=np.float32)

    if "correction_rig_T" in sensor.keys():
        correction_T = sensor["correction_rig_T"]

    if "correction_sensor_R_FLU" in sensor.keys():
        assert "roll-pitch-yaw" in sensor["correction_sensor_R_FLU"].keys(), str(sensor["correction_sensor_R_FLU"])
        correction_R = sensor["correction_sensor_R_FLU"]["roll-pitch-yaw"]

    nominal_R = rotation_utils.euler_2_so3(nominal_R)
    correction_R = rotation_utils.euler_2_so3(correction_R)

    R = nominal_R @ correction_R
    T = np.array(nominal_T, dtype=np.float32) + np.array(correction_T, dtype=np.float32)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R
    transform[:3, 3] = T

    sensor_to_rig = transform @ sensor_to_FLU

    return sensor_to_rig


def traj2camera(
    points: np.ndarray,
    camera_xyz: np.ndarray,
    camera_quat: np.ndarray,
    camera_to_rig: np.ndarray,
    camera_intrinsic: "ndas_camera_model.FThetaCamera",
):
    """Project the trajectory points to the camera image.

    Args:
        points: (N, 3) The trajectory points in global coordinate.
        camera_xyz: (3,) The camera position in global coordinate.
        camera_quat: (4,) The camera quaternion in global coordinate.
        camera_to_rig: (4, 4) The camera extrinsics.
        camera_intrinsic: The camera intrinsics.

    Returns:
        uv: The projected points in pixel coordinate.
        z: The depth of the points.
    """
    ego_quat = Quaternion(camera_quat)
    points = (points - camera_xyz) @ ego_quat.inverse.rotation_matrix.T
    # closed-form inversion
    rig_to_camera = np.zeros_like(camera_to_rig)
    rig_to_camera[:3, :3] = camera_to_rig[:3, :3].T
    rig_to_camera[:3, 3] = -camera_to_rig[:3, :3].T @ camera_to_rig[:3, 3]
    camera_xyz = (points @ rig_to_camera[:3, :3].T) + rig_to_camera[:3, 3]
    uv = camera_intrinsic.ray2pixel(camera_xyz)
    z = camera_xyz[:, 2]
    return uv, z


def box2camera(
    points: np.ndarray,
    camera_to_rig: np.ndarray,
    camera_intrinsic: "ndas_camera_model.FThetaCamera",
) -> tuple[np.ndarray, np.ndarray]:
    """Project the 3D bounding box to the camera image.

    Args:
        points: (N, 3) The corner points of 3D bounding box in global coordinate.
        camera_to_rig: (4, 4) The camera extrinsics.
        camera_intrinsic: The camera intrinsics.

    Returns:
        uv: The projected points in pixel coordinate.
        z: The depth of the points.
    """
    # closed-form inversion
    rig_to_camera = np.zeros_like(camera_to_rig)
    rig_to_camera[:3, :3] = camera_to_rig[:3, :3].T
    rig_to_camera[:3, 3] = -camera_to_rig[:3, :3].T @ camera_to_rig[:3, 3]
    camera_xyz = (points @ rig_to_camera[:3, :3].T) + rig_to_camera[:3, 3]
    uv = camera_intrinsic.ray2pixel(camera_xyz)
    z = camera_xyz[:, 2]
    return uv, z


def get_traj_proj_on_image(
    traj_xyz: np.ndarray,
    camera_xyz: np.ndarray,
    camera_quat: np.ndarray,
    camera_intrinsic: "ndas_camera_model.FThetaCamera",
    camera_extrinsics: np.ndarray,
    maglev_conf: dict,
    output_width: int | None = None,
    output_height: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the projection of trajectory points onto an image.

    Args:
        traj_xyz (np.ndarray): (N, 3) Array of trajectory points in 3D space.
        camera_xyz (np.ndarray): Camera position in 3D space.
        camera_quat (np.ndarray): Camera orientation as a quaternion.
        camera_intrinsic (ndas_camera_model.FThetaCamera): Camera intrinsic data.
        camera_extrinsics (np.ndarray): Camera extrinsics matrix.
        maglev_conf (dict): Configuration parameters for Maglev.
        output_width (int): Width of the output image. TODO deprecate this for maglev_conf["size_w"]
        output_height (int): Height of the output image. TODO deprecate this for
            maglev_conf["size_h"]

    Returns:
        np.ndarray: Array of UV coordinates representing the projection of trajectory points on
            the image.
        np.ndarray: (N,) Array of boolean values indicating the validity of the projection.
    """
    uv, z = traj2camera(
        points=traj_xyz,
        camera_xyz=camera_xyz,
        camera_quat=camera_quat,
        camera_to_rig=camera_extrinsics,
        camera_intrinsic=camera_intrinsic,
    )
    # account for the cropping and resizing
    xbias, ybias, facw, _ = compute_preprocessing_transform(camera_intrinsic, maglev_conf)
    uv[:, 0] -= xbias
    uv[:, 1] -= ybias
    uv /= facw

    valid = (
        (z > 0)
        & (uv[:, 0] < maglev_conf["size_w"] - 0.5)
        & (uv[:, 0] > -0.5)
        & (uv[:, 1] < maglev_conf["size_h"] - 0.5)
        & (uv[:, 1] > -0.5)
    )
    uv = uv[valid]
    return uv, valid


def project_points_to_image(
    points: np.ndarray,
    camera_to_rig: np.ndarray,
    camera_intrinsic: "ndas_camera_model.FThetaCamera",
    maglev_processing_config: Mapping[str, float],
    output_width: int = sys.maxsize,
    output_height: int = sys.maxsize,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D points to an image.

    Args:
        points: 3D points to project. Shape (N, 3).
        camera_to_rig: Coordinate transformation from camera to rig frame. Shape (4, 4).
        camera_intrinsic: Camera intrinsic matrix.
        maglev_processing_config: Information about the preprocessing applied to the image, e.g.,
            cropping and resizing.
        output_width: Width of the output image.
        output_height: Height of the output image.

    Returns:
        uv: The UV (i.e. image) coordinates of the projected points.
        z: The depth of the projected points in camera space.
        mask: A boolean mask indicating if the projection is valid.
    """
    uv, z = box2camera(points, camera_to_rig, camera_intrinsic)

    # account for the cropping and resizing and apply the processing corrections
    xbias, ybias, facw, _ = compute_preprocessing_transform(camera_intrinsic, maglev_processing_config)
    uv[:, 0] -= xbias
    uv[:, 1] -= ybias
    uv /= facw

    xmask = np.logical_and(-0.5 < uv[:, 0], uv[:, 0] < output_width - 0.5)
    ymask = np.logical_and(-0.5 < uv[:, 1], uv[:, 1] < output_height - 0.5)
    mask = np.logical_and.reduce([xmask, ymask, z > 0])
    return uv, z, mask


def compute_preprocessing_transform(
    camera_intrinsic: "ndas_camera_model.FThetaCamera", maglev_conf: dict
) -> tuple[float, float]:
    """Pre-processing due to crop and resize.

    Args:
        camera_intrinsic: The FTheta camera intrinsic.
        maglev_conf: The maglev configuration from generator_config.yaml.

    Returns:
        xbias: The x-bias due to the crop.
        ybias: The y-bias due to the crop.
        facw: The factor to rescale width.
        fach: The factor to rescale height.
    """
    og_width = int(camera_intrinsic.width)
    og_height = int(camera_intrinsic.height)

    # TODO in the calibration comments, it seems like the
    # original size is 3840x2160? But actually sometimes the
    # calibration says the original size is 3848x2168

    # the pre-processing steps are
    # 1) bottom crop
    # 2) symmetric left-right crop
    # 3) rescaling

    # for transformation parameters, the bottom crop
    # only changes the extent. So in terms of the
    # change to the coordinate frame, we only take
    # into account the left-right crop and rescaling.

    xbias = (og_width - maglev_conf["crop_w"]) / 2
    ybias = (og_height - maglev_conf["crop_h"]) / 2
    # make sure resize is equal
    facw = maglev_conf["crop_w"] / maglev_conf["size_w"]
    fach = maglev_conf["crop_h"] / maglev_conf["size_h"]
    assert facw == fach, f"{facw} {fach}"
    return xbias, ybias, facw, fach


def get_video_parameters(
    camkeys: Iterable[str],
    datapath: str | None = None,
) -> dict[str, int]:
    """Load the pre-processing config file for the dataset.

    Currently we assert that all cameras use the same parameters with the only difference of
    size_w/h, we could loosen that in the future.
    """
    collect = {"crop_h": [], "crop_w": [], "size_h": [], "size_w": [], "fps": []}
    if datapath is None:
        datapath = __file__.replace("transformation.py", "generator_config.yaml")
    with open(datapath) as reader:
        conf = yaml.safe_load(reader)

    for camkey in camkeys:
        for key in collect:
            collect[key].append(conf["data"][camkey][key])
    # make sure they're all the same
    for key in collect:
        assert len(set(collect[key])) == 1, collect[key]

    # select the first one since they're all the same
    out = {k: v[0] for k, v in collect.items()}

    return out
