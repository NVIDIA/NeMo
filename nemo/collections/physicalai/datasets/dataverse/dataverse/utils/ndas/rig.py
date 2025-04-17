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

"""AutoNet v2 rig helper functions."""

import typing

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def get_sensor_to_sensor_flu(sensor: str):
    """Compute a rotation transformation matrix that rotate sensor to Front-Left-Up format.

    Args:
        sensor (str): sensor name.

    Returns:
        np.array: the resulting rotation matrix.
    """
    if "cam" in sensor:
        rot = [
            [0.0, 0.0, 1.0, 0.0],  #
            [-1.0, 0.0, 0.0, 0.0],  #
            [0.0, -1.0, 0.0, 0.0],  #
            [0.0, 0.0, 0.0, 1.0],  #
        ]
    else:
        rot = [
            [1.0, 0.0, 0.0, 0.0],  #
            [0.0, 1.0, 0.0, 0.0],  #
            [0.0, 0.0, 1.0, 0.0],  #
            [0.0, 0.0, 0.0, 1.0],  #
        ]

    return np.asarray(rot, dtype=np.float32)


def transform_from_eulers(rpy_deg: typing.Sequence[float], translation: typing.Sequence[float]):
    """Create a 4x4 rigid transformation matrix given euler angles and translation.

    Args:
        rpy_deg (typing.Sequence[float]): Euler angles as roll, pitch, yaw in degrees.
        translation (typing.Sequence[float]): x, y, z translation.

    Returns:
        np.array: the constructed transformation matrix.
    """
    transform = np.eye(4)
    transform[:3, :3] = R.from_euler(seq="xyz", angles=rpy_deg, degrees=True).as_matrix()
    transform[:3, 3] = translation

    return transform.astype(np.float32)


def apply_transform(
    transform: typing.Union[np.array, torch.Tensor],
    pts: typing.Union[np.array, torch.Tensor],
):
    """Applies the given transform to the given points in 3D space.

    Args:
        transform (typing.Union[np.array, torch.Tensor]): the transform to apply.
            Must be a 4x4 or 3x4 matrix.
        pts (typing.Union[np.array, torch.Tensor]): the 3D points to trasform (shape (n, 3))

    Returns:
        typing.Union[np.array, torch.Tensor]: the transformed points.
    """
    assert transform.shape in (
        (4, 4),
        (3, 4),
    ), "Only 4x4 and 3x4 transformation matrices are supported."
    assert type(transform) in (
        torch.Tensor,
        np.ndarray,
    ), "Only torch tensors and numpy arrays are supported."
    assert type(pts) in (
        torch.Tensor,
        np.ndarray,
    ), "Only torch tensors and numpy arrays are supported."

    if pts.ndim == 1:
        if isinstance(pts, np.ndarray):  # Numpy
            pts_ = pts[np.newaxis, :]
        else:  # Torch
            pts_ = pts.unsqueeze(0)
    else:
        pts_ = pts

    assert pts_.shape[1] == 3, "Input points must be of shape n x 3."

    # Extract R and T components
    R = transform[:3, :3]
    T = transform[:3, 3:]
    # Apply!
    result = R @ pts_.T + T
    return result.T.squeeze()


def parse_sensor_to_rig(sensor: dict, ignore_correction_T: bool = False):
    """Parses the provided rig-style transform dictionary into sensor to rig matrices.

    Args:
        sensor: dictionary for a sensor read from a calibration file
        ignore_correction_T (bool): if `True`, the correction translation values in the rig will
            be ignored and set to zero.

    Returns:
        sensor_to_rig (np.array): the 4x4 sensor to rig transformation matrix with the correction
            factors applied.
        nominal_sensor_to_rig (np.array) the nominal 4x4 sensor to rig transformation matrix
            without the correction factors.
    """
    sensor_to_FLU = get_sensor_to_sensor_flu(sensor["name"])

    nominal_FLU_to_rig = transform_from_eulers(
        sensor["nominalSensor2Rig_FLU"]["roll-pitch-yaw"],
        sensor["nominalSensor2Rig_FLU"]["t"],
    )

    # Use or ignore the correction for translation
    if "correction_rig_T" in sensor.keys() and not ignore_correction_T:
        correction_T = sensor["correction_rig_T"]
    else:
        correction_T = np.zeros(3, dtype=np.float32)

    correction_transform = transform_from_eulers(sensor["correction_sensor_R_FLU"]["roll-pitch-yaw"], correction_T)

    sensor_to_rig = nominal_FLU_to_rig @ correction_transform @ sensor_to_FLU
    nominal_sensor_to_rig = nominal_FLU_to_rig @ sensor_to_FLU

    return sensor_to_rig, nominal_sensor_to_rig


def get_rig_transform(sensor: dict, rig2sensor: bool):
    """Obtain the camera to rig coordinate transform.

    Args:
        sensor: dictionary for a sensor read from a calibration file
        rig2sensor: When True, returns the rig to sensor transform, returns sensor to rig otherwise.
    """
    # Calculate:
    # nominal_FLU_to_rig @ correction_transform @ sensor_to_FLU
    tf, _ = parse_sensor_to_rig(sensor)
    FLU_to_sensor = np.linalg.inv(get_sensor_to_sensor_flu(sensor["name"]))
    tf = tf @ FLU_to_sensor

    if rig2sensor:
        tf = np.linalg.inv(tf)
    return tf
