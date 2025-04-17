# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import torch

from nemo.collections.diffusion.data import camera
from nemo.collections.diffusion.data.camera import get_center_and_ray


def plucker_coordinates(pose: torch.tensor, intr: torch.tensor, width: int, height: int):
    """Return plücker coordinates from pose and intrinsics. Plücker coordinates are defined as
    [(rx,ry,rz),(rx,ry,rz)x(cx,cy,cz)] where (cx,cy,cz) is the camera origin
    and (rx,ry,rz) is the direction of the ray.
    Plücker coordinates are used to represent a line in 3D space.

    Useful references:
    - https://www.euclideanspace.com/maths/geometry/elements/line/plucker/index.htm


    Args:
        pose (torch.tensor): Extrinsics [B,3,4]
        intr (torch.tensor): Intrinsics [B,3,3]
        width (int): Image width
        height (int): Image height

    Returns:
        torch.tensor: plücker coordinates
    """
    center, ray = get_center_and_ray(pose, intr, [height, width])  # [B,HW,3]
    ray = ray / torch.norm(ray, dim=-1, keepdim=True)  # [B,HW,3], unit length
    plucker_coords = torch.cat([torch.cross(center, ray, dim=-1), ray], dim=-1)  # [B,HW,6]
    return plucker_coords


def get_relative_pose(pose_list: list[torch.Tensor | np.ndarray]) -> list[np.ndarray]:
    """
    Convert a list of 3x4 world to camera pose to relative pose to the first frame
    Args:
        pose_list (list[torch.Tensor | np.ndarray]): List of 3x4 world to camera pose
    Returns:
        ret_poses (list[np.ndarray]): List of relative poses
    """
    if isinstance(pose_list[0], np.ndarray):
        poses = torch.from_numpy(np.stack(list(pose_list), axis=0))  # [N,3,4]
    else:
        poses = torch.stack(list(pose_list), dim=0)  # [N,3,4]
    pose_0 = poses[:1]
    pose_0_inv = camera.pose.invert(pose_0)
    rel_poses = camera.pose.compose_pair(pose_0_inv, poses)
    # Homogeneous form (4x4)
    rel_poses_4x4 = torch.eye(4).repeat(len(rel_poses), 1, 1)
    rel_poses_4x4[:, :3, :] = rel_poses
    return rel_poses_4x4.numpy()


def estimate_pose_list_to_plucker_embedding(
    pose_list: list,
    latent_compression_ratio_h: int,
    latent_compression_ratio_w: int,
    image_size: torch.tensor,
    use_relative_pose: bool = True,
) -> torch.tensor:
    """
    Convert a list of pose to plücker coordinates
    Args:
        pose_list (list): List of pose, each element is a dict with keys "intrinsics", "rotation", "translation"
            e.g. {'intrinsics': [[0.4558800160884857, 0.0, 0.5], [0.0, 0.8124798536300659, 0.5], [0.0, 0.0, 0.0]],
            'rotation': [[0.5067835450172424, 0.4129045605659485, -0.7567564249038696],
                         [-0.41741496324539185, 0.8855977654457092, 0.20366966724395752],
                         [0.7542779445648193, 0.21266502141952515, 0.6211589574813843]
                        ],
            'translation': [1.5927585363388062, -0.41845059394836426, 0.6559827327728271]}
        image_size (torch.tensor): Image size of the current video after processing, the input is
            h_after_padded, w_after_padded, h_after_resize, w_after_resize,
            e.g. [ 704., 1280.,  704., 1252.] for input with raw shape [720, 1280]
        latent_compression_ratio_h (int): compression height of the plücker embedding image
        latent_compression_ratio_w (int): compression width of the plücker embedding image
        use_relative_pose (bool): Whether to use relative pose
    Returns:
        plücker_coords (torch.tensor): Plücker embedding of shape [num_frame, HW, 6]
    """
    num_frame = len(pose_list)
    # e.g. 704, 1280, 704, 1252
    h_after_padded, w_after_padded, h_after_resize, w_after_resize = image_size
    H = h_after_padded.item() // latent_compression_ratio_h  # e.g. 704 / 8 = 88
    W = w_after_padded.item() // latent_compression_ratio_w  # e.g. 1280 / 8 = 160
    ratio_w = w_after_resize.item() / w_after_padded.item()
    ratio_h = h_after_resize.item() / h_after_padded.item()

    H = int(H)
    W = int(W)
    # Compute mv_intr_denormalized
    mv_intr_denormalized = []
    for p in pose_list:
        intrinsic = torch.tensor(p["intrinsics"])
        intrinsic[2, 2] = 1
        intrinsic[0, :] *= W * ratio_w
        intrinsic[1, :] *= H * ratio_h
        mv_intr_denormalized.append(intrinsic)

    mv_pose = [
        torch.cat([torch.tensor(p["rotation"]), torch.tensor(p["translation"]).unsqueeze(1)], dim=1) for p in pose_list
    ]

    # Convert to pose relative to the first frame
    if use_relative_pose:
        mv_pose = get_relative_pose(mv_pose)
    mv_intr_denormalized = torch.stack(mv_intr_denormalized)
    mv_pose = torch.tensor(np.stack(mv_pose))
    mv_pose = mv_pose[:, :3]  # B*N,3,4
    mv_intr_denormalized = mv_intr_denormalized.view(num_frame, 3, 3)  # B*N,3,3

    # plucker coordinates to encode pose
    plucker_coords = plucker_coordinates(mv_pose, mv_intr_denormalized, W, H)  # [B,HW,6]

    return plucker_coords, H, W


def normalize_camera_trajectory_to_unit_sphere(pose_list: list[dict]) -> None:
    """
    Normalize the camera trajectory to fit within a unit sphere.
    This function takes a list of camera poses, each represented as a dictionary with a "translation" key,
    and normalizes the translation vectors such that the maximum distance between any two cameras is 1.
    The normalization is done in-place.
    Args:
        pose_list (list[dict]): A list of dictionaries, where each dictionary contains a "translation" key
                                with a list or array of three floats representing the camera translation vector.
    Returns:
        None
    """
    translation = np.array([pose["translation"] for pose in pose_list])  # [N,3]

    # Find the max distance between any two cameras. It is equivalent to the max distance of translation vectors.
    def _longest_distance(points):
        # Compute the pairwise distances.
        diff = points[:, None, :] - points[None, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        # Find the maximum distance
        max_distance = np.max(distances)
        return max_distance

    max_distance = _longest_distance(translation)
    for pose in pose_list:
        trans = np.array(pose["translation"])
        trans /= max_distance
        pose["translation"] = trans.tolist()
